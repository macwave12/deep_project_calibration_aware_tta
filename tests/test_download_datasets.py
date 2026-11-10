"""Offline tests for scripts/download_datasets.py.

A tiny fake copy of every source archive (same filenames and internal structure as the
real ones) is served from a local ``http.server`` with Range support and a fake Google
Drive endpoint that answers with the confirm interstitial first. No real network.
"""
import dataclasses
import importlib.util
import io
import json
import os
import re
import sys
import tarfile
import threading
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "drive"

_spec = importlib.util.spec_from_file_location("download_datasets", REPO / "scripts" / "download_datasets.py")
dd = importlib.util.module_from_spec(_spec)
sys.modules["download_datasets"] = dd  # dataclasses need the module registered
_spec.loader.exec_module(dd)


# --------------------------------------------------------------------------- fake server


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self._serve(head=True)

    def do_GET(self):
        self._serve(head=False)

    def _serve(self, head):
        srv = self.server
        u = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        srv.log.append((self.command, u.path, self.headers.get("Range")))
        if u.path == "/uc":  # fake drive.google.com/uc
            fid = q.get("id", "")
            if fid in srv.quota_ids:
                return self._send(200, FIXTURES.joinpath("quota_exceeded.html").read_bytes(), "text/html")
            if fid in srv.direct_ids:
                return self._send(200, srv.drive[fid], "application/json")
            page = FIXTURES.joinpath("virus_scan_form.html").read_text(encoding="utf-8")
            page = page.replace("https://drive.usercontent.google.com/download",
                                f"http://127.0.0.1:{srv.server_port}/download")
            page = re.sub(r'name="id" value="[^"]*"', f'name="id" value="{fid}"', page)
            return self._send(200, page.encode(), "text/html; charset=utf-8")
        if u.path == "/download":  # fake drive.usercontent.google.com/download
            if q.get("confirm") != "t" or q.get("id") not in srv.drive:
                return self._send(400, b"bad confirm", "text/plain")
            return self._send(200, srv.drive[q["id"]], "application/json")
        if u.path == "/short":  # announces more bytes than it sends
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"x" * 10)
            self.close_connection = True
            return
        data = srv.files.get(u.path)
        if data is None:
            return self._send(404, b"not found", "text/plain")
        rng = self.headers.get("Range")
        if rng and srv.honour_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.end_headers()
                return
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.log, srv.files, srv.drive = [], {}, {}
    srv.quota_ids, srv.direct_ids, srv.honour_range = set(), set(), True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    srv.base = f"http://127.0.0.1:{srv.server_port}"
    yield srv
    srv.shutdown()
    srv.server_close()


# --------------------------------------------------------------------------- fake archives


def _tar(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _split(prefix_cls):
    """CoOp-style split JSON: {split: [[relpath, label, classname], ...]}."""
    out = {}
    for split, n in (("train", 3), ("val", 2), ("test", 2)):
        out[split] = [[f"{c}/{c}_{split}{i}.jpg", li, c] for li, c in enumerate(prefix_cls)
                      for i in range(n)]
    return out


def _imgs(prefix, split_obj):
    return {f"{prefix}{e[0]}": b"jpg" for s in split_obj.values() for e in s}


FAKE_COUNTS = {"dtd": 4, "flower102": 4, "pets": 4, "eurosat": 4, "aircraft": 2}


def make_fake_world(srv):
    """Register fake archives + Drive files on ``srv``; return a SOURCES-like dict that
    keeps the real specs (folders, moves, required files) but points at the server."""
    cls = ["banded", "woven"]
    dtd = _split(cls)
    flw = {k: [[e[0].split("/")[1], e[1], e[2]] for e in v] for k, v in _split(cls).items()}
    pets = {k: [[e[0].split("/")[1], e[1], e[2]] for e in v] for k, v in _split(cls).items()}
    euro = _split(["AnnualCrop", "Forest"])
    files = {
        "dtd-r1.0.1.tar.gz": _tar({**_imgs("dtd/images/", dtd), "dtd/labels/labels_joint_anno.txt": b"x",
                                   "dtd/imdb/imdb.mat": b"m"}),
        "102flowers.tgz": _tar(_imgs("jpg/", flw)),
        "imagelabels.mat": b"labels", "setid.mat": b"setid",
        "images.tar.gz": _tar(_imgs("images/", pets)),
        "annotations.tar.gz": _tar({"annotations/list.txt": b"l"}),
        "EuroSAT_RGB.zip": _zip(_imgs("EuroSAT_RGB/", euro)),
    }
    variants = ["707-320", "A300B4"]
    air = {"fgvc-aircraft-2013b/README.md": b"r",
           "fgvc-aircraft-2013b/data/variants.txt": ("\n".join(variants) + "\n").encode(),
           "fgvc-aircraft-2013b/data/families.txt": b"Boeing 707\n"}
    for split, ids in (("train", ["0001", "0002"]), ("val", ["0003", "0004"]),
                       ("test", ["0005", "0006"]), ("trainval", ["0001", "0002", "0003", "0004"])):
        lines = "".join(f"{i} {variants[k % 2]}\n" for k, i in enumerate(ids))
        air[f"fgvc-aircraft-2013b/data/images_variant_{split}.txt"] = lines.encode()
    for i in range(1, 7):
        air[f"fgvc-aircraft-2013b/data/images/000{i}.jpg"] = b"jpg"
    files["fgvc-aircraft-2013b.tar.gz"] = _tar(air)
    for name, data in files.items():
        srv.files[f"/files/{name}"] = data

    drive_json = {"split_zhou_DescribableTextures.json": dtd, "split_zhou_OxfordFlowers.json": flw,
                  "split_zhou_OxfordPets.json": pets, "split_zhou_EuroSAT.json": euro,
                  "cat_to_name.json": {"1": "pink primrose"}}
    sources = {}
    for key, spec in dd.SOURCES.items():
        archives = []
        for a in spec.archives:
            urls = [f"{srv.base}/files/{a.filename}"]
            if key == "eurosat":  # first mirror is down -> exercises the fallback
                urls.insert(0, f"{srv.base}/files/missing-mirror.zip")
            archives.append(dataclasses.replace(a, urls=urls))
        for df in spec.drive:
            srv.drive[df.file_id] = json.dumps(drive_json[df.dest]).encode()
        sources[key] = dataclasses.replace(spec, archives=archives, expected_test=FAKE_COUNTS[key])
    return sources


def run_main(srv, sources, root, *extra):
    return dd.main(["--data-root", str(root), *extra], sources=sources,
                   drive_url=srv.base + "/uc?export=download&id={id}", drive_fallback=None)


# --------------------------------------------------------------------------- pipeline


def test_full_pipeline_produces_documented_layout(server, tmp_path):
    sources = make_fake_world(server)
    root = tmp_path / "data_root"
    assert run_main(server, sources, root) == 0

    expect = [
        "dtd/images/banded/banded_test0.jpg", "dtd/labels", "dtd/imdb",
        "dtd/split_zhou_DescribableTextures.json",
        "oxford_flowers/jpg", "oxford_flowers/imagelabels.mat", "oxford_flowers/setid.mat",
        "oxford_flowers/cat_to_name.json", "oxford_flowers/split_zhou_OxfordFlowers.json",
        "oxford_pets/images", "oxford_pets/annotations/list.txt", "oxford_pets/split_zhou_OxfordPets.json",
        "eurosat/2750/AnnualCrop/AnnualCrop_test0.jpg", "eurosat/split_zhou_EuroSAT.json",
        "fgvc_aircraft/images/0005.jpg", "fgvc_aircraft/variants.txt", "fgvc_aircraft/families.txt",
        "fgvc_aircraft/images_variant_train.txt", "fgvc_aircraft/images_variant_val.txt",
        "fgvc_aircraft/images_variant_test.txt", "fgvc_aircraft/images_variant_trainval.txt",
        "_splits/dtd/split_zhou_DescribableTextures.json",
    ]
    for rel in expect:
        assert (root / rel).exists(), rel
    assert not (root / "eurosat" / "EuroSAT_RGB").exists()  # renamed to 2750
    assert not (root / "fgvc_aircraft" / "data").exists()  # flattened
    assert not (root / "_staging").exists()
    # archives deleted by default (plain .mat downloads stay, they are tiny)
    left = sorted(p.name for p in (root / "_archives").iterdir())
    assert left == ["imagelabels.mat", "setid.mat"]

    report = {r["dataset"]: r for r in json.loads((root / "_state" / "verify.json").read_text(encoding="utf-8"))}
    assert all(r["ok"] for r in report.values())
    assert report["eurosat"]["splits"]["test"] == {"count": 4, "missing": 0, "reference": 8100,
                                                   "first_missing": []}
    assert report["aircraft"]["splits"]["val"]["count"] == 2
    assert report["dtd"]["splits"]["train"]["count"] == 6


def test_layout_matches_upstream_loader_paths(server, tmp_path):
    """The produced tree is what data/fewshot_datasets.py actually reads."""
    sources = make_fake_world(server)
    root = tmp_path / "dr"
    assert run_main(server, sources, root, "--datasets", "eurosat", "aircraft") == 0
    from data.datautils import ID_to_DIRNAME
    assert ID_to_DIRNAME["eurosat"] == sources["eurosat"].folder
    assert ID_to_DIRNAME["aircraft"] == sources["aircraft"].folder
    src = (REPO / "data" / "fewshot_datasets.py").read_text(encoding="utf-8")
    assert '"eurosat": ["2750"' in src
    assert "images_variant_{:s}.txt" in src


def test_rerun_is_idempotent_and_needs_no_network(server, tmp_path):
    sources = make_fake_world(server)
    root = tmp_path / "dr"
    assert run_main(server, sources, root, "--keep-archives") == 0
    n = len(server.log)
    assert run_main(server, sources, root, "--keep-archives") == 0
    assert len(server.log) == n, server.log[n:]
    server.shutdown()  # completely offline now
    assert run_main(server, sources, root) == 0


def test_interrupted_layout_is_redone_from_kept_archive(server, tmp_path):
    sources = make_fake_world(server)
    root = tmp_path / "dr"
    assert run_main(server, sources, root, "--datasets", "eurosat", "--keep-archives") == 0
    import shutil
    shutil.rmtree(root / "eurosat" / "2750")  # e.g. antivirus quarantined it / user deleted
    n = len(server.log)
    assert run_main(server, sources, root, "--datasets", "eurosat") == 0
    assert (root / "eurosat" / "2750" / "Forest" / "Forest_val1.jpg").is_file()
    assert len(server.log) == n  # re-laid out from the kept archive, no re-download


def test_subset_flag_only_touches_requested(server, tmp_path):
    sources = make_fake_world(server)
    root = tmp_path / "dr"
    assert run_main(server, sources, root, "--datasets", "dtd") == 0
    assert (root / "dtd").is_dir() and not (root / "oxford_pets").exists()
    assert all("dtd" in p or p in ("/uc", "/download") for _, p, _ in server.log)


def test_manual_archive_in_archives_dir_is_used(server, tmp_path):
    sources = make_fake_world(server)
    root = tmp_path / "dr"
    (root / "_archives").mkdir(parents=True)
    (root / "_archives" / "EuroSAT_RGB.zip").write_bytes(server.files["/files/EuroSAT_RGB.zip"])
    assert run_main(server, sources, root, "--datasets", "eurosat") == 0
    assert not [e for e in server.log if e[0] == "GET" and e[1].startswith("/files/")]


def test_quota_error_fails_cleanly(server, tmp_path, capsys):
    sources = make_fake_world(server)
    server.quota_ids.add(sources["pets"].drive[0].file_id)
    assert run_main(server, sources, tmp_path / "dr", "--datasets", "pets") == 1
    out = capsys.readouterr().out
    assert "quota exceeded" in out and "NOT READY: pets" in out


# --------------------------------------------------------------------------- downloads


def test_resume_uses_range_and_reassembles(server, tmp_path):
    data = os.urandom(3 * dd.CHUNK + 123)
    server.files["/files/big.bin"] = data
    dest = tmp_path / "big.bin"
    Path(str(dest) + ".part").write_bytes(data[: dd.CHUNK + 7])
    assert dd.download_file(server.base + "/files/big.bin", str(dest)) == len(data)
    assert dest.read_bytes() == data
    assert server.log[-1][2] == f"bytes={dd.CHUNK + 7}-"
    assert not Path(str(dest) + ".part").exists()


def test_server_ignoring_range_restarts_cleanly(server, tmp_path):
    server.honour_range = False
    data = b"abcdef" * 1000
    server.files["/files/f.bin"] = data
    dest = tmp_path / "f.bin"
    Path(str(dest) + ".part").write_bytes(b"GARBAGE")
    dd.download_file(server.base + "/files/f.bin", str(dest))
    assert dest.read_bytes() == data


def test_complete_part_file_with_416(server, tmp_path):
    data = b"z" * 5000
    server.files["/files/f.bin"] = data
    dest = tmp_path / "f.bin"
    Path(str(dest) + ".part").write_bytes(data)
    dd.download_file(server.base + "/files/f.bin", str(dest))
    assert dest.read_bytes() == data


def test_truncated_transfer_is_an_error(server, tmp_path):
    with pytest.raises(dd.DownloadError):
        dd.download_file(server.base + "/short", str(tmp_path / "s.bin"), retries=2)
    assert not (tmp_path / "s.bin").exists()


def test_404_is_not_retried(server, tmp_path):
    with pytest.raises(dd.DownloadError):
        dd.download_file(server.base + "/files/nope", str(tmp_path / "n"), retries=5)
    assert len([e for e in server.log if e[1] == "/files/nope"]) == 1


def test_fetch_clip_checks_sha(server, tmp_path):
    import hashlib
    blob = b"fake clip weights"
    server.files["/clip/ViT-B-16.pt"] = blob
    good = hashlib.sha256(blob).hexdigest()
    p = dd.fetch_clip(str(tmp_path), urls=[server.base + "/clip/ViT-B-16.pt"], sha256=good)
    assert Path(p).read_bytes() == blob and Path(p).name == "ViT-B-16.pt"
    with pytest.raises(dd.DownloadError, match="sha256"):
        dd.fetch_clip(str(tmp_path / "other"), urls=[server.base + "/clip/ViT-B-16.pt"], sha256="0" * 64)


# --------------------------------------------------------------------------- Google Drive


FID = "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x"


def test_parse_drive_form_interstitial():
    url = dd.parse_drive_confirm(FIXTURES.joinpath("virus_scan_form.html").read_text(encoding="utf-8"), FID)
    u = urllib.parse.urlsplit(url)
    q = dict(urllib.parse.parse_qsl(u.query))
    assert f"{u.scheme}://{u.netloc}{u.path}" == "https://drive.usercontent.google.com/download"
    assert q == {"id": FID, "export": "download", "confirm": "t",
                 "uuid": "5f1c2d3e-aaaa-bbbb-cccc-0123456789ab"}


def test_parse_drive_legacy_confirm_link():
    fid = "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o"
    url = dd.parse_drive_confirm(FIXTURES.joinpath("legacy_confirm_link.html").read_text(encoding="utf-8"), fid)
    assert url == f"https://drive.google.com/uc?export=download&confirm=Xy3_9-Qz&id={fid}"


def test_parse_drive_cookie_token():
    url = dd.parse_drive_confirm("<html><body>scan warning</body></html>", FID,
                                 cookies={"download_warning_123_abc": "TOK"})
    assert "confirm=TOK" in url and FID in url


def test_parse_drive_quota_and_unknown():
    with pytest.raises(dd.DriveQuotaError):
        dd.parse_drive_confirm(FIXTURES.joinpath("quota_exceeded.html").read_text(encoding="utf-8"), FID)
    with pytest.raises(dd.DownloadError, match="unrecognised"):
        dd.parse_drive_confirm("<html><body>something else</body></html>", FID)


def test_drive_download_follows_interstitial(server, tmp_path):
    server.drive[FID] = json.dumps({"train": [], "val": [], "test": [["a.jpg", 0, "a"]]}).encode()
    dest = tmp_path / "s.json"
    dd.drive_download(FID, str(dest), url_template=server.base + "/uc?export=download&id={id}",
                      fallback_template=None, json_keys=("train", "val", "test"))
    assert json.loads(dest.read_text(encoding="utf-8"))["test"] == [["a.jpg", 0, "a"]]
    assert [p for _, p, _ in server.log] == ["/uc", "/download"]


def test_drive_download_direct_and_bad_json(server, tmp_path):
    server.drive[FID] = b'{"train": []}'
    server.direct_ids.add(FID)
    tmpl = server.base + "/uc?export=download&id={id}"
    dd.drive_download(FID, str(tmp_path / "a.json"), url_template=tmpl, fallback_template=None)
    with pytest.raises(dd.DownloadError, match="lacks keys"):
        dd.drive_download(FID, str(tmp_path / "b.json"), url_template=tmpl, fallback_template=None,
                          json_keys=("test",))
    assert not (tmp_path / "b.json").exists()


# --------------------------------------------------------------------------- layout/verify units


def test_eurosat_mirror_with_2750_top_dir(tmp_path):
    arch = dd.SOURCES["eurosat"].archives[0]
    z = tmp_path / "EuroSAT.zip"
    z.write_bytes(_zip({"2750/Forest/Forest_1.jpg": b"j"}))
    dd.layout_archive(arch, str(z), str(tmp_path / "eurosat"), str(tmp_path / "stg"))
    assert (tmp_path / "eurosat" / "2750" / "Forest" / "Forest_1.jpg").is_file()


def test_extract_refuses_path_traversal(tmp_path):
    t = tmp_path / "evil.tar.gz"
    t.write_bytes(_tar({"../escape.txt": b"x"}))
    with pytest.raises(dd.DownloadError, match="escapes"):
        dd.extract_archive(str(t), str(tmp_path / "out"))
    assert not (tmp_path / "escape.txt").exists()


def test_verify_reports_missing_files_and_wrong_counts(server, tmp_path):
    sources = make_fake_world(server)
    root = tmp_path / "dr"
    assert run_main(server, sources, root, "--datasets", "dtd", "aircraft") == 0
    os.remove(root / "dtd" / "images" / "woven" / "woven_test1.jpg")
    r = dd.verify_dataset(sources["dtd"], str(root))
    assert not r["ok"] and r["splits"]["test"]["missing"] == 1
    r = dd.verify_dataset(dataclasses.replace(sources["aircraft"], expected_test=3333), str(root))
    assert not r["ok"] and any("contract 3333" in p for p in r["problems"])
    with open(root / "fgvc_aircraft" / "images_variant_test.txt", "a", encoding="utf-8") as f:
        f.write("0005 NotAVariant\n")
    r = dd.verify_dataset(sources["aircraft"], str(root))
    assert not r["ok"] and "not in variants" in " ".join(r["problems"])
    assert dd.main(["--data-root", str(root), "--datasets", "dtd", "--verify-only"],
                   sources=sources) == 1


def test_real_spec_matches_documented_contract():
    doc = (REPO / "docs" / "DATASETS.md").read_text(encoding="utf-8")
    contract = {"dtd": 1692, "flower102": 2463, "pets": 3669, "eurosat": 8100, "aircraft": 3333}
    assert {k: s.expected_test for k, s in dd.SOURCES.items()} == contract
    for spec in dd.SOURCES.values():
        assert f"`{spec.folder}/`" in doc
        for df in spec.drive:
            assert df.file_id in doc and df.dest in doc
    for url in ("https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz",
                "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip",
                "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"):
        assert url in doc
        assert any(url in u for s in dd.SOURCES.values() for a in s.archives for u in a.urls)


def test_clean_path_handles_windows_trailing_quote():
    assert dd._clean_path('/tmp/data_root/"') == os.path.abspath("/tmp/data_root")
    assert dd._clean_path("/tmp/data_root/") == os.path.abspath("/tmp/data_root")
