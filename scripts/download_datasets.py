#!/usr/bin/env python
"""Download, extract, lay out and verify the five benchmark datasets.

Produces exactly the layout documented in ``docs/DATASETS.md`` under ``--data-root``
(default: ``$DATA_ROOT``), then checks it against the documented contract:

    test images  dtd 1692 · flower102 2463 · pets 3669 · eurosat 8100 · aircraft 3333
    0 missing files in any split the loaders read (train / val / test).

Cross-platform, standard library only (``tqdm`` is used for progress bars when installed).

* Downloads are resumable (``.part`` file + HTTP ``Range``), retried with backoff, size-
  checked against the server's length, and mirrored where a mirror exists.
* The CoOp split JSONs come from Google Drive by file ID; Drive's "can't scan for viruses"
  / confirm-token interstitial is handled, and a quota page is reported as such.
* Every step leaves a marker in ``<data-root>/_state/``; rerunning skips finished steps,
  so an interrupted run is continued by running the same command again.

Usage::

    python scripts/download_datasets.py --data-root /content/data_root
    python scripts/download_datasets.py --datasets dtd eurosat --keep-archives
    python scripts/download_datasets.py --verify-only
    python scripts/download_datasets.py --with-clip --clip-cache /content/clip_cache
"""
from __future__ import annotations

import argparse
import hashlib
import html
import http.client
import http.cookiejar
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:  # optional, already in requirements.txt
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback when tqdm is missing
    _tqdm = None

USER_AGENT = "Mozilla/5.0 (cal-tta dataset downloader; +python-urllib)"
CHUNK = 1 << 20
DRIVE_URL = "https://drive.google.com/uc?export=download&id={id}"
DRIVE_FALLBACK_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# ---------------------------------------------------------------------------------------
# Specification (mirrors docs/DATASETS.md -- that file is the contract)
# ---------------------------------------------------------------------------------------


@dataclass
class Archive:
    """One remote file. ``moves`` relocate paths from the extraction staging dir into the
    dataset folder; each source is a tuple of alternatives (first one present wins), which
    lets one spec accept differently-packed mirrors (e.g. EuroSAT_RGB/ vs 2750/)."""

    filename: str
    urls: List[str]
    approx_mb: int
    moves: List[Tuple[Tuple[str, ...], str]] = field(default_factory=list)
    extract: bool = True  # False: a plain file copied to ``plain_dest`` in the dataset dir
    plain_dest: str = ""
    inner: str = ""  # an archive inside this one (path in staging) to extract before ``moves``


@dataclass
class DriveFile:
    file_id: str
    dest: str  # relative to the dataset folder
    json_keys: Tuple[str, ...] = ()  # keys the JSON must have (validation)


@dataclass
class DatasetSpec:
    key: str  # our lowercase key, == --test_sets lower()
    folder: str  # folder under data_root (== data/datautils.py ID_to_DIRNAME)
    archives: List[Archive]
    drive: List[DriveFile]
    kind: str  # "json" (CoOp split JSON), "aircraft" (images_variant_*.txt), "imagefolder" (test only)
    image_dir: str  # == data/fewshot_datasets.py path_dict[key][0] ("images" for aircraft)
    split_file: str = ""  # json kind: split JSON relative to folder
    expected_test: int = 0
    reference_counts: Dict[str, int] = field(default_factory=dict)  # informational
    required: Tuple[str, ...] = ()  # paths (relative to folder) that must exist after layout


SPLIT_KEYS = ("train", "val", "test")
OX = "https://thor.robots.ox.ac.uk"
VGG = "https://www.robots.ox.ac.uk/~vgg/data"

SOURCES: Dict[str, DatasetSpec] = {
    "dtd": DatasetSpec(
        key="dtd", folder="dtd", kind="json", image_dir="images",
        split_file="split_zhou_DescribableTextures.json",
        archives=[Archive("dtd-r1.0.1.tar.gz",
                          [f"{OX}/dtd/dtd-r1.0.1.tar.gz", f"{VGG}/dtd/download/dtd-r1.0.1.tar.gz"],
                          625, moves=[(("dtd/images",), "images"), (("dtd/labels",), "labels"),
                                      (("dtd/imdb",), "imdb")])],
        drive=[DriveFile("1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x", "split_zhou_DescribableTextures.json",
                         SPLIT_KEYS)],
        expected_test=1692, reference_counts={"train": 2820, "val": 1128, "test": 1692},
        required=("images",),
    ),
    "flower102": DatasetSpec(
        key="flower102", folder="oxford_flowers", kind="json", image_dir="jpg",
        split_file="split_zhou_OxfordFlowers.json",
        archives=[
            Archive("102flowers.tgz", [f"{OX}/flowers/102/102flowers.tgz",
                                       f"{VGG}/flowers/102/102flowers.tgz"],
                    330, moves=[(("jpg",), "jpg")]),
            Archive("imagelabels.mat", [f"{OX}/flowers/102/imagelabels.mat",
                                        f"{VGG}/flowers/102/imagelabels.mat"],
                    1, extract=False, plain_dest="imagelabels.mat"),
            Archive("setid.mat", [f"{OX}/flowers/102/setid.mat", f"{VGG}/flowers/102/setid.mat"],
                    1, extract=False, plain_dest="setid.mat"),
        ],
        drive=[DriveFile("1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT", "split_zhou_OxfordFlowers.json", SPLIT_KEYS),
               DriveFile("1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0", "cat_to_name.json")],
        expected_test=2463, reference_counts={"train": 4093, "val": 1633, "test": 2463},
        required=("jpg",),
    ),
    "pets": DatasetSpec(
        key="pets", folder="oxford_pets", kind="json", image_dir="images",
        split_file="split_zhou_OxfordPets.json",
        archives=[
            Archive("images.tar.gz", [f"{OX}/pets/images.tar.gz", f"{VGG}/pets/data/images.tar.gz"],
                    755, moves=[(("images",), "images")]),
            Archive("annotations.tar.gz", [f"{OX}/pets/annotations.tar.gz",
                                           f"{VGG}/pets/data/annotations.tar.gz"],
                    19, moves=[(("annotations",), "annotations")]),
        ],
        drive=[DriveFile("1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs", "split_zhou_OxfordPets.json", SPLIT_KEYS)],
        expected_test=3669, reference_counts={"train": 2944, "val": 736, "test": 3669},
        required=("images", "annotations"),
    ),
    "eurosat": DatasetSpec(
        key="eurosat", folder="eurosat", kind="json", image_dir="2750",
        split_file="split_zhou_EuroSAT.json",
        archives=[Archive("EuroSAT_RGB.zip",
                          ["https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip",
                           "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
                           # original mirror, often offline; its archive's top dir is 2750/
                           "https://madm.dfki.de/files/sentinel/EuroSAT.zip"],
                          90, moves=[(("EuroSAT_RGB", "2750"), "2750")])],
        drive=[DriveFile("1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o", "split_zhou_EuroSAT.json", SPLIT_KEYS)],
        expected_test=8100, reference_counts={"train": 13500, "val": 5400, "test": 8100},
        required=("2750",),
    ),
    "aircraft": DatasetSpec(
        key="aircraft", folder="fgvc_aircraft", kind="aircraft", image_dir="images",
        archives=[Archive("fgvc-aircraft-2013b.tar.gz",
                          [f"{VGG}/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"],
                          2650, moves=[(("fgvc-aircraft-2013b/data",), ".")])],
        drive=[],
        expected_test=3333, reference_counts={"train": 3334, "val": 3333, "test": 3333},
        required=("images", "variants.txt", "images_variant_test.txt", "images_variant_val.txt",
                  "images_variant_train.txt", "images_variant_trainval.txt"),
    ),
}

# T14 development datasets (not part of the docs/DATASETS.md contract; `--datasets` opt-in).
DRIVE_DIRECT = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
DEV_SOURCES: Dict[str, DatasetSpec] = {
    "imagenetv2": DatasetSpec(
        key="imagenetv2", folder="imagenetv2", kind="imagefolder",
        image_dir="imagenetv2-matched-frequency-format-val",
        archives=[Archive("imagenetv2-matched-frequency.tar.gz",
                          ["https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/"
                           "imagenetv2-matched-frequency.tar.gz",
                           "https://s3-us-west-2.amazonaws.com/imagenetv2public/"
                           "imagenetv2-matched-frequency.tar.gz"],
                          1206, moves=[(("imagenetv2-matched-frequency-format-val",),
                                        "imagenetv2-matched-frequency-format-val")])],
        drive=[], expected_test=10000, reference_counts={"test": 10000},
        required=("imagenetv2-matched-frequency-format-val",),
    ),
    "caltech101": DatasetSpec(
        key="caltech101", folder="caltech-101", kind="json", image_dir="101_ObjectCategories",
        split_file="split_zhou_Caltech101.json",
        archives=[Archive("caltech-101.zip",
                          ["https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"],
                          131, inner="caltech-101/101_ObjectCategories.tar.gz",
                          moves=[(("101_ObjectCategories",), "101_ObjectCategories")])],
        drive=[DriveFile("1hyarUivQE36mY6jSomru6Fjd-JzwcCzN", "split_zhou_Caltech101.json", SPLIT_KEYS)],
        expected_test=2465, reference_counts={"train": 4128, "val": 1649, "test": 2465},
        required=("101_ObjectCategories",),
    ),
    "food101": DatasetSpec(
        key="food101", folder="food-101", kind="json", image_dir="images",
        split_file="split_zhou_Food101.json",
        archives=[Archive("food-101.tar.gz", ["https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"],
                          4765, moves=[(("food-101/images",), "images"), (("food-101/meta",), "meta")])],
        drive=[DriveFile("1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl", "split_zhou_Food101.json", SPLIT_KEYS)],
        expected_test=30300, reference_counts={"train": 50500, "val": 20200, "test": 30300},
        required=("images",),
    ),
    "ucf101": DatasetSpec(
        key="ucf101", folder="ucf101", kind="json", image_dir="UCF-101-midframes",
        split_file="split_zhou_UCF101.json",
        archives=[Archive("UCF-101-midframes.zip", [DRIVE_DIRECT.format("10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O")],
                          134, moves=[(("UCF-101-midframes",), "UCF-101-midframes")])],
        drive=[DriveFile("1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y", "split_zhou_UCF101.json", SPLIT_KEYS)],
        expected_test=3783, reference_counts={"train": 7639, "val": 1898, "test": 3783},
        required=("UCF-101-midframes",),
    ),
}
ALL_SOURCES: Dict[str, DatasetSpec] = {**SOURCES, **DEV_SOURCES}

CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_URLS = [
    f"https://openaipublic.azureedge.net/clip/models/{CLIP_SHA256}/ViT-B-16.pt",
    f"https://openaipublic.blob.core.windows.net/clip/models/{CLIP_SHA256}/ViT-B-16.pt",
]


# ---------------------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------------------


class DownloadError(RuntimeError):
    pass


class DriveQuotaError(DownloadError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def build_opener() -> urllib.request.OpenerDirector:
    """An opener with a cookie jar (Drive's legacy flow sets a download_warning cookie)."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    opener.cookiejar = jar  # type: ignore[attr-defined]
    return opener


def retry_fs(fn: Callable, *args, attempts: int = 8, delay: float = 0.5):
    """Retry a filesystem op that Windows antivirus / indexer may briefly lock."""
    for i in range(attempts):
        try:
            return fn(*args)
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


def state_dir(data_root: str) -> str:
    d = os.path.join(data_root, "_state")
    os.makedirs(d, exist_ok=True)
    return d


def marker_path(data_root: str, name: str) -> str:
    return os.path.join(state_dir(data_root), name + ".done")


def has_marker(data_root: str, name: str) -> bool:
    return os.path.isfile(marker_path(data_root, name))


def write_marker(data_root: str, name: str, info: Optional[dict] = None) -> None:
    with open(marker_path(data_root, name), "w", encoding="utf-8") as f:
        json.dump(info or {}, f, indent=1)


def read_marker(data_root: str, name: str) -> dict:
    try:
        with open(marker_path(data_root, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _progress(total: Optional[int], initial: int, desc: str):
    if _tqdm is not None:
        return _tqdm(total=total, initial=initial, unit="B", unit_scale=True,
                     unit_divisor=1024, desc=desc, ncols=90, leave=True)

    class _Plain:
        def __init__(self):
            self.n, self.last = initial, 0.0

        def update(self, k):
            self.n += k
            if time.time() - self.last > 5:
                self.last = time.time()
                tot = f"/{total / 2**20:.0f}" if total else ""
                log(f"  {desc}: {self.n / 2**20:.0f}{tot} MB")

        def close(self):
            pass

    return _Plain()


# ---------------------------------------------------------------------------------------
# Resumable HTTP download
# ---------------------------------------------------------------------------------------


def _remote_total(resp, offset: int) -> Optional[int]:
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    cl = resp.headers.get("Content-Length")
    if cl and cl.isdigit():
        return int(cl) + (offset if resp.status == 206 else 0)
    return None


def download_file(url: str, dest: str, opener=None, retries: int = 5,
                  timeout: float = 60.0, desc: str = "") -> int:
    """Download ``url`` to ``dest`` via ``dest + '.part'``, resuming with HTTP Range.

    Returns the final size. Raises DownloadError if the size the server announced is not
    what arrived after all retries.
    """
    opener = opener or build_opener()
    part = dest + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    desc = desc or os.path.basename(dest)
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        offset = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url)
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        try:
            with opener.open(req, timeout=timeout) as resp:
                if offset and resp.status != 206:
                    offset = 0  # server ignored Range: restart from scratch
                total = _remote_total(resp, offset)
                bar = _progress(total, offset, desc)
                with open(part, "ab" if offset else "wb") as out:
                    while True:
                        buf = resp.read(CHUNK)
                        if not buf:
                            break
                        out.write(buf)
                        bar.update(len(buf))
                bar.close()
            size = os.path.getsize(part)
            if total is not None and size != total:
                raise DownloadError(f"size mismatch: got {size} of {total} bytes")
            retry_fs(os.replace, part, dest)
            return size
        except urllib.error.HTTPError as e:
            if e.code == 416 and offset:  # range not satisfiable: .part is complete or bad
                head_total = remote_size(url, opener)
                if head_total == offset:
                    retry_fs(os.replace, part, dest)
                    return offset
                os.remove(part)
            last_err = e
            if 400 <= e.code < 500 and e.code not in (408, 416, 429):
                break  # not worth retrying (404, 403, ...)
        except (urllib.error.URLError, OSError, http.client.HTTPException, DownloadError) as e:
            last_err = e  # HTTPException covers IncompleteRead (connection cut mid-file)
        if attempt < retries:
            wait = min(2 ** attempt, 30)
            log(f"  retry {attempt}/{retries - 1} for {desc} in {wait}s ({last_err})")
            time.sleep(wait)
    raise DownloadError(f"failed to download {url}: {last_err}")


def remote_size(url: str, opener=None, timeout: float = 30.0) -> Optional[int]:
    opener = opener or build_opener()
    try:
        with opener.open(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None


def fetch_archive(arch: Archive, archives_dir: str, data_root: str, opener=None) -> str:
    """Make sure ``archives_dir/arch.filename`` is complete. Idempotent."""
    dest = os.path.join(archives_dir, arch.filename)
    mname = f"download.{arch.filename}"
    info = read_marker(data_root, mname)
    if os.path.isfile(dest) and info.get("size") == os.path.getsize(dest):
        log(f"  [skip] {arch.filename} already downloaded ({info['size'] / 2**20:.0f} MB)")
        return dest
    if os.path.isfile(dest):
        # A file placed by hand (or from an older run): accept it if it matches the
        # server's size, or if the server cannot tell us (extraction will catch damage).
        local = os.path.getsize(dest)
        sizes = [remote_size(u, opener) for u in arch.urls]
        known = [s for s in sizes if s]
        if not known or local in known:
            write_marker(data_root, mname, {"size": local, "url": "pre-existing"})
            log(f"  [skip] {arch.filename} found in archives dir ({local / 2**20:.0f} MB)")
            return dest
        log(f"  {arch.filename}: local size {local} != remote {known[0]}; resuming")
        retry_fs(os.replace, dest, dest + ".part")
    errors = []
    for url in arch.urls:
        log(f"  downloading {arch.filename} (~{arch.approx_mb} MB) from {urllib.parse.urlsplit(url).netloc}")
        try:
            size = download_file(url, dest, opener, desc=arch.filename)
            write_marker(data_root, mname, {"size": size, "url": url})
            return dest
        except DownloadError as e:
            errors.append(str(e))
            log(f"  mirror failed: {e}")
            # a .part from a different mirror may not be byte-identical: start fresh
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
    raise DownloadError(f"all mirrors failed for {arch.filename}:\n    " + "\n    ".join(errors))


# ---------------------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------------------

_QUOTA_PATTERNS = (
    "too many users have viewed or downloaded this file recently",
    "download quota for this file has been exceeded",
    "quota exceeded",
)


class _FormParser:
    """Tiny regex-based extractor for Drive's download form (no html.parser needed)."""

    FORM_RE = re.compile(r"<form\b[^>]*\bid=[\"']download-form[\"'][^>]*>(.*?)</form>",
                         re.IGNORECASE | re.DOTALL)
    ACTION_RE = re.compile(r"\baction=[\"']([^\"']+)[\"']", re.IGNORECASE)
    INPUT_RE = re.compile(r"<input\b[^>]*>", re.IGNORECASE)
    ATTR_RE = re.compile(r"\b(name|value|type)=[\"']([^\"']*)[\"']", re.IGNORECASE)

    @classmethod
    def parse(cls, page: str) -> Optional[Tuple[str, Dict[str, str]]]:
        m = cls.FORM_RE.search(page)
        if not m:
            return None
        form_tag = page[m.start(): m.start() + page[m.start():].find(">") + 1]
        am = cls.ACTION_RE.search(form_tag)
        if not am:
            return None
        params: Dict[str, str] = {}
        for tag in cls.INPUT_RE.findall(m.group(1)):
            attrs = {k.lower(): html.unescape(v) for k, v in cls.ATTR_RE.findall(tag)}
            if attrs.get("type", "").lower() == "hidden" and "name" in attrs:
                params[attrs["name"]] = attrs.get("value", "")
        return html.unescape(am.group(1)), params


def parse_drive_confirm(page: str, file_id: str, base_url: str = "https://drive.google.com",
                        cookies: Optional[Dict[str, str]] = None) -> str:
    """Given the HTML Drive returned instead of the file, return the URL to fetch next.

    Handles (1) the current ``<form id="download-form">`` interstitial, (2) the legacy
    ``confirm=<token>`` link, (3) the legacy ``download_warning_*`` cookie. Raises
    DriveQuotaError on a quota page, DownloadError when the page is anything else.
    """
    low = page.lower()
    if any(p in low for p in _QUOTA_PATTERNS):
        raise DriveQuotaError(
            f"Google Drive download quota exceeded for file {file_id}. Wait a few hours, or "
            f"open https://drive.google.com/file/d/{file_id}/view in a browser and save it as "
            f"<data-root>/_splits/<folder>/<file name>, then rerun (docs/COLAB_EXTRACTION.md).")
    form = _FormParser.parse(page)
    if form:
        action, params = form
        action = urllib.parse.urljoin(base_url + "/", action)
        params.setdefault("id", file_id)
        return action + ("&" if "?" in action else "?") + urllib.parse.urlencode(params)
    m = re.search(r"confirm=([0-9A-Za-z_\-]+)", html.unescape(page))
    if m:
        return (f"{base_url}/uc?export=download&confirm={m.group(1)}&id="
                f"{urllib.parse.quote(file_id)}")
    for k, v in (cookies or {}).items():
        if k.startswith("download_warning"):
            return f"{base_url}/uc?export=download&confirm={v}&id={urllib.parse.quote(file_id)}"
    if "accounts.google.com" in low or "servicelogin" in low:
        raise DownloadError(f"Drive file {file_id} requires sign-in (not shared publicly?)")
    raise DownloadError(f"unrecognised Google Drive response for {file_id}")


def _looks_like_html(body: bytes) -> bool:
    head = body[:4096].lstrip().lower()
    return head.startswith(b"<") and (b"<html" in head or b"<!doctype html" in head)


def drive_download(file_id: str, dest: str, opener=None, url_template: str = DRIVE_URL,
                   fallback_template: Optional[str] = DRIVE_FALLBACK_URL,
                   json_keys: Sequence[str] = (), max_hops: int = 3) -> int:
    """Download a (small) public Drive file, following the confirm interstitial.
    Validates the result as JSON when ``dest`` ends in ``.json``."""
    opener = opener or build_opener()
    templates = [url_template] + ([fallback_template] if fallback_template else [])
    errors: List[str] = []
    for tmpl in templates:
        url = tmpl.format(id=file_id)
        base = "{0.scheme}://{0.netloc}".format(urllib.parse.urlsplit(url))
        try:
            for _hop in range(max_hops):
                with opener.open(urllib.request.Request(url), timeout=60) as resp:
                    body = resp.read()
                if not _looks_like_html(body):
                    break
                jar = getattr(opener, "cookiejar", None)
                cookies = {c.name: c.value for c in jar} if jar is not None else {}
                url = parse_drive_confirm(body.decode("utf-8", "replace"), file_id, base, cookies)
            else:
                raise DownloadError(f"Drive kept returning HTML for {file_id}")
            if dest.endswith(".json"):
                try:
                    obj = json.loads(body.decode("utf-8"))
                except ValueError as e:
                    raise DownloadError(f"Drive file {file_id} is not valid JSON: {e}")
                missing = [k for k in json_keys if k not in obj]
                if missing:
                    raise DownloadError(f"Drive file {file_id} lacks keys {missing}")
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(body)
            retry_fs(os.replace, tmp, dest)
            return len(body)
        except DriveQuotaError:
            raise
        except (DownloadError, urllib.error.URLError, OSError, http.client.HTTPException) as e:
            errors.append(f"{url}: {e}")
    raise DownloadError(f"could not fetch Drive file {file_id}:\n    " + "\n    ".join(errors))


def _valid_json(path: str, keys: Sequence[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return all(k in obj for k in keys)
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------------------
# Extraction and layout
# ---------------------------------------------------------------------------------------


def _safe_target(root: str, name: str) -> str:
    target = os.path.realpath(os.path.join(root, name))
    root_r = os.path.realpath(root)
    if os.path.isabs(name) or not (target == root_r or target.startswith(root_r + os.sep)):
        raise DownloadError(f"archive member escapes extraction dir: {name!r}")
    return target


def extract_archive(path: str, out_dir: str) -> None:
    """Extract a .tar(.gz)/.tgz/.zip into ``out_dir``, refusing path traversal."""
    os.makedirs(out_dir, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                _safe_target(out_dir, n)
            zf.extractall(out_dir)
        return
    with tarfile.open(path, "r:*") as tf:
        members = []
        for m in tf.getmembers():
            _safe_target(out_dir, m.name)
            if m.issym() or m.islnk() or m.isdev():
                continue  # none of our archives need links/devices; skip them defensively
            members.append(m)
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(out_dir, members=members, **kwargs)


def _move_into(src: str, dst: str) -> None:
    """Move ``src`` to ``dst``; if both are directories, merge contents."""
    if os.path.isdir(src) and os.path.isdir(dst):
        for name in os.listdir(src):
            _move_into(os.path.join(src, name), os.path.join(dst, name))
        retry_fs(os.rmdir, src)
        return
    if os.path.exists(dst):
        if os.path.isdir(dst):
            retry_fs(shutil.rmtree, dst)
        else:
            retry_fs(os.remove, dst)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    retry_fs(shutil.move, src, dst)


def layout_archive(arch: Archive, archive_path: str, dataset_dir: str, staging_root: str) -> None:
    os.makedirs(dataset_dir, exist_ok=True)
    if not arch.extract:
        tmp = os.path.join(dataset_dir, arch.plain_dest + ".part")
        shutil.copyfile(archive_path, tmp)
        retry_fs(os.replace, tmp, os.path.join(dataset_dir, arch.plain_dest))
        return
    staging = os.path.join(staging_root, arch.filename + ".x")
    if os.path.exists(staging):
        retry_fs(shutil.rmtree, staging)
    log(f"  extracting {arch.filename} ...")
    extract_archive(archive_path, staging)
    if arch.inner:
        inner = os.path.join(staging, arch.inner)
        if not os.path.isfile(inner):
            raise DownloadError(f"{arch.filename}: inner archive {arch.inner} not found")
        log(f"  extracting {arch.inner} ...")
        extract_archive(inner, staging)
    for alternatives, dst_rel in arch.moves:
        src = next((os.path.join(staging, a) for a in alternatives
                    if os.path.exists(os.path.join(staging, a))), None)
        if src is None:
            raise DownloadError(f"{arch.filename}: none of {alternatives} found in archive")
        dst = os.path.normpath(os.path.join(dataset_dir, dst_rel))
        if dst_rel == ".":
            for name in os.listdir(src):
                _move_into(os.path.join(src, name), os.path.join(dataset_dir, name))
        else:
            if os.path.isdir(dst):  # stale partial layout from an interrupted run
                retry_fs(shutil.rmtree, dst)
            elif os.path.exists(dst):
                retry_fs(os.remove, dst)
            _move_into(src, dst)
    try:
        retry_fs(shutil.rmtree, staging)
    except OSError as e:  # never fatal: leftovers only cost disk
        log(f"  warning: could not remove staging dir {staging}: {e}")


# ---------------------------------------------------------------------------------------
# Verification against the documented contract
# ---------------------------------------------------------------------------------------


def _split_entries(spec: DatasetSpec, ds_dir: str) -> Dict[str, List[str]]:
    """Image paths (relative to the dataset dir) per split, read the way the loaders read
    them (data/fewshot_datasets.py BaseJsonDataset / Aircraft)."""
    out: Dict[str, List[str]] = {}
    if spec.kind == "json":
        with open(os.path.join(ds_dir, spec.split_file), encoding="utf-8") as f:
            splits = json.load(f)
        for s in SPLIT_KEYS:
            out[s] = [os.path.join(spec.image_dir, e[0]) for e in splits.get(s, [])]
    elif spec.kind == "imagefolder":
        root = os.path.join(ds_dir, spec.image_dir)
        out["test"] = sorted(os.path.relpath(os.path.join(d, n), ds_dir)
                             for d, _, names in os.walk(root) for n in names)
    else:
        with open(os.path.join(ds_dir, "variants.txt"), encoding="utf-8") as f:
            variants = {l.rstrip("\n") for l in f}
        for s in SPLIT_KEYS:
            p = os.path.join(ds_dir, f"images_variant_{s}.txt")
            rows = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    img, _, label = line.partition(" ")
                    if label not in variants:
                        raise DownloadError(f"aircraft {s}: label {label!r} not in variants.txt")
                    rows.append(os.path.join(spec.image_dir, img + ".jpg"))
            out[s] = rows
    return out


def verify_dataset(spec: DatasetSpec, data_root: str) -> dict:
    ds_dir = os.path.join(data_root, spec.folder)
    res = {"dataset": spec.key, "folder": spec.folder, "contract": spec.expected_test,
           "ok": True, "problems": [], "splits": {}}
    for rel in spec.required + ((spec.split_file,) if spec.split_file else ()):
        if not os.path.exists(os.path.join(ds_dir, rel)):
            res["ok"] = False
            res["problems"].append(f"missing {spec.folder}/{rel}")
    if not res["ok"]:
        return res
    try:
        entries = _split_entries(spec, ds_dir)
    except (OSError, ValueError, DownloadError) as e:
        res["ok"] = False
        res["problems"].append(f"cannot read split files: {e}")
        return res
    for s, paths in entries.items():
        missing = [p for p in paths if not os.path.isfile(os.path.join(ds_dir, p))]
        res["splits"][s] = {"count": len(paths), "missing": len(missing),
                            "reference": spec.reference_counts.get(s),
                            "first_missing": missing[:3]}
        if missing:
            res["ok"] = False
            res["problems"].append(f"{s}: {len(missing)} missing files, e.g. {missing[:3]}")
    test_n = res["splits"]["test"]["count"]
    if test_n != spec.expected_test:
        res["ok"] = False
        res["problems"].append(f"test count {test_n} != contract {spec.expected_test}")
    return res


def print_report(results: List[dict]) -> None:
    log("")
    log(f"{'dataset':<10} {'folder':<15} {'test':>6} {'contract':>8} {'val':>6} {'train':>6} "
        f"{'missing':>7}  status")
    total = 0
    for r in results:
        sp = r.get("splits", {})
        g = lambda s, k="count": sp.get(s, {}).get(k, "-")  # noqa: E731
        miss = sum(v.get("missing", 0) for v in sp.values()) if sp else "-"
        contract = r.get("contract", "-")
        if isinstance(g("test"), int):
            total += g("test")
        log(f"{r['dataset']:<10} {r['folder']:<15} {g('test'):>6} {contract:>8} {g('val'):>6} "
            f"{g('train'):>6} {miss:>7}  {'OK' if r['ok'] else 'FAIL'}")
        for s in ("val", "train"):
            ref, n = sp.get(s, {}).get("reference"), sp.get(s, {}).get("count")
            if ref and n is not None and n != ref:
                log(f"    note: {s} has {n} entries, CoOp reference is {ref}")
        for p in r["problems"]:
            log(f"    problem: {p}")
    log(f"total test images: {total}")


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------


def prepare_dataset(spec: DatasetSpec, data_root: str, keep_archives: bool = False,
                    opener=None, drive_url: str = DRIVE_URL,
                    drive_fallback: Optional[str] = DRIVE_FALLBACK_URL) -> None:
    opener = opener or build_opener()
    archives_dir = os.path.join(data_root, "_archives")
    splits_dir = os.path.join(data_root, "_splits", spec.folder)
    staging_root = os.path.join(data_root, "_staging")
    ds_dir = os.path.join(data_root, spec.folder)
    log(f"== {spec.key} -> {ds_dir}")

    for arch in spec.archives:
        lname = f"layout.{spec.key}.{arch.filename}"
        if not arch.extract:
            targets = [arch.plain_dest]
        else:
            targets = [d for _, d in arch.moves if d != "."] or list(spec.required)
        if has_marker(data_root, lname) and all(os.path.exists(os.path.join(ds_dir, t)) for t in targets):
            log(f"  [skip] {arch.filename} already laid out")
            continue
        path = fetch_archive(arch, archives_dir, data_root, opener)
        layout_archive(arch, path, ds_dir, staging_root)
        write_marker(data_root, lname, {"archive": arch.filename})
        if not keep_archives and arch.extract:
            try:
                retry_fs(os.remove, path)
            except OSError as e:
                log(f"  warning: could not delete {path}: {e}")

    for df in spec.drive:
        dest = os.path.join(ds_dir, df.dest)
        pristine = os.path.join(splits_dir, df.dest)
        if _valid_json(dest, df.json_keys):
            log(f"  [skip] {df.dest} present")
            continue
        if not _valid_json(pristine, df.json_keys):
            log(f"  fetching {df.dest} from Google Drive ({df.file_id})")
            drive_download(df.file_id, pristine, opener, drive_url, drive_fallback, df.json_keys)
        os.makedirs(ds_dir, exist_ok=True)
        shutil.copyfile(pristine, dest)

    if os.path.isdir(staging_root) and not os.listdir(staging_root):
        os.rmdir(staging_root)


def fetch_clip(cache_dir: str, opener=None, urls: Sequence[str] = CLIP_URLS,
               sha256: str = CLIP_SHA256) -> str:
    """Put ViT-B-16.pt in ``cache_dir`` (the CLIP_CACHE_ROOT clip.load reads), sha-checked."""
    dest = os.path.join(cache_dir, "ViT-B-16.pt")

    def _ok(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(CHUNK), b""):
                h.update(b)
        return h.hexdigest() == sha256

    if os.path.isfile(dest) and _ok(dest):
        log(f"  [skip] CLIP ViT-B/16 already in {cache_dir}")
        return dest
    errors = []
    for url in urls:
        log(f"  downloading CLIP ViT-B/16 (~335 MB) from {urllib.parse.urlsplit(url).netloc}")
        try:
            download_file(url, dest, opener, desc="ViT-B-16.pt")
            if _ok(dest):
                return dest
            os.remove(dest)
            errors.append(f"{url}: sha256 mismatch")
        except DownloadError as e:
            errors.append(str(e))
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
    raise DownloadError("could not fetch CLIP ViT-B/16:\n    " + "\n    ".join(errors))


def _check_disk(data_root: str, specs: Sequence[DatasetSpec], keep: bool) -> None:
    need_mb = sum(a.approx_mb for s in specs for a in s.archives)
    need_mb = int(need_mb * (2.1 if keep else 1.6))  # archives + extracted (+ peak slack)
    try:
        free_mb = shutil.disk_usage(data_root).free // 2**20
    except OSError:
        return
    if free_mb < need_mb:
        log(f"WARNING: {free_mb} MB free on the data drive, ~{need_mb} MB may be needed")


def parse_args(argv=None, sources: Optional[Dict[str, DatasetSpec]] = None):
    default = list(SOURCES) if sources is None else list(sources)
    sources = ALL_SOURCES if sources is None else sources
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT"),
                   help="dataset root (default: $DATA_ROOT)")
    p.add_argument("--datasets", nargs="+", default=default, choices=list(sources),
                   metavar="NAME", help=f"subset of {' '.join(sources)} (default: {' '.join(default)})")
    p.add_argument("--keep-archives", action="store_true",
                   help="keep downloaded archives in <data-root>/_archives after extraction")
    p.add_argument("--verify-only", action="store_true", help="only run the verification")
    p.add_argument("--with-clip", action="store_true",
                   help="also fetch the CLIP ViT-B/16 checkpoint into --clip-cache")
    p.add_argument("--clip-cache", default=os.environ.get("CLIP_CACHE_ROOT"),
                   help="CLIP checkpoint cache dir (default: $CLIP_CACHE_ROOT)")
    a = p.parse_args(argv)
    if not a.data_root:
        p.error("pass --data-root or set DATA_ROOT")
    if a.with_clip and not a.clip_cache:
        p.error("--with-clip needs --clip-cache or CLIP_CACHE_ROOT")
    a.data_root = _clean_path(a.data_root)
    if a.clip_cache:
        a.clip_cache = _clean_path(a.clip_cache)
    return a


def _clean_path(p: str) -> str:
    """Normalise a user path. Strips a stray trailing quote, which is what Windows argv
    parsing leaves when PowerShell passes ``"D:\\data\\"`` (the ``\\"`` escapes the quote);
    abspath then drops any trailing separator without turning ``D:\\`` into ``D:``."""
    return os.path.abspath(os.path.expanduser(p.strip().strip('"').strip()))


def main(argv=None, sources: Optional[Dict[str, DatasetSpec]] = None, opener=None,
         drive_url: str = DRIVE_URL, drive_fallback: Optional[str] = DRIVE_FALLBACK_URL) -> int:
    args = parse_args(argv, sources)
    sources = ALL_SOURCES if sources is None else sources
    specs = [sources[k] for k in dict.fromkeys(args.datasets)]
    os.makedirs(args.data_root, exist_ok=True)
    log(f"DATA_ROOT = {args.data_root}")
    failed: List[str] = []
    if not args.verify_only:
        _check_disk(args.data_root, specs, args.keep_archives)
        for spec in specs:
            try:
                prepare_dataset(spec, args.data_root, args.keep_archives, opener,
                                drive_url, drive_fallback)
            except DownloadError as e:
                log(f"ERROR [{spec.key}]: {e}")
                failed.append(spec.key)
        if args.with_clip:
            try:
                fetch_clip(os.path.abspath(args.clip_cache), opener)
            except DownloadError as e:
                log(f"ERROR [clip]: {e}")
                failed.append("clip")
    results = [verify_dataset(s, args.data_root) for s in specs]
    print_report(results)
    with open(os.path.join(state_dir(args.data_root), "verify.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    bad = [r["dataset"] for r in results if not r["ok"]]
    if failed or bad:
        log(f"\nNOT READY: {', '.join(dict.fromkeys(failed + bad))}. Fix the problem above and "
            f"rerun the same command; finished steps are skipped.")
        return 1
    log("\nAll requested datasets match the contract in docs/DATASETS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
