"""Offline checks for notebooks/extract_features_colab.{ipynb,py} (no Colab, no GPU)."""
import ast
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "extract_features_colab.ipynb"
PY_PATH = REPO / "notebooks" / "extract_features_colab.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sync = _load("sync_colab_notebook", REPO / "notebooks" / "sync_colab_notebook.py")


def _code_cells():
    return [src for kind, src in sync.parse_percent(PY_PATH.read_text(encoding="utf-8")) if kind == "code"]


@pytest.fixture(scope="module")
def helpers():
    cell = next(src for src in _code_cells() if src.startswith("# --- helpers ---"))
    ns = {"__name__": "colab_helpers"}
    exec(compile(cell, "helpers-cell", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def settings():
    ns = {}
    exec(_code_cells()[0], ns)
    return ns


def test_notebook_is_valid_json_with_gpu_metadata():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert all(c["cell_type"] in ("code", "markdown") for c in nb["cells"])


def test_py_mirror_matches_notebook_cells():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    assert sync.notebook_cells(nb) == sync.parse_percent(PY_PATH.read_text(encoding="utf-8"))
    assert sync.main(["--check"]) == 0


def test_every_step_has_markdown_and_code_compiles():
    cells = sync.parse_percent(PY_PATH.read_text(encoding="utf-8"))
    for kind, src in cells:
        if kind == "code":
            ast.parse(src)
            # plain Python only (no shell/magic lines), and torch never imported in the kernel
            for line in src.splitlines():
                assert not line.lstrip().startswith(("!", "%")), line
                assert not line.strip().startswith(("import torch", "from torch")), line
    headings = [src.splitlines()[0] for kind, src in cells if kind == "markdown"]
    for step in "abcdefgh":
        assert any(h.startswith(f"## {step}.") for h in headings), step
    # each step's markdown is directly followed by its code cell
    for i, (kind, src) in enumerate(cells):
        if kind == "markdown" and src.startswith("## ") and src[3] in "abcdefgh" and src[4] == ".":
            assert cells[i + 1][0] == "code"


def test_settings(settings):
    assert settings["BRANCH"] == "deck/refinement"
    assert settings["REPO_URL"] == "https://github.com/macwave12/deep_project_calibration_aware_tta"
    assert settings["DRIVE_FOLDER"] == "cal_tta_features"
    assert settings["DATASETS"] == ["dtd", "flower102", "pets", "eurosat", "aircraft"]
    assert not settings["DATA_ROOT"].startswith("/content/drive")  # datasets on local disk


def test_downloader_accepts_the_notebook_flags(settings):
    dd = _load("download_datasets_nb", REPO / "scripts" / "download_datasets.py")
    a = dd.parse_args(["--data-root", settings["DATA_ROOT"], "--datasets", *settings["DATASETS"],
                       "--with-clip", "--clip-cache", settings["CLIP_CACHE"]])
    assert a.with_clip and a.datasets == settings["DATASETS"]


def test_read_torch_pins_matches_requirements(helpers):
    pins = helpers["read_torch_pins"]((REPO / "requirements.txt").read_text())
    # the extractor's guard expects exactly the requirements.txt pin on a GPU machine
    assert pins == {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128",
                    "index_url": "https://download.pytorch.org/whl/cu128"}


def test_parse_nvidia_smi(helpers):
    parse = helpers["parse_nvidia_smi"]
    assert parse("Tesla T4, 7.5\n") == [("Tesla T4", "7.5")]
    assert parse("") == []
    assert parse("NVIDIA A100-SXM4-40GB, 8.0\nTesla T4, 7.5") == [("NVIDIA A100-SXM4-40GB", "8.0"),
                                                                ("Tesla T4", "7.5")]
    assert "T4 GPU" in helpers["NO_GPU_MESSAGE"] and "Change runtime type" in helpers["NO_GPU_MESSAGE"]


def test_pending_datasets_and_atomic_copy(helpers, tmp_path):
    drive = tmp_path / "drive"
    (drive / "dtd").mkdir(parents=True)
    (drive / "dtd" / "meta.json").write_text("{}")
    (drive / "pets").mkdir()  # partial: no meta.json
    assert helpers["pending_datasets"](drive, ["dtd", "pets", "eurosat"]) == ["pets", "eurosat"]

    local = tmp_path / "local" / "pets"
    local.mkdir(parents=True)
    (local / "test_feats.pt").write_bytes(b"x")
    (local / "meta.json").write_text("{}")
    (drive / "pets.tmp").mkdir()  # leftover from an interrupted copy
    helpers["copy_dir_atomic"](local, drive / "pets")
    assert (drive / "pets" / "test_feats.pt").read_bytes() == b"x"
    assert not (drive / "pets.tmp").exists()
    assert helpers["pending_datasets"](drive, ["dtd", "pets"]) == []


def test_make_bundle(helpers, tmp_path):
    feats = tmp_path / "features"
    for key in ("dtd", "pets"):
        (feats / key).mkdir(parents=True)
        (feats / key / "meta.json").write_text("{}")
        (feats / key / "test_feats.pt").write_bytes(b"f")
    (tmp_path / "pytest.txt").write_text("4 passed")
    z = helpers["make_bundle"](tmp_path / "out.zip", feats, ["dtd", "pets", "eurosat"],
                               {"pytest_gpu.txt": tmp_path / "pytest.txt", "absent.txt": tmp_path / "nope"})
    names = sorted(zipfile.ZipFile(z).namelist())
    assert names == ["features/dtd/meta.json", "features/dtd/test_feats.pt", "features/pets/meta.json",
                     "features/pets/test_feats.pt", "pytest_gpu.txt"]
    assert not (tmp_path / "out.zip.tmp").exists()


def test_run_streams_logs_and_checks(helpers, tmp_path, capsys):
    log = tmp_path / "log.txt"
    code = helpers["run"]([sys.executable, "-c", "print('hello'); import sys; print('err', file=sys.stderr)"],
                          log_path=log)
    assert code == 0
    assert "hello" in capsys.readouterr().out
    assert log.read_text().split() == ["hello", "err"]
    assert helpers["run"]([sys.executable, "-c", "raise SystemExit(3)"], check=False) == 3
    with pytest.raises(RuntimeError, match="exit code 3"):
        helpers["run"]([sys.executable, "-c", "raise SystemExit(3)"])


def test_python_info(helpers):
    assert helpers["python_info"]("import json; print('noise'); print(json.dumps({'a': 1}))") == {"a": 1}
