# %% [markdown]
# # T01 feature extraction on a free Colab T4
#
# One-time GPU job: export the CLIP ViT-B/16 features that the CPU replay needs (all five
# datasets; test / val / train16 splits; 10 augmented train16 views; the extractor's default
# seeds). Guide: `docs/COLAB_EXTRACTION.md`.
#
# **How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**.
# Allow the Google Drive prompt. Expect about 1–1.5 h. The result is
# `MyDrive/cal_tta_features/features_<date>.zip`.
#
# **If Colab disconnects:** reconnect and Run all again. Finished datasets are kept on your
# Drive and skipped; only unfinished ones are downloaded and extracted again.
#
# Maintainers: `notebooks/extract_features_colab.py` holds the same cells as this notebook;
# edit the `.py`, then run `python notebooks/sync_colab_notebook.py`.

# %% [markdown]
# ## Settings
# Change these only if asked to.

# %%
REPO_URL = "https://github.com/macwave12/deep_project_calibration_aware_tta"
BRANCH = "deck/refinement"
DRIVE_FOLDER = "cal_tta_features"  # under MyDrive/
DATASETS = ["dtd", "flower102", "pets", "eurosat", "aircraft"]
EXTRA_EXTRACT_ARGS = []  # passed to scripts/extract_features.py; empty = its defaults

REPO_DIR = "/content/deep_project_calibration_aware_tta"
DATA_ROOT = "/content/data_root"  # local disk (fast), not Drive
CLIP_CACHE = "/content/clip_cache"
LOCAL_FEATURES = "/content/features"
DRIVE_MOUNT = "/content/drive"

# %% [markdown]
# ## Helpers
# Plain functions used by the cells below (unit-tested offline in
# `tests/test_colab_notebook.py`). Torch is never imported in this kernel: every step runs
# in a subprocess, so installing a different torch needs no runtime restart.

# %%
# --- helpers ---
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def run(cmd, cwd=None, env=None, log_path=None, check=True):
    """Run a command, streaming its output live (and appending it to log_path)."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=full_env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log:
                log.write(line)
        code = proc.wait()
    finally:
        if log:
            log.close()
    if check and code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(str(c) for c in cmd)}")
    return code


def parse_nvidia_smi(text):
    """'Tesla T4, 7.5' lines -> [(name, compute capability)]."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            gpus.append((parts[0], parts[1]))
    return gpus


NO_GPU_MESSAGE = ("No GPU is attached to this runtime. In the menu choose Runtime -> Change runtime "
                  "type -> Hardware accelerator: T4 GPU -> Save, then Runtime -> Run all again.")


def read_torch_pins(requirements_text):
    """The torch / torchvision pins and the extra index URL from requirements.txt."""
    def pin(name):
        m = re.search(rf"^\s*{name}==(\S+)", requirements_text, flags=re.MULTILINE)
        return m.group(1) if m else None
    idx = re.search(r"^\s*--(?:extra-)?index-url\s+(\S+)", requirements_text, flags=re.MULTILINE)
    return {"torch": pin("torch"), "torchvision": pin("torchvision"),
            "index_url": idx.group(1) if idx else None}


def dataset_done(features_root, key):
    """The extractor writes meta.json last: its presence marks a finished dataset."""
    return (Path(features_root) / key / "meta.json").is_file()


def pending_datasets(drive_features, datasets):
    return [d for d in datasets if not dataset_done(drive_features, d)]


def copy_dir_atomic(src, dst):
    """Copy src to dst via dst.tmp, so a disconnect never leaves a half-copied dst."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)


def make_bundle(zip_path, features_root, datasets, extras):
    """Zip features/<dataset>/... plus extra files (name in zip -> local path)."""
    zip_path = Path(zip_path)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in datasets:
            base = Path(features_root) / key
            if not base.is_dir():
                continue
            for f in sorted(base.rglob("*")):
                if f.is_file():
                    zf.write(f, Path("features") / key / f.relative_to(base))
        for name, path in extras.items():
            if path and Path(path).is_file():
                zf.write(path, name)
    tmp.replace(zip_path)
    return zip_path


def python_info(code):
    """Run a snippet in a fresh interpreter and parse its last stdout line as JSON."""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stdout + out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


# %% [markdown]
# ## a. GPU check
# Fails with instructions if no GPU is attached. A T4 reports compute capability 7.5.

# %%
smi = shutil.which("nvidia-smi")
gpus = []
if smi:
    q = subprocess.run([smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                       capture_output=True, text=True)
    gpus = parse_nvidia_smi(q.stdout) if q.returncode == 0 else []
if not gpus:
    raise RuntimeError(NO_GPU_MESSAGE)
print("GPU:", gpus)
if not any("T4" in name for name, _ in gpus):
    print("Note: not a T4. Any CUDA GPU works; the guide's timings assume a T4.")

# %% [markdown]
# ## b. Google Drive
# Mounts your Drive. Finished datasets, logs and the final zip go to `MyDrive/<DRIVE_FOLDER>/`.

# %%
from google.colab import drive  # noqa: E402  (Colab only)

drive.mount(DRIVE_MOUNT)
OUT_DIR = Path(DRIVE_MOUNT) / "MyDrive" / DRIVE_FOLDER
WORK_DIR = OUT_DIR / "work"
DRIVE_FEATURES = WORK_DIR / "features"
DRIVE_FEATURES.mkdir(parents=True, exist_ok=True)
RUN_INFO_PATH = WORK_DIR / "colab_run.json"
run_info = json.loads(RUN_INFO_PATH.read_text()) if RUN_INFO_PATH.exists() else {"sessions": []}
run_info["sessions"].append({"started": datetime.datetime.now().isoformat(timespec="seconds"),
                             "gpus": gpus})
print("output folder:", OUT_DIR)
print("already finished:", [d for d in DATASETS if dataset_done(DRIVE_FEATURES, d)] or "none")

# %% [markdown]
# ## c. Clone the repository
# Clones `BRANCH` (or updates an existing clone in this session) and checks that the files this
# notebook needs are there.

# %%
if (Path(REPO_DIR) / ".git").is_dir():
    run(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR)
    run(["git", "checkout", "-B", BRANCH, "FETCH_HEAD"], cwd=REPO_DIR)
else:
    run(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR])
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True,
                        text=True).stdout.strip()
print("commit:", commit)
run_info["sessions"][-1]["commit"] = commit
needed = ["scripts/extract_features.py", "scripts/extract_features.sh",
          "scripts/download_datasets.py", "requirements.txt"]
missing = [p for p in needed if not (Path(REPO_DIR) / p).is_file()]
if missing:
    raise RuntimeError(f"branch {BRANCH} lacks {missing}; ask the lead which BRANCH to use")

# %% [markdown]
# ## d. Install the pinned torch
# Installs exactly the torch / torchvision that `requirements.txt` pins (the extractor refuses
# any other version), from the PyTorch CUDA index named there, then the remaining
# requirements. The CUDA 12.8 build includes T4 (sm_75) kernels; a GPU matmul proves it.

# %%
pins = read_torch_pins((Path(REPO_DIR) / "requirements.txt").read_text())
print("pins:", pins)
TORCH_CHECK = r"""
import json, torch
x = torch.randn(1024, 1024, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    y = float((x @ x).float().sum())
print(json.dumps({"torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                  "arch_list": torch.cuda.get_arch_list(), "matmul_finite": y == y}))
"""
try:
    have = python_info("import json, torch; print(json.dumps({'torch': torch.__version__}))")["torch"]
except Exception:
    have = None
if have != pins["torch"]:
    run([sys.executable, "-m", "pip", "install", "-q", f"torch=={pins['torch']}",
         f"torchvision=={pins['torchvision']}", "--index-url", pins["index_url"]])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO_DIR)
torch_info = python_info(TORCH_CHECK)
print(json.dumps(torch_info, indent=1))
run_info["sessions"][-1]["torch"] = torch_info
if torch_info["torch"] != pins["torch"]:
    raise RuntimeError(f"torch {torch_info['torch']} installed, extractor expects {pins['torch']}")
if not torch_info["matmul_finite"]:
    raise RuntimeError("GPU matmul returned a non-finite value: this torch build cannot use this GPU")

# %% [markdown]
# ## e. Datasets
# Downloads only the datasets not finished yet, to local disk (`/content/data_root`), plus the
# CLIP ViT-B/16 weights, then verifies the layout: test counts 1692 / 2463 / 3669 / 8100 / 3333
# with 0 missing files. Rerunning skips what is already there. ~4.4 GB, typically 10–30 min.

# %%
todo = pending_datasets(DRIVE_FEATURES, DATASETS)
print("datasets to extract:", todo or "none (all finished)")
ENV = {"DATA_ROOT": DATA_ROOT, "CLIP_CACHE_ROOT": CLIP_CACHE}
if todo:
    run([sys.executable, "scripts/download_datasets.py", "--data-root", DATA_ROOT,
         "--datasets", *todo, "--with-clip", "--clip-cache", CLIP_CACHE],
        cwd=REPO_DIR, env=ENV, log_path=WORK_DIR / "download_datasets.log")
    verify = Path(DATA_ROOT) / "_state" / "verify.json"
    shutil.copyfile(verify, WORK_DIR / f"datasets_verify_{'_'.join(todo)}.json")

# %% [markdown]
# ## f. Feature extraction
# Runs `scripts/extract_features.sh` (→ `scripts/extract_features.py`, its default settings:
# test / val / train16, 10 augmented train16 views, seeds 0/1/2) one dataset at a time. Each
# finished dataset is copied to Drive at once, so a disconnect loses at most the dataset in
# progress. Watch for `stream order matches recorded run` (`n/a` is expected here: the recorded
# runs are not on Colab).

# %%
for key in todo:
    t0 = time.time()
    run(["bash", "scripts/extract_features.sh", "--datasets", key, *EXTRA_EXTRACT_ARGS],
        cwd=REPO_DIR, env={**ENV, "OUT": LOCAL_FEATURES})
    if not dataset_done(LOCAL_FEATURES, key):
        raise RuntimeError(f"{key}: the extractor finished without meta.json; see the output above")
    copy_dir_atomic(Path(LOCAL_FEATURES) / key, DRIVE_FEATURES / key)
    log = Path(LOCAL_FEATURES) / "extract_features.log"
    if log.is_file():
        with open(WORK_DIR / "extract_features.log", "a", encoding="utf-8") as f:
            f.write(log.read_text(encoding="utf-8"))
        log.unlink()
    run_info.setdefault("extract_seconds", {})[key] = round(time.time() - t0)
    RUN_INFO_PATH.write_text(json.dumps(run_info, indent=1))
    print(f"[{key}] saved to Drive in {time.time() - t0:.0f}s")
print("finished:", [d for d in DATASETS if dataset_done(DRIVE_FEATURES, d)])

# %% [markdown]
# ## g. GPU tests
# The GPU-only checks the extractor's handoff asks for. Their output goes into the zip; please
# also paste the `CUDA autocast vs emulation:` lines to the lead.

# %%
GPU_TESTS = ["tests/test_feature_replay.py", "tests/test_device_agnostic.py",
             "tests/test_autocast_emulation_gpu.py"]
present = [t for t in GPU_TESTS if (Path(REPO_DIR) / t).is_file()]
if len(present) < len(GPU_TESTS):
    print("WARNING: not on this branch:", sorted(set(GPU_TESTS) - set(present)))
PYTEST_LOG = WORK_DIR / "pytest_gpu.txt"
PYTEST_LOG.write_text("")
pytest_code = run([sys.executable, "-m", "pytest", "-s", "-rs", *present], cwd=REPO_DIR,
                  env=ENV, log_path=PYTEST_LOG, check=False) if present else None
run_info["pytest_gpu_exit_code"] = pytest_code
RUN_INFO_PATH.write_text(json.dumps(run_info, indent=1))
if pytest_code != 0:
    print("\nWARNING: GPU tests did not all pass. The zip is still made; tell the lead.")

# %% [markdown]
# ## h. Bundle to Drive
# Zips `features/` (with each dataset's `meta.json`), the logs, the pytest output and this
# run's info into `MyDrive/<DRIVE_FOLDER>/features_<date>.zip`. Send the lead that file (or just
# say it is there).

# %%
done = [d for d in DATASETS if dataset_done(DRIVE_FEATURES, d)]
stamp = datetime.date.today().isoformat()
name = f"features_{stamp}.zip" if len(done) == len(DATASETS) else f"features_{stamp}_partial.zip"
run_info["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
run_info["datasets_done"] = done
RUN_INFO_PATH.write_text(json.dumps(run_info, indent=1))
extras = {"colab_run.json": RUN_INFO_PATH, "pytest_gpu.txt": PYTEST_LOG,
          "features/extract_features.log": WORK_DIR / "extract_features.log",
          "download_datasets.log": WORK_DIR / "download_datasets.log"}
extras.update({p.name: p for p in WORK_DIR.glob("datasets_verify_*.json")})
local_zip = make_bundle(Path("/content") / name, DRIVE_FEATURES, done, extras)
final = OUT_DIR / name
shutil.copyfile(local_zip, final)
print(f"\n{final}  ({final.stat().st_size / 1e6:.1f} MB)  datasets: {', '.join(done)}")
if len(done) < len(DATASETS):
    print("PARTIAL: missing", sorted(set(DATASETS) - set(done)), "- run all again to finish.")
