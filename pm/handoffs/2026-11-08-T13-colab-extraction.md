# 2026-11-08 — T13 Colab extraction notebook (engineering agent)

Branch `worktree-agent-a45f3048fbe02cb1b`, rebased onto `deck/refinement` (1bf4271). Status `review`.

## What was done
- `scripts/download_datasets.py` (stdlib only; tqdm if present): resumable `.part` + Range
  downloads with size check, retries and mirrors; safe tar/zip extraction; layout per
  `docs/DATASETS.md`; Drive split JSONs (form interstitial, legacy `confirm=` link, cookie token,
  quota page → clear error); `--with-clip` fetches ViT-B-16.pt with sha256 check (azureedge, then
  blob.core.windows.net). Markers in `<data-root>/_state/`, report in `_state/verify.json`.
  Archives or split files placed by hand in `_archives/` / `_splits/<folder>/` are used.
- `notebooks/extract_features_colab.ipynb` and its `.py` mirror; `notebooks/sync_colab_notebook.py`.
  The kernel never imports torch (everything runs in subprocesses), so there is no runtime restart
  after the torch install. It extracts one dataset per extractor call into `/content/features`
  and copies each finished dataset (meta.json present) to `MyDrive/cal_tta_features/work/features/`.
  A rerun skips datasets already there.
- `docs/COLAB_EXTRACTION.md`, `pm/tasks/T13-colab-extraction.md`; `docs/DATASETS.md` links to it.
- The Windows CPU scripts (`setup_local_cpu.ps1`, `download_datasets.ps1`, `requirements-cpu.txt`,
  `docs/LOCAL_CPU_SETUP.md`) were written and then removed as out of scope. They are in history at
  `398f568` / `6764b01`.

## How it was verified
- `python3 -m pytest -q`: 146 passed, 19 skipped (was 111 / 19 on this branch). New:
  `tests/test_download_datasets.py` (24, ~10 s) builds fake archives with the real file names and
  internal structure and serves them from a local `http.server` with Range and a fake Drive endpoint.
  It checks the layout, mirror fallback, idempotency (no requests on rerun, works with the server
  stopped), resume, the 416 case, truncation, path traversal, verification failures, the quota page,
  Drive HTML parsing from saved samples (`tests/fixtures/drive/`) and that the spec matches
  `docs/DATASETS.md`. `tests/test_colab_notebook.py` (11, <1 s) checks the notebook JSON, the
  `.py` ↔ `.ipynb` match, that the code compiles without shell magics, the helper functions, and
  that the torch pin read from `requirements.txt` is `2.11.0+cu128`.

## Not verified (could not be run here)
- **Nothing ran on Colab, on a GPU, or against the real mirrors or Drive.** The real Drive
  interstitial HTML may differ from the samples. The Zenodo, Oxford and madm URLs, and the
  `www.robots.ox.ac.uk` fallbacks, were not fetched.
- The CoOp val/train reference counts (DTD 1128/2820, Flowers 1633/4093, Pets 736/2944,
  EuroSAT 5400/13500, Aircraft 3333/3334) come from memory of the CoOp paper. A mismatch is only
  printed as a note, not treated as a failure. The test counts are the enforced contract.
- I assume the `torch==2.11.0+cu128` wheel ships sm_75 kernels (cu128 builds list 7.5 as far as I
  know) and has a wheel for Colab's Python. Cell d proves it with a GPU matmul and stops if it
  fails.
- Timings in the guide are estimates.
- `extract_features.py`/`.sh` and the three GPU test files are read from
  `worktree-agent-ace5471a3fbddc65c`. They are **not on `deck/refinement` yet**: until they are,
  cell c stops with "branch lacks ...". The notebook URL in the guide works only once this branch
  and that one are merged into `deck/refinement`.
- The recorded-run check prints `n/a` on Colab because `online_results/ckps/*/save_results.pt` is
  not in git. The stream order is then confirmed only later, by T02's replay against `summary.csv`.

## Next person
- Lead: merge the extractor branch and this branch into `deck/refinement`. Then send Idan
  `docs/COLAB_EXTRACTION.md`.
- Reviewer: check the notebook cell by cell against the extractor handoff
  (`pm/handoffs/2026-11-07-T01-T02-feature-replay.md`). After the run, check `colab_run.json` and
  `pytest_gpu.txt` in the zip.
- The extractor handoff's CPU alternative still points to `docs/LOCAL_CPU_SETUP.md`, which is
  no longer on this branch.
