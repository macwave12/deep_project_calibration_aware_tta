# T13 — Colab extraction notebook

Status: review · Owner: engineering agent (notebook), Idan (run) · Depends on: T01 extractor merged
into `deck/refinement`

Re-scoped 2026-11-08 (DECISIONS): was "local CPU setup" for a Windows machine without a GPU; the
one-time extraction now runs on a free Google Colab T4. The Windows scripts were dropped; they
remain in this branch's history (commits `398f568`, `6764b01`) as a fallback.

## Goal
Let Idan run T01 on Colab with "Run all": no local install, no other teammate, result on his
Google Drive for the lead to pick up.

## Deliverables
- `scripts/download_datasets.py`: downloads, extracts and lays out the five datasets exactly as
  `docs/DATASETS.md` documents (EuroSAT_RGB → 2750, aircraft `data/` flattened incl.
  `images_variant_{train,val,trainval,test}.txt`), CoOp split JSONs from Drive (confirm page
  handled), optional CLIP ViT-B/16 with sha256 check; resumable, mirrored, idempotent; verifies
  the contract (test 1692/2463/3669/8100/3333, 0 missing in train/val/test).
- `notebooks/extract_features_colab.ipynb` + `.py` mirror (`notebooks/sync_colab_notebook.py`
  regenerates the notebook): GPU check, Drive mount, clone, exact torch pin, data, per-dataset
  extraction with Drive checkpointing, GPU tests, `MyDrive/cal_tta_features/features_<date>.zip`.
- `docs/COLAB_EXTRACTION.md`: one-page guide with the Colab URL, timings, success criteria,
  failure handling.
- Tests (offline, CPU): `tests/test_download_datasets.py`, `tests/test_colab_notebook.py`.

## Acceptance
- `python3 -m pytest -q` green; the new tests need no network and run in seconds.
- Notebook JSON valid; `.py` mirror equals the notebook's cells (test).
- Notebook installs exactly the `requirements.txt` torch pin (what `scripts/extract_features.py`'s
  guard expects on a GPU).
- Real run on Colab (Idan): verification table all `OK`, five datasets extracted, GPU tests pass,
  zip on Drive. **Not yet done** — nothing here has run on Colab or against the real mirrors.
