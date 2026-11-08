# Feature extraction on Google Colab (free T4)

One-time GPU job (T01): export the CLIP features every later experiment replays on CPU.
Nothing to install locally; you need a Google account with ~1 GB free on Google Drive.

## Run it

1. Open the notebook straight from GitHub:
   <https://colab.research.google.com/github/macwave12/deep_project_calibration_aware_tta/blob/deck/refinement/notebooks/extract_features_colab.ipynb>
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
3. **Runtime → Run all.** Accept the Google Drive permission prompt (cell b).
4. Keep the tab open until the last cell prints the zip path. Colab's free tier disconnects
   idle sessions after ~90 min without interaction, so check in now and then.

Nothing needs editing. The settings cell at the top (repo, branch `deck/refinement`, Drive
folder `cal_tta_features`) is there for when the lead asks for a change.

## What it does

| Cell | Step | Typical time (estimated) |
|---|---|---|
| a | checks a GPU is attached | seconds |
| b | mounts Drive; output folder `MyDrive/cal_tta_features/` | seconds |
| c | clones `deck/refinement` | seconds |
| d | installs exactly `torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` (the `requirements.txt` pin the extractor insists on) and the rest of `requirements.txt`; runs a GPU matmul | 3–6 min |
| e | downloads the 5 datasets (~4.4 GB) + CLIP ViT-B/16 to the Colab disk and verifies them | 10–30 min |
| f | runs `scripts/extract_features.sh` per dataset with its defaults (test/val/train16, 10 augmented train16 views, seeds 0/1/2); each finished dataset is copied to Drive | 30–60 min |
| g | GPU tests: `tests/test_feature_replay.py`, `tests/test_device_agnostic.py`, `tests/test_autocast_emulation_gpu.py` | 1–2 min |
| h | writes `MyDrive/cal_tta_features/features_<date>.zip` | 1 min |

Total: about 1–1.5 h. These are estimates; the notebook has not been timed on Colab yet.

## What success looks like

- Cell d prints `"torch": "2.11.0+cu128"`, `"gpu": "Tesla T4"`, `"matmul_finite": true`.
- Cell e ends with a table with `OK` on every row, test counts
  1692 / 2463 / 3669 / 8100 / 3333, `missing 0`, and
  `All requested datasets match the contract in docs/DATASETS.md.`
- Cell f prints, per dataset, `stream order matches recorded run: n/a (no save_results.pt)`
  (expected on Colab) and `[<dataset>] saved to Drive`.
- Cell g ends with `passed` and no `failed`. Copy the two `CUDA autocast vs emulation:` lines
  to the lead.
- Cell h prints `/content/drive/MyDrive/cal_tta_features/features_<date>.zip (~120 MB)
  datasets: dtd, flower102, pets, eurosat, aircraft`.

Then tell the lead the zip is on Drive (or share it). The zip holds `features/<dataset>/...`
with each dataset's `meta.json`, `features/extract_features.log`, `pytest_gpu.txt`,
`colab_run.json` (GPU, torch, commit, timings), the dataset verification report and the
download log.

## When something goes wrong

**Disconnected / runtime restarted / closed the tab.** Reconnect, choose the T4 again if asked,
and **Run all** again. Datasets already in `MyDrive/cal_tta_features/work/features/` (they have a
`meta.json`) are skipped; only the rest are downloaded again (the Colab disk is wiped on
disconnect) and extracted again.

**Cell a: "No GPU is attached".** Do step 2 above. If Colab says it cannot connect to a GPU
backend, the free GPU quota is used up for now; try again in a few hours.

**Cell c: "branch deck/refinement lacks ...".** The extractor or downloader isn't merged yet.
Ask the lead which branch to use and put it in `BRANCH` in the settings cell.

**Cell d: torch install fails or the version check fails.** Rerun the cell once (PyPI hiccups
happen). If it still fails, send the lead the cell output; do not pass `--force` to the extractor.

**Cell e: a dataset mirror is down.** The downloader already tries a second mirror (Oxford
`thor` → `www.robots.ox.ac.uk`; EuroSAT Zenodo → `madm.dfki.de`). Wait and rerun the cell; it
resumes partial downloads. As a last resort, upload the archive under its original name
(`dtd-r1.0.1.tar.gz`, `102flowers.tgz`, `images.tar.gz`, `annotations.tar.gz`,
`EuroSAT_RGB.zip`, `fgvc-aircraft-2013b.tar.gz`) into `/content/data_root/_archives/` with
Colab's file browser, then rerun: a file found there is used instead of downloading.

**Cell e: Google Drive quota ("Too many users have viewed or downloaded this file
recently").** This concerns the small CoOp split JSONs. Wait an hour and rerun, or open
`https://drive.google.com/file/d/<ID>/view` (IDs in `docs/DATASETS.md`), download it, and
upload it as `/content/data_root/_splits/<folder>/<file name>` (e.g.
`_splits/dtd/split_zhou_DescribableTextures.json`), then rerun.

**Cell e: verification fails.** Rerun the cell; if the same files are still missing, send the
lead the table.

**Cell f: an error inside the extractor.** Send the lead the last ~30 lines. Rerunning resumes:
finished datasets are skipped, and the extractor keeps finished pieces of the current one
while the session lasts.

**Cell g: a test fails.** The zip is still made. Tell the lead; the output is in
`pytest_gpu.txt` inside the zip.

**Your Drive is full.** The run needs ~250 MB on Drive (work folder + zip). Free some space and
rerun cell h (or Run all).
