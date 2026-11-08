# Local CPU setup (Windows, no GPU)

How to prepare a Windows 10/11 machine with miniconda3 and no GPU for the one-time feature
extraction, without anyone else's machine. All commands are PowerShell, run from the repo
root. Every step can be rerun safely: finished work is skipped.

| | Download | Disk | Time (typical) |
|---|---:|---:|---|
| Python env (torch CPU + deps) | ~0.4 GB | ~1.5 GB | 5–10 min |
| Datasets (5 archives + split files) | ~4.4 GB | ~6 GB, ~10 GB with `-KeepArchives` | 20–90 min download, 5–15 min extraction |
| CLIP ViT-B/16 weights | ~335 MB | ~335 MB | 1–5 min |

Aircraft alone is 2.6 GB; the Oxford servers are often the slow part.

## 0. Once per machine

Allow local scripts for your user (or prefix every script call with
`powershell -ExecutionPolicy Bypass -File`):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Use a **short** data folder on a drive with ≥ 12 GB free, e.g. `D:\cal_tta\data_root`.

## 1. Environment

```powershell
.\scripts\setup_local_cpu.ps1 -DataRoot D:\cal_tta\data_root -ClipCache D:\cal_tta\clip_cache
```

This creates the conda env `cal-tta-cpu` (Python 3.12, conda-forge), installs
`torch==2.11.0` / `torchvision==0.26.0` CPU wheels from `https://download.pytorch.org/whl/cpu`
and then `requirements-cpu.txt` (= `requirements.txt` without its CUDA lines), stores
`DATA_ROOT` and `CLIP_CACHE_ROOT` in the env, and runs `python -m pytest -q`.
Without `-DataRoot`/`-ClipCache` it uses `%USERPROFILE%\cal_tta\data_root` and
`%USERPROFILE%\cal_tta\clip_cache`.

Expected end: pytest reports `passed` with some `skipped` (the dataset split checks skip
until the data exists) and the script prints next steps. Then:

```powershell
conda activate cal-tta-cpu
$env:DATA_ROOT; $env:CLIP_CACHE_ROOT     # both must print the folders above
```

`DATA_ROOT` is read by `data/fewshot_datasets.py` (split files); `CLIP_CACHE_ROOT` by
`clip/constants.py` (where `ViT-B-16.pt` is cached). If you ever run outside the env, set
them by hand: `$env:DATA_ROOT = "D:\cal_tta\data_root"`.

## 2. Datasets and CLIP weights

```powershell
.\scripts\download_datasets.ps1 -WithClip
```

Interrupted, or one dataset failed? Run the same command again — downloads resume from the
`.part` file and finished datasets are skipped. Useful options:

```powershell
.\scripts\download_datasets.ps1 -Datasets dtd,eurosat     # a subset
.\scripts\download_datasets.ps1 -KeepArchives             # keep _archives\ (+4.4 GB) for rebuilds
.\scripts\download_datasets.ps1 -VerifyOnly               # just re-check the layout
```

It produces the exact layout in `docs/DATASETS.md` (EuroSAT's `EuroSAT_RGB\` becomes `2750\`,
Aircraft's `fgvc-aircraft-2013b\data\` is flattened into `fgvc_aircraft\` with its own
`images_variant_{train,val,trainval,test}.txt`; the CoOp `split_zhou_*.json` files come from
Google Drive, a pristine copy is kept in `_splits\`).

## 3. Verify

The downloader ends with a table; this is what "ready" looks like:

```
dataset    folder            test contract    val  train missing  status
dtd        dtd               1692     1692   1128   2820       0  OK
flower102  oxford_flowers    2463     2463   1633   4093       0  OK
pets       oxford_pets       3669     3669    736   2944       0  OK
eurosat    eurosat           8100     8100   5400  13500       0  OK
aircraft   fgvc_aircraft     3333     3333   3333   3334       0  OK
total test images: 19257
```

The test counts and `missing = 0` are the contract (any mismatch fails the run). The
val/train numbers are the CoOp reference sizes; a difference there is printed as a `note`.
The full report is saved as `DATA_ROOT\_state\verify.json`. Independent check through the
real loaders:

```powershell
python -m pytest -q tests/test_dataset_splits.py      # 10 passed
```

## 4. Run the extraction

*Placeholder — the CPU mode is being added by another task.* The entry point will be:

```powershell
.\scripts\extract_features.ps1 -AllowCpu
```

See that script's header for its options. Expect it to take far longer on CPU than the
~10–15 min it takes on a GPU.

## Troubleshooting

**Google Drive: quota / "can't scan for viruses" page.** The confirm page is handled
automatically. "Too many users have viewed or downloaded this file recently" is a quota
limit: wait a few hours, or open `https://drive.google.com/file/d/<ID>/view` (IDs in
`docs/DATASETS.md`), download the file in the browser and save it as
`DATA_ROOT\_splits\<folder>\<file name>` (e.g. `_splits\dtd\split_zhou_DescribableTextures.json`),
then rerun. The script validates it and copies it into place.

**Oxford mirror down / very slow.** DTD, Flowers and Pets try `thor.robots.ox.ac.uk` and
then `www.robots.ox.ac.uk`; EuroSAT tries Zenodo, then `madm.dfki.de`. If all fail, get the
archive any way you can (browser, another mirror, a teammate) and put it in
`DATA_ROOT\_archives\` under its original name (`dtd-r1.0.1.tar.gz`, `102flowers.tgz`,
`images.tar.gz`, `annotations.tar.gz`, `EuroSAT_RGB.zip`, `fgvc-aircraft-2013b.tar.gz`), then
rerun: a file found there is used instead of downloading.

**Antivirus locking or quarantining files.** Extraction writes ~60,000 small images; Windows
Defender scanning them can make it slow and briefly lock files (`PermissionError`,
`WinError 32`). The script retries; if it still fails, rerun. For speed, add `DATA_ROOT` as a
Defender exclusion (Windows Security → Virus & threat protection → Exclusions). If files go
missing later, `-VerifyOnly` will say which; delete that dataset's folder and rerun.

**Long paths.** Keep `DATA_ROOT` short (e.g. `D:\cal_tta\data_root`), not deep inside
OneDrive or Documents. If you get `FileNotFoundError` on long paths, enable long paths
(admin PowerShell, then reboot):
`New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`.

**OneDrive-synced folders.** Don't put `DATA_ROOT` under a OneDrive folder: syncing
thousands of images interferes with extraction and uses your cloud quota.

**`conda` not found / env issues.** Use the "Anaconda PowerShell Prompt" or run
`conda init powershell` once and reopen PowerShell. To start over:
`conda env remove -n cal-tta-cpu`, then rerun step 1.

**Disk full mid-run.** Free space and rerun; the partial download resumes. Without
`-KeepArchives`, each archive is deleted right after its dataset is laid out.
