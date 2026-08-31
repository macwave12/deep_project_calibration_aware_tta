# Datasets

All five datasets are downloaded, extracted and verified. **You do not need to download anything.**

## Local root

```
DATA_ROOT = D:\second_degree\deep\data_root
```

Set it before running anything:

```powershell
$env:DATA_ROOT = "D:\second_degree\deep\data_root"
```

It lives outside the repo and is git-ignored. `_archives\` (4.4 GB of source tarballs) and
`_splits\` (the pristine CoOp split files) are kept so the layout can be rebuilt without
re-downloading; both are safe to delete once you are confident in the setup.

## Verified state

Every file referenced by each test split was checked to exist on disk on 2026-08-16:

| `--test_sets` key | Folder | Test images | Verified | Total on disk |
|---|---|---:|---|---:|
| `dtd` | `dtd/` | 1692 | 0 missing | 5640 |
| `flower102` | `oxford_flowers/` | 2463 | 0 missing | 8189 |
| `pets` | `oxford_pets/` | 3669 | 0 missing | 7390 |
| `eurosat` | `eurosat/` | 8100 | 0 missing | 27000 |
| `aircraft` | `fgvc_aircraft/` | 3333 | 0 missing | 10000 |

Total evaluated per method: **19,257 images**. These counts are the contract — if a loader
returns a different number, the loader is wrong, not the table.

## Layout

Upstream `tta-vlm` follows CoOp's convention. Subfolder names are hardcoded in
`data/fewshot_datasets.py` (`jpg` for flowers, `2750` for eurosat, `images` elsewhere), so
they are not free to rename.

```
data_root/
├── dtd/
│   ├── images/                              (47 texture classes)
│   ├── labels/  imdb/
│   └── split_zhou_DescribableTextures.json
├── oxford_flowers/
│   ├── jpg/                                 (8189 images, 102 classes)
│   ├── imagelabels.mat  setid.mat  cat_to_name.json
│   └── split_zhou_OxfordFlowers.json
├── oxford_pets/
│   ├── images/  annotations/                (37 breeds)
│   └── split_zhou_OxfordPets.json
├── eurosat/
│   ├── 2750/                                (10 land-use classes, RGB)
│   └── split_zhou_EuroSAT.json
└── fgvc_aircraft/
    ├── images/                              (10000 images, 100 variants)
    └── images_variant_test.txt  variants.txt  families.txt  ...
```

`fgvc_aircraft` has **no** `split_zhou` file — upstream special-cases it and reads
`images_variant_test.txt` (3333 lines) directly.

## Sources

| What | URL |
|---|---|
| DTD | `https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz` |
| Flowers 102 images | `https://thor.robots.ox.ac.uk/flowers/102/102flowers.tgz` |
| Flowers 102 labels | `https://thor.robots.ox.ac.uk/flowers/102/imagelabels.mat`, `.../setid.mat` |
| Oxford Pets | `https://thor.robots.ox.ac.uk/pets/images.tar.gz`, `.../annotations.tar.gz` |
| EuroSAT (RGB) | `https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip` |
| FGVC Aircraft | `https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz` |

The EuroSAT copy comes from Zenodo rather than the frequently-offline `madm.dfki.de` mirror;
the archive's top-level `EuroSAT_RGB/` was renamed to `2750/` to match what upstream expects.

### CoOp split files (Google Drive)

These define the exact test splits the benchmark uses — the datasets' own native splits give
different counts (e.g. DTD's native test set is 1880, not 1692), so these files are required
for our numbers to be comparable to the benchmark paper.

| File | Drive ID |
|---|---|
| `split_zhou_DescribableTextures.json` | `1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x` |
| `split_zhou_OxfordFlowers.json` | `1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT` |
| `cat_to_name.json` | `1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0` |
| `split_zhou_OxfordPets.json` | `1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs` |
| `split_zhou_EuroSAT.json` | `1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o` |

Fetch pattern: `https://drive.google.com/uc?export=download&id=<ID>`

## Upstream patches that must be reapplied after any merge

1. **`data/fewshot_datasets.py`** ships `path_dict` with the original author's absolute paths
   (`/data/shenglijun/dataset/few-shot-datasets/...`). Nothing loads until these are
   repointed at `DATA_ROOT` (see `path_dict` and the `DATA_ROOT = os.environ.get(...)` line
   near the top of that file for the fix already applied in this fork).
2. **`clip/constants.py`, `online_tta.py`, `instance_tta.py`** contain a `your_cache_path`
   placeholder that must be replaced with a real folder (where CLIP weights are cached).
