# T01 — Feature extraction script (the only GPU job)

Status: open · Owner: engineering agent (script), macwave (run) · Depends on: —

## Goal
A script that, run once on a CUDA machine, exports everything the CPU replay needs, so no
other task ever touches images or a GPU.

## What it must export, per dataset (dtd, flower102, pets, eurosat, aircraft)
- `test`: image features in **dataset order** (shuffle=False), exactly as
  `model.forward_features(images)` returns them for `clipzs`/`tda` (same transform:
  Resize(224, BICUBIC) → CenterCrop(224) → ToTensor → CLIP normalize; fp16 autocast;
  `--ctx_init a_photo_of_a`, ViT-B/16). Keep the returned dtype (fp16) — do not upcast.
- `val`: same, CoOp val split.
- `train16`: same, `build_dataset(..., mode='train', n_shot=16)` (for Tip-Adapter).
- Text features (C × D) and `logit_scale` from the same model.
- Stream permutations for seeds 0, 1, 2 on `test` and seed 0 on `val`: reproduce the run
  path exactly — `set_random_seed(seed)`, build the model and the
  `DataLoader(..., shuffle=True)` as `online_tta.py` does, then read the order from the
  loader's sampler before any other RNG use. Record the dataset index of each stream
  position.
- Labels in a **separate file** (`labels.pt`) that only the scorer / analysis code reads.
- `meta.json`: torch/CUDA/GPU versions, git commit, transform, prompt, class names, counts.

Layout: `features/<dataset>/{test,val,train16}_feats.pt`, `text_feats.pt`,
`perm_<split>_seed<k>.pt`, `labels_<split>.pt`, `meta.json`. Total ≈ 60 MB for ViT-B/16.

## Acceptance
- CPU unit test with a stub model and a tiny fake dataset: dataset-order features,
  permutation equals the order a `DataLoader(shuffle=True)` yields after
  `set_random_seed(seed)`; labels never written into the feature files.
- A one-command PowerShell and bash entry point; README section with the exact command.
- Verified indirectly by T02: replaying `clipzs` and `tda` from these features reproduces
  `results/summary.csv`.

## Handoff to macwave
Command, expected runtime (~10–15 min on an RTX 5060), output folder, and how to share it
(zip ≈ 60 MB → Google Drive or a release asset; not committed to git).
