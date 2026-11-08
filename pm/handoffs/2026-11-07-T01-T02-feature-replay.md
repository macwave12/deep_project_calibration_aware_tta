# 2026-11-07 — T01 feature extraction + T02 CPU feature replay (engineering agent)

Branch `worktree-agent-ace5471a3fbddc65c`, fast-forwarded to `deck/refinement` (77303a5) before
any work. Both tasks are at `review`. T01's GPU run and T02's real-data acceptance are still to
come. Revised the same day after the independent review: the lead's items 1–10 and nits.

## What was done

| Commit | Change |
|---|---|
| `b5a3be0` fix(methods) | `compute_cache_logits` (`.cuda()` ×2), `OnZeta.rho`, and `DMN_WEAK`'s memory bank now use the input tensor's device. Besides the protocol switch, these are the only edits to method files. |
| `28159c4` feat(tda) | `--protocol {published,clean}`, resolved in `TDA.prepare_model_and_optimization`. `CalTDA` copies `pos_alpha/pos_beta` after `super()`, so `clean` gives α=2, β=5 to both methods. The negative cache is unchanged. `published` is the default. |
| `548985f` feat(features) | T01 core: `utils/feature_cache.py`, `utils/feature_extraction.py`, `scripts/extract_features.{py,ps1,sh}`. |
| `638ce25` feat(replay) | T02 core: `utils/feature_replay.py` (method side, label-free), `utils/replay_runner.py`, `scripts/replay.py`, `analysis/compare_replay.py`. |
| `c00c1db` docs(pm) | First handoff, README section, KNOWLEDGE K22–K26. |
| `841bc29` feat(runner) | Item 5. `online_tta.py` moves its parser into `build_parser()` and `main(argv)` (behaviour unchanged). `default_run_name()` gives tda/cal_tda a `_clean` record name under P2; clipzs keeps its name. `run_one.{ps1,sh}` gain `-Protocol`/`--protocol` and suffix both the run name and the checkpoint folder. The replay uses the same naming, and its clean suite has no duplicate clipzs run. `--autocast off` records go to `autocast_off/`. |
| `da0da68` feat(replay) | Items 7, 8, 10 and nits. The fp16-list ops are emulated as fp16-rounded inputs, fp32 compute, one rounding to fp16. An op report (fp16 / fp32 / neutral / unlisted) is written into each record, and an unlisted op warns. `compare_replay` also diffs `hyperparams` and maps `autocast_off/` onto the GPU reference. There is a GPU-only emulation test. `test_no_label_leak` now covers `tda.py` and `clipzs.py`, and the vacuous check is gone. |
| `c4ab7bf` feat(features) | Items 1–4 and 9, plus nits: the recorded-run order check (in the job, and on CPU via `analysis/check_stream_order.py`), the torch-pin guard, preflight, resume with extend/redo, 10 Tip-Adapter train16 views, val seeds 0/1/2, a real `free_memory`, the renamed consistency check, and location-independent logging wrappers. |
| `e158422` test(features) | Item 6. The real `online_tta.main()` and `extract_features.main()` reach the stream draw in the same RNG state with the same call arguments. |

## How it was verified (all on CPU; no GPU anywhere)

- `python3 -m pytest -q`: **193 passed, 14 skipped** (111 / 19 before this work). The skips are 10
  `DATA_ROOT` split checks and 4 GPU-only tests. My test files take about 25 s of CPU time; the
  whole-suite wall time here varied from 10 s to 2.5 min because other agents' jobs shared the 2 CPUs.
- **Device fix:** a verbatim copy of the old `compute_cache_logits` is bit-identical to the new one.
- **Protocol:** the override test is parametrized over no attribute / `published` / `clean`, on
  DTD and eurosat, for tda and cal_tda. An explicit `published` stream is bit-identical to the default.
- **Real-runner RNG path** (`tests/test_real_runner_rng_path.py`): with `get_coop`/`build_dataset`
  replaced by RNG-drawing stubs, `online_tta.main()` and `extract_features.main()` reach the shuffled
  loader with the same `torch.get_rng_state()` and the same call arguments, for clipzs/tda/cal_tda at
  seeds 0/1/2. A control shows that one extra random draw changes the state.
- **T01:** permutations equal a real `DataLoader(shuffle=True)`'s order after the same steps. Tests also cover:
  - features in dataset order, dtype kept, labels only in `labels_<split>.pt`;
  - resume: skip, extend (the added permutations equal a fresh draw) and redo on an identity change;
  - preflight skipping and recording unavailable splits;
  - deterministic augmented views, raw and of shape (views, N, D);
  - `free_memory` called for every discarded model.
- **Recorded-run check** (`tests/test_recorded_run_check.py`): runs fabricated by the real loop are
  confirmed with their seed. A swapped pair of steps, or clipzs logits off by 0.5, give `False`.
  With no run present the verdict is `None`.
- **T02:** replay is bit-identical to driving the method directly (clipzs, tda, cal_tda variants;
  fp32 and emulated). Stub extraction followed by replay reproduces a hand-written `online_tta`
  loop exactly. The emulated matmul equals `fp16(fp32(fp16 a) @ fp32(fp16 b))`. For
  clipzs/tda/cal_tda the op report contains only matmul, softmax and neutral ops; ecalp's sparse
  ops show up as unlisted and warn.
- **Hyperparameter diff:** for all 85 GPU records (25 headline, 40 ablation, 20 ablation_gate), the
  hyperparams a replay would record match the stored ones. Fields added after a record was written
  count as their default.
- **Real code path smoke test** (scratch): `bash scripts/extract_features.sh --allow-cpu --force`
  with a tiny random-weight CLIP TorchScript checkpoint and a fake DTD tree, using the real
  `get_coop`/`build_dataset`.
  - Preflight skipped train16 (the fake DTD has no train images) and recorded it.
  - A trailing `/` in `DATA_ROOT` worked, the log was written, and an identical re-run was skipped.
  - Stream order and features matched the runner's own path sample by sample.
  - Without `--force` the job refuses torch 2.15 against the 2.11.0+cu128 pin.
- **Wall time** (from the first round): a five-dataset `tda` sweep on synthetic data of the real
  sizes took 3.6 min on CPU with emulation.

## Assumptions the real-data check must confirm

1. **Stream order.** The extractor repeats `set_random_seed` → `build_dataset` → `get_coop` → `.cuda()`
   before every draw. It matches the original runs only under the same torch and model code. The
   job now refuses a non-pinned torch, and the recorded-run check confirms the order exactly from
   `save_results.pt`, when that file is on disk.
2. **dtype.** Code analysis says the image/text features are fp32 (K22), not fp16 as T01's text
   states. The augmented views are raw `image_encoder` outputs, probably fp16 under autocast.
   `meta.json` records both.
3. **Numerics.** The emulation models cuBLAS's fp16 GEMM; summation order can still differ by
   an fp16 ulp. `tests/test_autocast_emulation_gpu.py` measures this on the GPU machine.
   Expect agreement within tolerance, not bit-identity. `--autocast off` is the A/B check.
4. **CPU-extracted features** (if T01 runs with `--allow-cpu`) are fp32 rather than
   autocast-computed. The order is still checked exactly, by labels. Whether replaying them still
   reproduces `summary.csv` within 0.2 / 0.3 points is untested: CPU rounding is a slightly
   different numeric path from the GPU runs. If a dataset misses, that is the first suspect.

## Open questions

- Aircraft val/train16 need `images_variant_{val,train}.txt`. If they are missing, preflight now
  skips those splits with one message instead of crashing, and `meta.json` records it.
- The recorded-run check uses the newest `save_results.pt` per method. That file holds the last
  run with that checkpoint tag; its seed is found by matching against the extracted seeds.
- Val sizes (≈1128 / 1633 / 736 / 5400 / 3333) come from the CoOp split files and are not verified here.
- dmn_weak, ecalp and onzeta replay but are not scored (T08). LOO-temperature variants are slow on
  CPU until T03.

## For macwave: run T01 (one-shot GPU job)

Estimated 30–45 min, not measured: about 84k image passes at batch size 1 including the 10
augmented train16 views, plus about 30 model builds for the permutations. Run from any directory,
with the project venv active, torch **2.11.0+cu128** installed, and the CLIP checkpoint where
`get_coop` finds it (`CLIP_CACHE_ROOT`, default `D:/second_degree/deep/.cache/clip/`).

**0. Optional but useful (2 min): GPU sanity tests**

```powershell
python -m pytest -s tests/test_feature_replay.py tests/test_device_agnostic.py tests/test_autocast_emulation_gpu.py
```
Please paste the two `CUDA autocast vs emulation:` lines.

**1. Extract**

PowerShell:
```powershell
git pull
.\scripts\extract_features.ps1 -DataRoot D:\second_degree\deep\data_root
```
bash:
```bash
DATA_ROOT=/path/to/data_root bash scripts/extract_features.sh
```
Direct fallback, if a wrapper misbehaves (run from the repo root):
```powershell
python scripts\extract_features.py --data-root D:\second_degree\deep\data_root --out features
```

- It prints the torch version first and stops if it isn't the pin (`--force` overrides it, but tell the lead).
- Then a preflight line per dataset (skipped splits are named).
- Then per dataset `stream order matches recorded run: True/False/n/a`. `False` means stop and
  report; `n/a` means no `save_results.pt` was found.
- The output is also in `features\extract_features.log`.
- If it stops, run the same command again: finished pieces are kept.
- Do not add `*>&1`.

**Expected output** (about 125 MB: about 74 MB of fp32 centre-crop features plus about 48 MB of fp16 augmented views):
```
features/<dtd|flower102|pets|eurosat|aircraft>/
  test_feats.pt  val_feats.pt  train16_feats.pt      (N, 512), dataset order
  train16_aug_feats.pt                               (10, N16, 512), raw encoder outputs
  text_feats.pt
  perm_test_seed{0,1,2}.pt  perm_val_seed{0,1,2}.pt
  labels_test.pt  labels_val.pt  labels_train16.pt
  meta.json
features/extract_features.log
```

### Alternative: extract on a CPU-only Windows machine (`--allow-cpu`, supported)

Use this when no GPU is available. The commands are the same, plus `--allow-cpu`. Environment set-up is
in `docs/COLAB_EXTRACTION.md` (another agent maintains it).

- **torch build:** the guard expects the CPU build of the pin, **`2.11.0+cpu`**, and prints
  `mode: CPU (fp32, no autocast); expected torch 2.11.0+cpu`. Any other version stops the job
  unless you pass `--force`; if you do, tell the lead.
- **What differs:** without CUDA, CLIP loads in fp32 and autocast is off, so the features are fp32
  computed in fp32. They differ in the last bits from what the GPU runs saw. The stream order does
  not depend on the device: it comes from the same seeded CPU generator, and
  `test_real_runner_rng_path.py` covers that path. In this mode the recorded-run check judges the
  order by labels only and reports the clipzs logit comparison as informational
  (`"informational": true`), so CPU rounding cannot produce a false "mismatch". `meta.json` records
  `identity.device = "cpu"`. A later GPU extraction into the same folder redoes it rather than mixing
  the two.
- **Speed:** use `--batch-size 32` on CPU. The batch-1 default only matters for matching cuBLAS
  kernel shapes on the GPU; on CPU, batching changes fp32 results only in the last bits and is
  faster. `--workers 2` can overlap JPEG decoding with compute.
- **Runtime caveat (estimated):** the ViT-B/16 image encoder measured 0.34 s/image at batch 32
  (0.43 s at batch 1) on a contended 2-vCPU cloud box, with random weights (same cost). A modern
  8-core desktop should be several times faster. Expect about **1–4 h** for the centre-crop pass
  (about 36k images) and about as long again for the 10 augmented train16 views (about 47k passes).
- **Run the augmented views last, as a second pass.** Resume adds only what is missing:

```powershell
# pass 1: centre-crop features, all permutations (the part T02 needs)
.\scripts\extract_features.ps1 -DataRoot D:\second_degree\deep\data_root --allow-cpu --batch-size 32 --train16-views 0
# pass 2 (optional, for Tip-Adapter / T08): adds train16_aug_feats.pt, keeps everything else
.\scripts\extract_features.ps1 -DataRoot D:\second_degree\deep\data_root --allow-cpu --batch-size 32
```
bash: `DATA_ROOT=... bash scripts/extract_features.sh --allow-cpu --batch-size 32 --train16-views 0`,
then the same without `--train16-views 0`.

Both passes must use the same `--batch-size`, which is part of the folder's identity; changing it
redoes the folder. Step 0's GPU tests are skipped on CPU and the other tests run as usual.

**2. Bundle and share** (features plus the recorded runs, for the exact CPU check). Not for git.

PowerShell:
```powershell
$stage = "t01_bundle"
New-Item -ItemType Directory -Force "$stage\online_results\ckps" | Out-Null
Copy-Item -Recurse -Force features "$stage\"
Copy-Item -Recurse -Force online_results\ckps\clipzs, online_results\ckps\tda "$stage\online_results\ckps\"
Compress-Archive -Path "$stage\*" -DestinationPath t01_bundle.zip -Force
```
bash:
```bash
zip -r t01_bundle.zip features online_results/ckps/clipzs online_results/ckps/tda
```
Upload `t01_bundle.zip` to Google Drive (or as a release asset) and paste the last ~20 log lines
into the thread.

## Next person (T02 real-data check, CPU)

```bash
unzip t01_bundle.zip                       # at the repo root -> features/, online_results/ckps/
python analysis/check_stream_order.py      # exit 0 = exact per-sample order match on CPU
python scripts/replay.py --suite headline
python scripts/replay.py --suite ablation
python scripts/replay.py --suite ablation_gate
python analysis/compare_replay.py          # exit 0 = within 0.2 acc / 0.3 ECE pts, hyperparams equal
```
Any gap blocks Phase C until it is explained (RULES §7). For T02's wall-time figure, report the
`hyperparams.replay.seconds` of each record.
