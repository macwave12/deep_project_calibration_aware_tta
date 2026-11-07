# 2026-11-07 — T01 feature extraction + T02 CPU feature replay (engineering agent)

Branch `worktree-agent-ace5471a3fbddc65c` (fast-forwarded to `deck/refinement` 77303a5 first).
Both tasks are at `review`. T01's GPU run and T02's real-data acceptance are still to come.

## What was done

| Commit | Change |
|---|---|
| `fix(methods)` | `compute_cache_logits` (`.cuda()` ×2), `OnZeta.rho` (`.cuda(gpu)`), `DMN_WEAK` memory bank (`device=gpu_id`) now use the input tensor's device. These are the only edits to method files besides the protocol switch. |
| `feat(tda)` | `--protocol {published,clean}`, resolved in `TDA.prepare_model_and_optimization` (`resolve_protocol`, `CLEAN_PROTOCOL`). `CalTDA` copies `pos_alpha/pos_beta` after `super()`, so both methods get α=2, β=5 under `clean`. The negative cache (0.117 / 1.0) is unchanged in both protocols. `online_tta.py` has the flag (default `published`) and records `protocol` in tda/cal_tda records. |
| `feat(features)` | T01: `utils/feature_cache.py` (layout, `stream_order`, `FeatureStore`), `utils/feature_extraction.py` (model-agnostic pipeline), `scripts/extract_features.{py,ps1,sh}`, `.gitignore` (`features/`, `replay_results/`). |
| `feat(replay)` | T02: `utils/feature_replay.py` (method side: `FeatureReplayModel`, `build_trainer`, `run_stream`, CUDA-autocast emulation), `utils/replay_runner.py` (scorer glue), `scripts/replay.py` (single runs and `--suite headline/ablation/ablation_gate`), `analysis/compare_replay.py` (RULES §7 check). |
| `docs` | README "Feature cache and CPU replay", KNOWLEDGE K19 update and K22–K26, this note, task statuses. |

## How it was verified (all on CPU, no GPU anywhere)

- `python3 -m pytest -q`: **169 passed, 12 skipped** (was 111 / 19). The 9 `test_cal_tda_runner`
  tests that were skipped for lack of CUDA now run. The 12 skips are 10 `DATA_ROOT` split
  checks and 2 GPU-only checks.
  The new test files (`test_feature_extraction`, `test_feature_replay`, `test_device_agnostic`, plus
  the label-leak additions) take about 14 s. Whole-suite wall time ranged from 10 s to 2 min during
  this session because other agents' jobs shared the 2-core box. The slow tests are the existing
  40-sample LOO tests in `test_cal_tda_runner.py`, not the new ones.
- Device fix (RULES §7): a verbatim copy of the old `compute_cache_logits`, run on CPU with
  `Tensor.cuda` patched to the identity, is **bit-identical** to the new one for the positive and
  negative caches.
- Protocol: `test_dataset_alpha_and_beta_override_the_method_config` is parametrized over
  `None` (an `args` without the attribute, i.e. every existing caller), `published` and `clean`, on DTD and eurosat, for
  `tda` and `cal_tda`. An explicit `published` stream is bit-identical to the default.
- T01 acceptance: `stream_permutation` equals the order a real `DataLoader(shuffle=True,
  num_workers=0, pin_memory=True)` yields after `set_random_seed(seed)` and the same preamble,
  for seeds 0/1/2 and val. Features are in dataset order and unchanged (dtype kept). Labels and
  file names are only in `labels_<split>.pt`. Resume and skip work.
- T02 synthetic acceptance: replaying clipzs / tda / four cal_tda configs is **bit-identical** to
  driving the method directly on the same features, both with and without autocast emulation.
  End to end (stub extraction → `replay_one` → record) reproduces a hand-written
  `online_tta.py` loop over a shuffled DataLoader **exactly** (accuracy, ECE, reliability) for
  tda (seeds 0 and 1), clipzs (seed 2) and cal_tda_gate.
- Real code path smoke test (scratch, not committed): `scripts/extract_features.py --allow-cpu`
  run against a tiny random-weight ViT CLIP TorchScript checkpoint and a fake on-disk DTD
  split. Re-running `online_tta.py`'s pre-loop steps by hand with the **real** `get_coop` /
  `build_dataset` / transform gives the same stream order and the same features, sample by sample.
- Wall time (T02 criterion): full five-dataset `tda` sweep on synthetic stores of the real sizes
  (19,257 test steps, D=512) took **3.6 min** on this contended 2-vCPU box with emulation
  (DTD 42 s with emulation, 25 s with `--autocast off`). clipzs on DTD took 12 s.

## Assumptions the real-data check must confirm

1. **RNG / stream order.** The stream depends on the global torch RNG state at the start of
   iteration. That state follows from `set_random_seed(seed)` → `build_dataset` (python `random`
   only) → `get_coop` (CLIP `__init__` draws random init weights) → `.cuda()`. The extractor
   repeats exactly these steps per (split, seed) with the real functions, building the model once
   per permutation. This is exact only if macwave runs the **same torch version and RNG-consuming
   code as the original runs** (torch 2.11.0+cu128 per README/requirements; `meta.json`
   records the version used). If the replay misses `summary.csv` on tda but matches on clipzs
   (clipzs is order-independent), the stream order is the first suspect.
2. **dtype.** The task text says features are fp16. Code analysis says fp32 (K22), because with
   fp16 features the original TDA runs would have crashed. The extractor saves whatever
   `forward_features` returns and records it in `meta.json` (`splits.*.dtype`). If it really is
   fp16, TDA's replay will raise `Half != float` in `compute_cache_logits`, which means the
   original runs could not have used that path.
3. **Numerics.** The default `--autocast emulate` reproduces CUDA autocast's casts (fp16 logits,
   fp16 accumulation in TDA, fp32 softmax). The fp16 rounding of a CPU matmul may differ in the
   last bit from cuBLAS, and the GPU ran with `cudnn.benchmark`. So bit-identity with the GPU
   records is **not** expected, only the RULES §7 tolerance. If a dataset misses, try `--autocast off`
   as the A/B check.
4. **Batch size.** Extraction defaults to `--batch-size 1` (the runs' kernel shapes). A larger
   batch is faster but may change features in the last bits.

## Open questions / surprises

- The Aircraft val split needs `images_variant_val.txt` and train16 needs
  `images_variant_train.txt` in `fgvc_aircraft/`. `docs/DATASETS.md` only lists the test file.
  If either is missing, run with `--splits test val` or `--splits test` and tell the lead.
- Val split sizes (≈1128 / 1633 / 736 / 5400 / 3333) are from the CoOp split files and are not
  verified here.
- `dmn_weak`, `ecalp`, `onzeta` replay on CPU and their raw outputs are saved, but they are **not
  scored**. `online_tta.py` never scores them either (their output scales were never inspected),
  so T08 decides how they are scored.
- `cal_tda` variants with LOO temperature are O(N²) on CPU (K18). A full `--suite headline` will be
  slow on eurosat until T03 lands. The tda/clipzs/gate-only runs are fast.
- Upstream `clip.load` only takes TorchScript archives (K24). That's irrelevant for the real
  ViT-B-16.pt and only mattered for the smoke test.

## For macwave: run T01 (≈15–25 min estimated, not measured; most of it is JPEG decoding)

From the repository root on the branch that carries this work, with the project venv active
and the CLIP checkpoint where `get_coop` finds it (`CLIP_CACHE_ROOT`, default
`D:/second_degree/deep/.cache/clip/`):

PowerShell:
```powershell
git pull
$env:DATA_ROOT = "D:\second_degree\deep\data_root"
.\scripts\extract_features.ps1 -DataRoot $env:DATA_ROOT
```

bash:
```bash
DATA_ROOT=/path/to/data_root bash scripts/extract_features.sh
```

Do **not** pipe with `*>&1` in PowerShell 5.1 (a `FutureWarning` on stderr aborts it, see README).
If it stops, run the same command again: finished datasets are skipped. The log prints, per
dataset, the first five indices of each stream and a `forward_features check` that should read
`max_abs_diff: 0.0`. Please paste the last ~20 lines of output into the handoff thread.

Expected output (`features/`, about **76 MB** if the features are fp32 as expected, about 38 MB if fp16):

```
features/<dtd|flower102|pets|eurosat|aircraft>/
  test_feats.pt  val_feats.pt  train16_feats.pt   (N, 512)
  text_feats.pt                                   text features x2, logit_scale x2
  perm_test_seed0.pt perm_test_seed1.pt perm_test_seed2.pt perm_val_seed0.pt
  labels_test.pt labels_val.pt labels_train16.pt
  meta.json
```

(19,257 test + ≈12,230 val + 4,736 train16 images × 512 × 4 B ≈ 74 MB of image features.)
Share it as a zip (`Compress-Archive features features.zip`) on Google Drive or as a release asset.
Do not commit it (`features/` is git-ignored).

## Next person (reviewer / T02 real-data check)

```bash
unzip features.zip            # -> features/
python scripts/replay.py --suite headline
python scripts/replay.py --suite ablation
python scripts/replay.py --suite ablation_gate
python analysis/compare_replay.py      # exit 0 = every record within 0.2 acc / 0.3 ECE points
```

Any gap above tolerance blocks Phase C until it is explained (RULES §7). Report the per-run
`seconds` from each record's `hyperparams.replay` as the T02 wall-time figure.
