# T02 — CPU feature replay

Status: done (real-data acceptance 2026-11-10, K34–K35) · Owner: engineering agent · Depends on: T01 format (not its data)

## Goal
Run any single-view method (clipzs, tda, cal_tda variants, dmn_weak, ecalp, onzeta) on CPU
from cached features, with the **unchanged** method code, producing the same run records
(`results/raw/...json`) the GPU runner produces.

## Design
- `FeatureReplayModel`: stub exposing `forward_features(images)` → (image feature of the
  current stream position, text features, logit_scale), plus whatever attributes the
  methods read from `self.model`. The stream position is advanced by the runner.
- `scripts/replay.py --dataset --split --seed --method --config --protocol {published,clean}`.
  Reuses the scorer (`utils/evaluation.py`) and the run-record format.
- Device: make method code device-agnostic where it hard-codes `.cuda()`
  (`tda.py` `compute_cache_logits`, and any in ecalp/dmn_weak) by using the input tensor's
  device. This is the only allowed edit to method files here; it must not change GPU
  behaviour.
- `--protocol clean` sets α = 2, β = 5 for tda and cal_tda alike (switch in
  `prepare_model_and_optimization`; `tests/test_cal_tda_runner.py::test_dataset_alpha_and_beta_override_the_method_config`
  becomes parametrized over the two protocols).

## Acceptance
- Synthetic test: replay of tda on random features equals running TDA directly on the same
  features (bit-identical on CPU).
- With real features (after the GPU job): replay of clipzs, tda, cal_tda_gate, cal_tda_temp,
  margin and fusion variants reproduces `results/summary.csv` / ablation CSVs within
  0.2 accuracy points and 0.3 ECE points per dataset (RULES §7). Any larger gap blocks
  Phase C until explained.
- Full five-dataset tda sweep on CPU in minutes, not hours (report wall time).
