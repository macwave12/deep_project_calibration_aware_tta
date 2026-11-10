# 2026-11-10 — Local environment, replay smoke test, seed statistics tool (lead, local)

Branch `deck/refinement`, uncommitted at the time of writing. Status `review`.

## What was done
- Local env `caltta` (conda at `%USERPROFILE%\miniconda3`, not on PATH; Python 3.12, CPU torch
  2.11.0) per the lead handoff, step 1. Features unzipped to `features/` (git-ignored).
- `tests/test_download_datasets.py`: `read_text()` / `open()` given `encoding="utf-8"`. On a
  cp1255 Windows locale, `test_real_spec_matches_documented_contract` failed to decode
  `docs/DATASETS.md`.
- `analysis/seed_stats.py` + `tests/test_seed_stats.py` (8 tests, < 2 s). It reads
  `<raw>/*.json` (seed 0) and `<raw>/seed<k>/*.json` and writes
  `results/analysis/seed_stats_<name>_methods.csv` (per dataset × method × metric: n, seeds, mean,
  std ddof=1, MAE) and `..._paired.csv` (method − reference, paired by seed: mean, std, MAE of the
  difference, wins/ties/losses; a win is higher accuracy or lower ECE). It also writes a
  `mean` dataset row: per seed, the unweighted mean over all datasets, using only seeds that
  have every dataset. The defaults compare against `tda` and `clipzs`.
- Replayed the headline suite for seeds 1 and 2 (`replay_results/raw/seed{1,2}/`). The
  headline CSVs are written from seeds 0/1/2.

## Definition chosen (needs Idan's confirmation)
- The policy says "per method mean, std, MAE" without defining MAE for a single method. I
  implemented it as the **mean absolute deviation from the seed mean**, a spread measure that
  does not depend on n the way std does. For pairs, MAE is the mean |difference|. If a
  different MAE was meant (e.g. against the recorded GPU number), only `method_stats` changes.

## How it was verified
- Full suite: 310 passed / 14 skipped (CUDA- and data-only).
- Smoke test: `scripts/replay.py --suite headline` (seed 0, 179 s) and then
  `analysis/compare_replay.py`. 18/25 records are within tolerance, and the misses are exactly
  those in K34/K35.
- Seed 1/2 EuroSAT numbers match K35: TDA 55.54/52.41, gate 47.30/46.60.

## What the numbers say (3 seeds, research-phase; not final)
- `cal_tda_gate` vs TDA, mean over datasets: accuracy −1.82 (0/3 wins), ECE −6.60 (3/3 wins).
  Most of the accuracy cost is EuroSAT (−7.31, 0/3); DTD −1.30 and Flowers −0.84 also lose
  on every seed. On Pets, gate *raises* ECE (+2.85, 0/3): there TDA is already better
  calibrated (2.65) than CLIP (4.36).
- `cal_tda_temp` vs TDA: accuracy unchanged (−0.02), ECE −2.36 (3/3). That mean hides Pets
  +12.89 (0/3) and DTD's high spread (ECE std 1.43).
- Gate and gate+temp have identical accuracy, as expected: temperature does not change argmax.
- These match the existing story, but they make it concrete: the gate trades accuracy for
  calibration, and the trade is not small on EuroSAT. That belongs on the limitations slide
  and in T07.

## Open / next
- Replaying seeds 1+2 took 3852 s for 50 runs, against 179 s for 25 runs at seed 0. I did
  not investigate why. Possible causes are CPU contention or throttling, or the pytest run
  that overlapped the start. Plan for that when budgeting the 10-seed run.
- Seeds ≥ 3 need CPU-generated permutations (DECISIONS 2026-11-10), which are not built yet.
  `seed_stats.py` already accepts any seed list.
- Not run yet: the ablation suites with seeds 1/2 (`--suite ablation --seed 1 2`, then
  `seed_stats.py --raw-dir replay_results/raw/ablation --name ablation`).
- Untracked files left at the repo root, from earlier sessions: `colab_run.json`,
  `datasets_verify_*.json`, `download_datasets.log`, `pytest_gpu.txt`. K33 cites
  `colab_run.json`, so it should probably be tracked (or moved under `results/`).
