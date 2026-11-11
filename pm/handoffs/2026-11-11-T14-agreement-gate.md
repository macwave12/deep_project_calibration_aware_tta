# 2026-11-11 — T14 overnight: diagnosis, agreement-gated cache, results (lead, local)

Branch `deck/refinement`. **Nothing committed since the three `git am` patches**, so all of the
work below is in the working tree. Status: T07 review, T14 review.

## What was done
- **Decisions** (DECISIONS 2026-11-10 rows):
  - P2 is the headline and P1 goes to the appendix;
  - the α floor is CLIP + negative cache;
  - T14 order adapted;
  - T14 pre-registration and two amendments (joint temperature; Idan's dominance rule);
  - method frozen before test.
- **Dev datasets, end to end on CPU** (no Colab needed):
  - downloader specs (`DEV_SOURCES`; opt-in, the five-dataset contract is unchanged) and nested
    archives (Caltech);
  - the ImageNetV2 label fix (`data/datautils.py imagenetv2_numeric_labels`: upstream's
    ImageFolder sorts the class folders 0..999 as strings);
  - dev paths moved under DATA_ROOT; `DEV_DATASETS` / `ALL_DATASETS`;
  - extractor `class_names`.
  - Data lives in `university/data_root_dev/` (outside the repo). Food101's test split is a fixed
    10k subset (`split_zhou_Food101.json`; the full file is kept as `.full.json`).
- **Method:** `CalTdaConfig.agreement_gate` (+ `agreement_temperature`, unused in the frozen
  method), configs `cal_tda_agree.yaml` and `cal_tda_agree_temp.yaml` (rejected variant).
- **Analysis:**
  - `analysis/adaptive_alpha.py`: signals, oracle, bin-optimal α, ladder, gate variants with LODO;
  - `analysis/t14_results.py`: α-curve, matched comparison, TDA + T, mechanism table, figure;
  - `analysis/t14_dev.py`;
  - `analysis/p1_appendix.py`;
  - 90% CIs in `analysis/seed_stats.py` (n ≥ 10);
  - `scripts/make_permutations.py` (seeds 3–9);
  - `scripts/replay_dev.sh`.
- **Runs** (replay_results/, git-ignored):
  - P2 headline + gate, 10 seeds;
  - P1 gate, 3 seeds;
  - TDA full traces on val and test;
  - dev sets, P1/P2 × 3 seeds.

## How it was verified
- Full suite: 343 passed / 16 skipped before the last additions. New tests:
  - online gate == offline rule;
  - temperature only touches gated samples;
  - dev specs opt-in;
  - ImageNetV2 labels;
  - nested archive;
  - permutations;
  - CI;
  - P1 appendix;
  - T14 analysis helpers.
- The online val gate reproduces the offline numbers (56.36 / 5.78 vs 56.38 / 5.78; the
  difference is float32 vs float64 argmax ties).

## Results
See `pm/research/T14-agreement-gate.md` and KNOWLEDGE K38–K42. Headline: test, P2, 10 seeds —
gate 56.66 / 5.51 vs TDA 56.75 / 10.96. Δ acc −0.09 [−0.13, −0.04], Δ ECE −5.45
[−5.54, −5.37]. On the 5-dataset mean the gate also beats CLIP on both metrics; per dataset
its ECE is worse than CLIP's on Aircraft, Flowers and Pets.

## Final state (02:40)
- **Dev sets, all four** (P2, 3 seeds): gate 76.78 / 3.27 vs TDA 76.79 / 7.35. ImageNetV2 and
  UCF101 are clear wins, Food101 a tie, Caltech101 a failure (underconfident CLIP; K42–K43).
  The write-up now says this plainly.
- **Independent reviewer:** 15 findings, all addressed (write-up §10). Main fixes: dev claims
  stated as measured; K38 corrected; unsourced val numbers replaced by tracked CSVs
  (`results/analysis/t14/val_online.csv`, `adaptive_alpha/agree_val.csv`); the NLL ladder run
  for real (`ladder_lodo_summary_val.csv`); the ImageNetV2 extractor restricted to test; the
  Food101 subset reproducible (`scripts/make_food101_dev_subset.py`, byte-identical to the
  file used).
- **Full suite:** 346 passed / 16 skipped.
- **Proposed commit split** (none made):
  1. `feat(data)`: dev datasets (download_datasets, datautils, fewshot_datasets, feature_cache,
     extract_features, make_food101_dev_subset, tests/test_dev_datasets.py);
  2. `feat(trace)`: full-vector trace + `--trace` in the replay (cal_tda_trace, cal_tda trace
     bits, online_tta, replay_runner, replay.py, tests);
  3. `feat(method)`: agreement gate (cal_tda, configs/method/cal_tda_agree*.yaml, tests);
  4. `feat(analysis)`: protocols, seed_stats CIs, adaptive_alpha, t14_results, t14_dev,
     p1_appendix, make_permutations, replay_dev.sh, tests;
  5. `results`: results/protocols*.csv, results/analysis/**, the figure, literature_values.csv;
  6. `pm`: DECISIONS, KNOWLEDGE, PLAN, STATUS, tasks, handoffs, research note, .gitignore.
  Leave out the untracked Colab leftovers at the repo root (colab_run.json etc.).

## Open / for Idan
- Review the report. In particular, check the two disclosed criterion amendments (DECISIONS):
  the change from NLL to the dominance rule was made after seeing val results.
- Confirm the per-method MAE definition (earlier handoff).
- Commit: the work is ready to split into about 6 commits (dev datasets; trace; gate; analysis;
  results CSVs; pm). Then push `deck/refinement`.
- Reviewer agent on T07 + T14 (RULES 13). Then deck rebuild (T10) to the new storyline.
- Limitations to carry into the deck: Pets and Caltech101 (underconfident CLIP), the gated
  branch's remaining overconfidence, and design on val of the same datasets.
