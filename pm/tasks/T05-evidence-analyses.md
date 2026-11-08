# T05 — Evidence analyses for the deck

Status: open · Owner: analysis agent · Depends on: T02, T04 (+ features)

Each item → a CSV under `results/analysis/` and a figure under `analysis/figures/`, produced
by a tested script. Labels are joined here, never in method code.

| ID | Question | Output | Deck slide |
|---|---|---|---|
| A1 | CLIP vs TDA per dataset: accuracy, mean confidence, ECE, gap sign | table + reliability diagrams | 14, 18 |
| A2 | Where does TDA's gain come from? Argmax flips vs CLIP's top-2 gap: count, correct→wrong, wrong→right | histogram per dataset | 15, gate motive |
| A3 | Anatomy of overconfidence: self-vote share; confidence lift for self-voted vs not; cache purity (wrong pseudo-labels) over the stream; class coverage | curves + table | 19 |
| A4 | A "poisoned" cached item: a wrongly pseudo-labeled cached image, the test images it pulled, with CLIP's and the cache's logit contribution | case-study figure (with thumbnails if images available, else feature-space description) | 19 |
| A5 | Margin admission: purity and class coverage with vs without; admitted fraction | table | margin slide |
| A6 | Scale issue: distribution of max softmax(c) vs max softmax(z); one worked 100-class example | histogram + example | fusion slide |
| A7 | Fusion dilution on CLIP-confident samples: reliability of the CLIP-confident subgroup before vs after fusion | subgroup reliability | fusion slide |
| A8 | Gate: firing rate per dataset; calibration of tie vs non-tie rows | table + per-branch reliability | gate slides |
| A9 | Temperature: â vs true running accuracy over the stream; estimator error; oracle-T (NLL with labels, reported as a ceiling) | curves + table | temperature slides |
| A10 | NLL, Brier, AURC for every headline run | table | backup |

Prerequisite (lead, 2026-11-08): the CPU replay (`scripts/replay.py`, `utils/replay_runner.py`)
does not yet write the T04 trace. Wire `--trace` through the replay so every replayed
cal_tda/tda run can emit `<dataset>_<method>.trace.pt`, with a test that the replayed trace
equals a direct-run trace on synthetic features.
