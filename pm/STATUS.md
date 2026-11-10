# Status

Updated 2026-11-10.

## Now
- Local env `caltta` (conda, CPU torch 2.11.0) set up on Idan's Windows machine: 301 passed /
  14 skipped / 1 failed (`test_real_spec_matches_documented_contract`: `read_text()` without
  `encoding="utf-8"` under the cp1255 locale; passes with `PYTHONUTF8=1`). Features unzipped;
  headline replay smoke test (179 s) + `compare_replay.py`: 18/25 within tolerance, same misses
  as K34/K35 (EuroSAT gate/TDA, temperature ECE). Locale test fixed: suite 310 passed / 14 skipped.
- Seed statistics tool `analysis/seed_stats.py` done (review): headline seeds 0/1/2 replayed,
  CSVs in `results/analysis/seed_stats_headline_*.csv`. Gate vs TDA (mean of 5): acc −1.82
  (0/3), ECE −6.60 (3/3). See `handoffs/2026-11-10-local-setup-seed-stats.md` (MAE definition
  needs Idan's OK).
- `deck/refinement`: T01/T02 (extraction, CPU replay, protocol switch) and T03/T04 (O(N)
  temperature, trace) merged; suite 250 passed / 14 skipped (CUDA- and data-only tests).
- T13 Colab notebook merged; Colab run in progress (Idan). T10 draft deck v2 merged (36 + 6 backup, placeholders for evidence): awaiting Idan's review and a T11 reviewer pass. Open questions in `handoffs/2026-11-08-T10-deck-v2-draft.md`.
- T09 related-work audit: second pass done.

- Features received (Colab T4) and placed in `features/` (git-ignored). T02 real-data
  acceptance done: see KNOWLEDGE K33–K35. Evidence phase (T05–T08) can start after Idan's
  deck review.

## Blockers
- Real features needed for every number in Phases C–F → **GPU job on Colab (T13 notebook), run by Idan**.

## Reminders for Idan
- [ ] When T01–T04 and T13 are merged and `deck/refinement` is pushed: open the Colab
      notebook (`docs/COLAB_EXTRACTION.md`), choose a T4 GPU, Run all (~30–40 min). The
      features zip lands in Google Drive `cal_tta_features/`; the lead pulls it from there.
- [ ] Push `deck/refinement` from the local clone (Claude cannot push to macwave12).
- [ ] Decide after T07: headline protocol P1 or P2 (rule in `DECISIONS.md`).

## Next milestones
1. Phase A merged with tests green.
2. Features received; replay reproduces `results/summary.csv` (T02 acceptance).
3. Evidence figures (T05, T06, T07) → review → deck rebuild (T10).
