# Status

Updated 2026-11-07.

## Now
- Branch `deck/refinement` (based on `plan/reservoir-cache-temperature` = `publish` @ 4def98a
  + plan docs). PM folder created; plan approved.
- Phase A (infrastructure) starting: T01, T02, T03, T04 in parallel, CPU-only, synthetic tests.
- T09 related-work audit: first pass done in `research/related-work.md`; to be completed.

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
