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
- [ ] When T01's script is merged: send macwave the run command and ask him to upload the
      feature files (instructions in `tasks/T01-feature-extraction.md`).
- [ ] Push `deck/refinement` from the local clone (Claude cannot push to macwave12).
- [ ] Decide after T07: headline protocol P1 or P2 (rule in `DECISIONS.md`).

## Next milestones
1. Phase A merged with tests green.
2. Features received; replay reproduces `results/summary.csv` (T02 acceptance).
3. Evidence figures (T05, T06, T07) → review → deck rebuild (T10).
