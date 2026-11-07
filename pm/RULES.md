# Standing rules

Every task, agent and commit follows these. Changing a rule needs a `DECISIONS.md` entry.

## Scientific integrity
1. **No selection on test.** Anything that chooses among variants, thresholds, capacities
   or hyperparameters is decided on the CoOp **val** split. Test is evaluated for final
   numbers and for sensitivity plots that are labelled as such.
2. **Labels only in the scorer.** Method code never sees a ground-truth label;
   `tests/test_no_label_leak.py` must keep passing. Diagnostics that need labels (purity,
   estimator error, correctness of flips) are computed in analysis code from saved traces.
3. **Disclose history.** Earlier choices made on test (Phase 3 exploration, per-dataset
   α/β) are stated in the deck and report, not hidden.
4. **Both protocols.** P1 (published) and P2 (clean) are always reported side by side.
5. **Seeds.** Final tables use seeds 0/1/2 (mean ± std) once replay makes it cheap.
6. **Say what failed.** Negative results stay, with their mechanism and evidence.

## Engineering
7. **Equivalence first.** Any change to method code that should not change results
   (feature replay, CPU device handling, O(N) temperature) ships with a test proving
   equality against the old path; the replay must reproduce `results/summary.csv`
   (accuracy within 0.2 points, ECE within 0.3 points) before new numbers are used.
8. **Defaults preserve behaviour.** New flags default to off; existing configs reproduce
   their recorded results.
9. **Numbers come from CSVs.** Slides and report pull every number from tracked CSVs via
   code. No hand-typed numbers.
10. **CPU only.** No job may assume CUDA. Long jobs are split, cached, and resumable.
    The only GPU job is T01, run by macwave.
11. **Tests.** Every new module has unit tests runnable on CPU in < 2 minutes total.
12. **Small commits** with conventional messages; one task per branch commit series.
    Commit messages end with the session's Co-Authored-By / Claude-Session lines.

## Process
13. Every task ends with a handoff note and a review by someone who did not do it.
14. The lead critiques work in progress and may re-scope a task; re-scoping is logged.
15. Idan is kept informed at each milestone; decisions that change scope go to him first.
16. Reminders that depend on people (e.g. the GPU job) are listed in `STATUS.md`.
