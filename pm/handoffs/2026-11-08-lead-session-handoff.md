# 2026-11-08 — Lead session handoff (cloud session → local Claude Code)

Read with `pm/README.md`, `pm/STATUS.md`, `pm/PLAN.md`, `pm/RULES.md`, `pm/KNOWLEDGE.md`,
`pm/DECISIONS.md`. This note holds what those files do not.

## State at handoff
- `deck/refinement` (GitHub, pushed by Idan): T01–T04 and T13 merged, suite 285 passed /
  14 skipped (CUDA- and data-only). T09 related work: second pass.
- **Colab extraction running** (Idan, 2026-11-08 ~23:15). Output:
  `MyDrive/cal_tta_features/features_<date>.zip`. Not yet verified on real hardware.
- **T10 draft deck v2**: being built by an agent in the cloud session. When it lands it is
  merged into `deck/refinement` before the switch; if this note is read before that, check
  `git log` for a `merge: T10 draft deck` commit.

## Next steps, in order
1. **Local environment (CPU, ~10 min)**, PowerShell, from the repo root:
   ```powershell
   conda create -n caltta python=3.12 -y; conda activate caltta
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
   Get-Content requirements.txt | Select-String -NotMatch '^(torch|torchvision)\b|index-url' | Set-Content $env:TEMP\req.txt
   pip install -r $env:TEMP\req.txt "setuptools<81" info-nce-pytorch pytest
   python -m pytest -q   # expect all pass, CUDA/data tests skipped
   ```
   (`setuptools<81` and `info-nce-pytorch` were needed for test collection in the cloud
   workspace; pkg_resources is imported by a dependency.)
2. **Get the features**: download the zip from Drive, unzip at the repo root so that
   `features/<dataset>/...` exists (git-ignored). Check `features/*/meta.json`: torch
   version, dtype, GPU name, splits present, `stream order matches recorded run` (will be
   `n/a` on Colab — the original `save_results.pt` files are only on macwave's machine).
3. **T02 real-data acceptance (blocking)**: `python scripts/replay.py --suite headline`,
   `--suite ablation`, `--suite ablation_gate`, then `python analysis/compare_replay.py`.
   Tolerance 0.2 accuracy / 0.3 ECE points per dataset (RULES §7). Colab T4 numerics differ
   from macwave's RTX 5060 runs in the last fp16 bits, so small drift is expected.
   - If clipzs matches but tda does not → suspect stream order first (torch version, RNG).
   - If both drift → features/numerics; try `--autocast off` as the A/B.
   - Record the outcome in `KNOWLEDGE.md` and `DECISIONS.md`. Decision already taken:
     final numbers are **re-baselined on the replay** (one pipeline for every method); the old
     GPU numbers become a sanity check.
4. **T05 prerequisite**: wire the T04 trace through the replay (see T05 task file).
5. **Evidence phase** (T05, T06, T07, T08) — only after Idan reviews the draft deck
   (he asked to approve before the heavy phase). Then T10 fill-in → T11 review → T12 report.

## How to work (Idan's standing preferences)
- Quota matters: avoid parallel agents unless clearly worth it; keep updates short.
- **Reviews are done by separate reviewer agents**, not by the lead reading everything
  (keeps the lead's context small). The lead decides on findings.
- Critique the work as you go and propose changes; Idan expects a researcher's judgement,
  not task execution only. Log scope changes in `DECISIONS.md`.
- Keep `STATUS.md` current; every task ends with a handoff note here.

## Lessons from this session
- Agent worktrees were created at an old base (the repo's first commit). First step in any
  agent brief: `git merge --ff-only deck/refinement` (or rebase).
- Agents share CPU cores; wall-time measurements under contention are unreliable.
- `online_tta.py` is now `build_parser()` + `main(argv)`; tests drive `main` directly.
- The deck is generated: edit `presentation/build_pptx.py`, never the .pptx.
- Rendering for review: `soffice --headless --convert-to pdf`, `pdftoppm -r 50 -png`.

## Open questions for Idan
- Headline protocol after T07 (rule in `DECISIONS.md`).
- Whether the report (T12) is updated before or after the presentation.
- Post-wrap-up research order: self-vote removal (E1), reservoir cache + better `â` (plan doc),
  soft-label tie-breaking cache (`research/cache-idea-formalization.md`).
