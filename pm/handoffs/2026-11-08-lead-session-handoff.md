# 2026-11-08 — Lead session handoff (cloud session → local Claude Code)

Read with `pm/README.md`, `pm/STATUS.md`, `pm/PLAN.md`, `pm/RULES.md`, `pm/KNOWLEDGE.md`,
`pm/DECISIONS.md`. This note holds what those files do not.

## State at handoff
- `deck/refinement` (GitHub, pushed by Idan): T01–T04 and T13 merged, suite 285 passed /
  14 skipped (CUDA- and data-only). T09 related work: second pass.
- Colab extraction done (T4), features validated by replay (K33–K35).
- **T10 draft deck v2**: being built by an agent in the cloud session. When it lands it is
  merged into `deck/refinement` before the switch; if this note is read before that, check
  `git log` for a `merge: T10 draft deck` commit.

## Next steps, in order (updated 2026-11-10)
1. **Local environment** (CPU, ~10 min), PowerShell, from the repo root:
   ```powershell
   conda create -n caltta python=3.12 -y; conda activate caltta
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
   Get-Content requirements.txt | Select-String -NotMatch '^(torch|torchvision)\b|index-url' | Set-Content $env:TEMP\req.txt
   pip install -r $env:TEMP\req.txt "setuptools<81" info-nce-pytorch pytest
   python -m pytest -q
   ```
2. **Features**: unzip `..\features_2026-11-08.zip` (in the `university` folder) at the repo
   root so `features/<dataset>/` exists (git-ignored). Already validated (KNOWLEDGE K33–K35):
   T01/T02 done. Re-run `python scripts/replay.py --suite headline` once as a smoke test
   (~1–2 min) and `python analysis/compare_replay.py`.
3. **Idan's deck review** of draft v2 (`deck_v2_draft.pdf`; open questions in
   `handoffs/2026-11-08-T10-deck-v2-draft.md`). Then a T11 reviewer agent on the draft.
4. **Seed statistics tool** (small): an analysis script that, for a set of replayed seeds,
   writes per-method mean/std/MAE and paired-by-seed differences (mean, MAE, wins out of n)
   to `results/analysis/`. Policy in DECISIONS (2026-11-10): 3 seeds now, 10 (+5 ablations)
   at wrap-up; seeds ≥3 need CPU-generated permutations (only 0–2 were extracted).
5. **Evidence phase** after the deck review: T05 (first wire the T04 trace through the replay),
   T06, T07 (P1/P2 × 3 seeds), T08 baselines. Then T10 fill-in → T11 → final 10-seed run → T12.

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
