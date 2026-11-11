# Status

Updated 2026-11-11 (overnight lead session). **Nothing committed since the three `git am`
patches**: the whole night's work is in the working tree, ready to be split into commits.

## Now
- **T14 (review): new method, the agreement-gated cache.** The positive cache is used only
  where its argmax disagrees with CLIP's. It has no parameters. Full write-up:
  `research/T14-agreement-gate.md`; facts in KNOWLEDGE K38–K43.
  - **Test, P2, 10 seeds:** 56.66 / 5.51 vs TDA 56.75 / 10.96. Δ −0.09 [−0.13, −0.04] accuracy,
    −5.45 [−5.54, −5.37] ECE.
  - **Dev sets (ImageNetV2, Caltech101, UCF101, Food101-10k):** mean Δ −0.01 / −4.07. 2 clear
    wins, 1 tie, and 1 failure (Caltech101: underconfident CLIP).
  - **Independent reviewer pass done;** 15 findings, all addressed (§10 of the write-up).
- **T07 (review):** the headline is P2 (Idan confirmed). P1 is in the appendix:
  - `results/analysis/p1_appendix_reproduction.csv`: the GPU run is within 0.16 points of the
    benchmark's Table 14;
  - `results/analysis/p1_appendix_p1_vs_p2.csv`.
- **Seed policy:** seeds 3–9 generated (`scripts/make_permutations.py`); 10-seed P2 tables for
  the headline methods in `results/protocols_10seeds.csv`, with 90% CIs.
- **Seed statistics tool** (review). A11 echo analysis is merged (P1, val). Research backlog:
  `research/BACKLOG.md`.
- **T10:** draft deck v2 is with Idan. It has to be rebuilt to the new storyline (PLAN).
- **T09** related-work audit: second pass done.

## Blockers
- None.

## Reminders for Idan
- [ ] Read `research/T14-agreement-gate.md`. Confirm or reject the design change: method
      frozen after the val proof of concept, dev sets used as evaluation rather than for
      fitting (DECISIONS 2026-11-11).
- [ ] Confirm the two disclosed T14 criterion amendments (DECISIONS 2026-11-10).
- [ ] Confirm the per-method MAE definition (mean absolute deviation from the seed mean).
- [ ] Commit (about 6 commits), then push `deck/refinement`.
- [ ] Deck v2 review comments, if still open.

## Next milestones
1. Commit and push; a second reviewer pass on the corrected write-up if wanted.
2. Deck rebuild (T10) to the storyline: diagnosis → agreement gate → fair evaluation (P2, 10
   seeds, matched comparison, dev sets) → limitations; P1 in the appendix. Then T11 reviewers.
3. Optional research before the deck: a label-free detector of underconfidence on agreeing
   samples (Pets, Caltech101); label-free calibration of the gated branch.
4. T12 report.
