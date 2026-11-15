# Status

Updated 2026-11-11 (T16 lead session, evening). Handoff: `handoffs/2026-11-11-T16-final-assets-handoff.md`.

## Now
- **Method frozen:** agreement gate, always on, P2 headline (KNOWLEDGE K41, K43, K46, K49–K52).
- **T16 analysis done (tested):** comparison table (`analysis/comparison.py`, ZAEC a tested function), fresh sets and
  "where it helps" (`analysis/fresh.py`), gated cache ablation (`analysis/story.py`, `gpos`), report and deck assets
  (`analysis/report_assets.py`, `analysis/paper_figures.py`). Full suite: 376 passed, 17 skipped (before the
  reviewer fixes; the touched test files pass after them).
- **Report v3** (`report/main.tex`, ~4 pages + appendix) builds with Tectonic (`scripts/build_report.ps1`).
  **Deck v3** (`presentation/build_pptx.py`): 20 main + 6 backup slides (opening: roadmap, task, CLIP, two ways to adapt, TDA); TDA section uses the TDA paper's Fig. 1 and 2.
- **Independent reviewer pass done:** 13 findings, all addressed (Lemma 2 vs the gate's condition stated with data, K50;
  the underconfidence story toned down to a tendency with exceptions; ablation wording; `fresh.py` robust to partial runs;
  one ΔECE source).
- **Finding that changes the framing (K51–K52):** on the fresh tier the gate keeps accuracy; mean ECE 6.27 → 5.49, 5/10 wins
  (large on ImageNet-A/-Sketch); it loses where CLIP is underconfident (CIFAR-10/100, ImageNet-R, STL-10).

- **2026-11-12 overnight (Idan: keep going until a simple win):** searched uses of agreement with 3 research agents + a
  flip analysis; all negative (K55) except a **confidence switch** (K54): keep the echo while the running mean CLIP
  confidence > 0.78 (now 0.77, see below), ECE below the gate on every tier. Reviewed (sound with caveats); tested `analysis/switch.py`; in the
  report as an exploratory paragraph + appendix E and in the deck as backup slide B6 (+ B7: negative results). Headline unchanged. Also K53:
  `comparison.py --generalization` (ZAEC on dev/fresh). Not committed.
- **2026-11-12 (with Idan):** switch threshold re-chosen on val + dev (both kinds of stream; test and fresh held out):
  0.77, test 5.10, fresh 5.48; appendix E gains the design table and the fresh-preferred range (K54, DECISIONS).
  α from corr(c, z) tried and negative (K55 f). Report (8 pages) and deck v4 rebuilt (20 main + 7 backup; conclusion carries the final claim, B7 lists the negative results). Not committed.
- **2026-11-13 (Idan: final flow):** new tested `analysis/no_info.py` (K56): ablation by agreement group on every tier
  (agreeing positive cache: no accuracy, only confidence) and flip information (CLIP's and the cache's scores, alone or
  combined, carry almost no information about which flips are right). Report: Table 3 (ablation), section "Is there
  information beyond binary agreement?" + Fig. 2, rates figure in appendix E; abstract, contributions, conclusion
  updated (9 pages). Deck v5: 20 main + 9 backup (ablation slide in the diagnosis section, "beyond one bit" slide in
  results; val agreement and reliability slides moved to backup; B8 rates). Committed 5a25365.
- **2026-11-13 (Idan's paper notes, part 1):** report restructured after two reader-flow reviews: intro (CLIP, shift,
  TTA, accuracy + ECE equation, online zero-shot TTA, TDA with its Fig. 2), Background (TDA math with c(x) as a vector,
  calibration cost, ZAEC with its equation), Motivation (thesis, self-vote, agreement lemma; old Lemma 1 moved to the
  appendix), Method (why argmax c vs CLIP's label), Results, "When does the gate help" (where it helps, beyond one bit,
  limitations). Abstract in changes vs CLIP. Switch and label-free proxies moved to appendix E. Main text 7 pages.
  Deck not yet aligned. Not committed.
- **2026-11-14 (Idan's paper notes, part 2; T17):** new **shift tier**: 9 sets far from CLIP's web photos (FER2013,
  PCam, SVHN, MNIST, CLEVR-count, Country211, Rendered SST2, DMLab, Retinopathy; CLIP_benchmark WDS mirror, 3000-image
  test subsets, seed 0; `scripts/fetch_wds.py`, `scripts/replay_shift.sh` -> `replay_results/t17_shift`). There TDA adds
  +2.77 acc and +12.16 ECE over CLIP; the gate keeps the accuracy (+2.93) and lowers ECE on 9/9 (34.52 -> 21.21, CLIP
  22.37). ImageNet and ImageNetV2 excluded from every tier (`EXCLUDED`: TDA's P2 was searched on ImageNet val); P2 = the
  TDA paper's single setting, P1 = the code release's per-dataset values. Gate mean ECE now below TDA's on every held-out
  tier. Fig. 5 replaced by `no_info_gain.png` (best val-chosen threshold rule per score vs the oracle: every rule loses
  accuracy, oracle +1.1 to +2.3); rates figure x = score value. Full suite: 395 passed, 17 skipped. Report rebuilt (7
  main pages). Deck not aligned (still uses no_info_auc.png). Not committed.
- **2026-11-14 (Idan): tiers are now design / near / far**, fixed by what the images are (`utils/feature_cache.py`
  NEAR, FAR, `held_tier`): near = Caltech101, Food101, VOC2007 (ordinary photos of everyday things; STL-10 dropped as
  ImageNet-derived); far = the 9 T17 sets + GTSRB (fine-grained labels). The other 8 held-out sets are in Appendix D only.
  Near: TDA -0.30 acc / +0.08 ECE vs CLIP, gate -0.37 / +0.45. Far: TDA +2.80 / +10.65, gate +2.96 / -1.24 (9/10 ECE
  wins). On near most TDA flips are wrong (right share 40%). Tests (excluding the deck): 386 passed, 17 skipped;
  `tests/test_build_pptx.py` fails until the deck uses the new macros. Not committed.
- **2026-11-14 (Idan), tiers revised once after first results (stated in the report):** near = Caltech101, Food101,
  Cars (VOC2007 out: multi-object scenes); far = GTSRB, ImageNet-R, ImageNet-Sketch, FER2013, MNIST, PCam, Rendered
  SST2, SVHN (Country211 out: label is not about the depicted object; CLEVR-count, DMLab, Retinopathy out: CLIP at or
  below chance). Near: TDA +0.60 / -0.04 vs CLIP, gate +0.52 / +0.15. Far: TDA +3.83 / +1.85, gate +3.96 / -1.64 (6/8
  ECE wins vs TDA; losses ImageNet-R, GTSRB). SUN397 registered (fetch_wds, class names) but abandoned: its shards are
  class-sorted, so a partial fetch gives 37 of 397 classes; a full fetch is 41 shards. Tests (no deck): 387 passed.
- **2026-11-14 (Idan): over- vs underconfidence.** `fresh.record_gaps` -> `results/analysis/t16/gaps.csv` (signed
  confidence - accuracy per set; CLIP seed 0, TDA and gate mean over seeds); `report_assets.gap_numbers` macros and a
  conf.-acc. block in Table 3. TDA raises CLIP's gap on 14/16 sets (overconfident on 13); the gate lowers TDA's on 16/16.
  Near: CLIP -2.8, TDA +2.0, gate -2.9 (same ECE, opposite sides). Where the gate loses ECE it stays underconfident.
  Report: intro sentence, 5.3 paragraph, limitation (i); main text 8 pages. Tests (no deck): 389 passed. Committed.
- **2026-11-15 (Idan): final makeover of paper and deck.** Report rewritten shorter (9 pages: about 6 main + 2
  appendix; was 13): only the near and far sets are reported (no tier-revision history, no other held-out sets),
  section 6 is "Discussion", the confidence switch, earlier-attempts table and proofs appendix are gone. New: a random
  control in the ablation (`no_info.random_mask`, variant `rand`: the cache on as many random samples as the gate opens
  on; design test 55.85 / 7.16 vs gate 56.72 / 5.51), flip figure with near and far only and no shading
  (`no_info_break_even.png`, all ten scores in `no_info_scores.png`), Table 1 bold = best per column, the alpha-sweep
  sentence re-checked and reworded (TDA needs alpha ~1.8 to match the gate's accuracy, ECE 10.4). Deck v6: 17 main + 3
  backup slides (was 20 + 9), aligned with the report; deck tests pass again. Full suite: 403 passed, 17 skipped.
  Reviewer cycles on both. Committed 3d387e1.
- **2026-11-15, second review round (paper and deck), applied:** section 3 and Fig. 2b now use the actual read-first
  TDA runs (`results/analysis/t16/read_first_val_test.csv`: val ECE 11.19 -> 8.30, acc 56.29 -> 56.03; the old 7.52
  was the self-vote subtraction approximation); threshold rules reported per held-out tier (near +0.00, far -0.03);
  App. A means over datasets (CLIP ECE 5.65); protocol paragraph names which analyses use logged runs; ZAEC authors.
  Deck: bold marks only the best values, ablation numbers labelled with their 3 orders, symbols g/b no longer reused,
  titles no stronger than the evidence. Full suite: 405 passed, 17 skipped. Plan for the clean final branch (macwave):
  `pm/handoffs/2026-11-15-clean-branch-plan.md`.

## Running (background, this machine)
- ECALP: done (58.34 / 32.32; K49).
- Fresh replays: done (all 10 sets, 3 seeds; ImageNet-Sketch −7.42 ECE). Report and deck regenerated; ready to commit.

## Reminders for Idan
- [ ] Decide the confidence switch's role (DECISIONS 2026-11-12): exploratory extension (current), or promote it.
- [ ] Framing decision (DECISIONS 2026-11-11, pending Idan): the fresh-set result (gate worse where CLIP is underconfident),
      reported as is, with CLIP's calibration gap as the label-based explanation.
- [ ] Commit (lead asks first), then push `deck/refinement`.
- [ ] Read the report PDF and the deck.

## Next milestones
1. Commit (after Idan's OK), push.
2. Optional research: control the echo's size instead of removing it (a temperature that can also sharpen), or a
   label-free underconfidence detector.
