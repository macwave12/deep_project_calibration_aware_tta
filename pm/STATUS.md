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
  results; val agreement and reliability slides moved to backup; B8 rates). Not committed.

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
