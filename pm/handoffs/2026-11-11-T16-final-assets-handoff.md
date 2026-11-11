# 2026-11-11 — T16 handoff: comparison, fresh sets, report v3, deck v3

Read with `pm/STATUS.md`, `pm/DECISIONS.md` (last 4 rows), `pm/KNOWLEDGE.md` K49–K52. This note holds only what those
files do not.

## Done in this session (uncommitted until Idan says so)
- `analysis/comparison.py` (+ `tests/test_comparison.py`): 9-row comparison, TDA the paired reference; `zaec()` tested.
- `analysis/fresh.py` (+ `tests/test_fresh.py`), `scripts/replay_fresh.sh`: fresh-set table, record-based ΔECE on every
  tier, Spearman ρ against label-free signals and CLIP's (label-based) calibration gap. Datasets with incomplete replays are
  left out with a warning, so it can be rerun at any time.
- `analysis/story.py`: `gpos` variant (gated positive cache alone), one agreement definition (`disagree_mask`'s
  tie-break), `agree_neg_moved` / `agree_tda_flips` (Lemma 2 vs the gate's condition), refactored where-it-helps plot.
- `analysis/paper_figures.py`, `analysis/report_assets.py` (+ tests): every report number is a macro in
  `report/generated/numbers.tex`; tables in `report/generated/tab_*.tex`.
- `report/main.tex` rewritten diagnosis-first (~4 pages + appendix: proofs, P1, earlier attempts, per-dataset comparison,
  selection history). `report/sections/`, `main_v2.pdf`, `verify_report.py` removed; `scripts/build_report.ps1` uses
  Tectonic (conda env `tex`, created this session; it also has PyMuPDF for rendering PDFs to PNG).
- `presentation/build_pptx.py` rewritten as deck v3 (19 + 4); `tests/test_build_pptx.py` updated (≤ 20 main slides, TDA
  figures cited, split labels, numbers from data). TDA paper figures in `presentation/assets/tda_paper/` (from the arXiv
  source of 2403.18293).
- Superseded scratch CSVs removed (`zaec_preliminary`, `negative_cache_ablation_gated`, `imagenet_val_records`).

## Still running (check `ps` / the logs before starting duplicates)
- ECALP: `replay_results/t16_baselines/` (O(n²); EuroSAT seeds 1–2 and Aircraft left at 18:15).
- Fresh: `replay_results/t16_fresh.log` (ImageNet-R gate, then ImageNet-Sketch: 50,889 × 1000 classes, hours; no full
  trace, signals from 500-sample trace runs).
Then run the regeneration chain in STATUS. Re-read the abstract's fresh-tier sentence and the deck's generalization and
where-it-helps slides: the text is phrased to stay true, but check it against the final numbers.

## Gotchas
- Shell heredocs and `sed` mangle LaTeX backslashes (`\r`, `\t` become control characters); edit `.tex` and LaTeX strings
  in Python with the Edit tool or a script file.
- matplotlib mathtext (deck equations) has no `\underbrace` or `\mathbb`.
- PowerPoint COM export renders the deck for visual checks (`$pres.Export(dir, "PNG", 1280, 720)`).
- The stray empty files a shell hook creates (`0$`, `1000`, fragments with `>`) must be deleted, never staged.
