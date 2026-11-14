# Report

`main.tex` and `refs.bib`: the course report (about 6 pages of main text, then references and a short appendix).
Nothing in it is hand-typed data:

- `generated/numbers.tex` (one `\newcommand` per number in the prose) and `generated/tab_*.tex` (all tables):
  `analysis/report_assets.py`, from tracked CSVs only (its docstring lists them).
- `figures/paper_*.png`: `analysis/paper_figures.py`; `figures/no_info_*.png`: `analysis/no_info.py` (both copied
  here from `analysis/figures/`). Diagnosis figures are from the val splits, results figures from the test splits
  and the held-out near and far tiers, and every caption says which.

Editing the generated files is pointless: re-running the scripts overwrites them.

## Building `main.pdf`

```powershell
python analysis/no_info.py; python analysis/paper_figures.py; python analysis/report_assets.py
.\scripts\build_report.ps1
```

The build uses Tectonic (`conda create -n tex -c conda-forge tectonic`) and fails on LaTeX errors and on
undefined references or citations.

`tables/` and the older `figures/*.png` are still written by the v2 pipeline (`scripts/run_all.*`);
`main.tex` no longer uses them.
