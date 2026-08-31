# Report

`main.tex` plus `sections/*.tex` and `refs.bib`. Two of the tables and all of the figures are
generated, not hand-written:

- `tables/ablation.tex` &mdash; `analysis/make_ablation_table.py`
- `../results/summary_table.tex` &mdash; `analysis/aggregate.py`
- `figures/*.png` &mdash; `analysis/make_figures.py`, copied here from `analysis/figures/`
  (copied, never symlinked; kept in sync by `scripts/run_all.ps1` / `.sh`)

Editing those files directly is pointless: re-running the analysis overwrites them.

## Building `main.pdf`

```powershell
.\scripts\build_report.ps1
```

Requires MiKTeX (`winget install MiKTeX.MiKTeX`) or TeX Live. The script runs
pdflatex &rarr; bibtex &rarr; pdflatex &rarr; pdflatex, which is the minimum that resolves
citations and cross-references &mdash; a single pass leaves `??` in place of every Section,
Table and Figure number &mdash; and then fails loudly if the log contains an error, an
undefined reference, or an undefined citation.

`-JobName <name>` writes to `<name>.pdf` instead, so an older build can be kept alongside a
new one for comparison.

Two things about the setup are load-bearing and easy to break:

- The bibliography style must stay one of natbib's own (`plainnat`). The text uses `\citet`,
  which plain BibTeX's `plain` renders as "Author undefined" without failing the build.
- `\usepackage[numbers]{natbib}` is what numbers the reference list and shows the number at
  each citation site. The `.bst` does not control that.

## Checking it without building

```powershell
python report/verify_report.py
```

A structural check that runs in a second and needs no TeX distribution: every
`\input`, `\includegraphics` and `\cite` target resolves to a real file or key; braces and
`\begin`/`\end` pairs balance; and every `\command` used is provided by base LaTeX, by a
loaded package, or by the document itself. That last check is the useful one &mdash; it
catches a macro used without its package (`\argmin`, `\mathbb`, `\citet`) before a build
does, and it is cheap enough to run on every edit.
