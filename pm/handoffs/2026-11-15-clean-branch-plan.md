# Plan: a clean final branch, then a full reproduction check (for macwave)

Written 2026-11-15 by the lead session, at Idan's request. Executor: macwave.
Reference state: branch `deck/refinement` at the last commit Idan pushed after this plan (at writing: `3d387e1`,
"shorter final paper and deck v6"). Call it `REF` below (`git rev-parse origin/deck/refinement`).

## Goal

1. A new branch, `final`, whose history tells the project in order: one commit (or a few) per task, from the
   upstream TDA import to the final paper and deck, each with a message that says what the task did and found.
2. Its last commit has the same tree as `REF`, except files that are agreed to be dead (step 3).
3. At the end, every number, table and figure of the paper and the deck is regenerated from the raw replays and
   compared with the committed ones. The check passes only if they match.

## Ground rules

- Do not rewrite or force-push any existing branch (`main`, `deck/refinement`, `publish`, ...). `final` is new.
- Commit by naming files (`git add <paths>`), never `git add -A`. Keep the Colab leftovers out (`colab_run.json`,
  `datasets_verify_*.json`, `download_datasets.log`, `pytest_gpu.txt`) and `upstream/`.
- Generated files are never edited by hand (`report/generated/*`, `presentation/slides.md`, figures): rerun their
  scripts (pm/RULES.md, "no hand-typed number").
- Push `final` only after Idan's OK (Idan pushes, as before).

## Step 0. Set up and record the baseline

1. Fresh clone, `git checkout REF`.
2. Environments: the `caltta` conda env (Python, torch CPU, pandas, matplotlib, python-pptx, scipy) and the `tex`
   env with Tectonic (`conda create -n tex -c conda-forge tectonic`; PyMuPDF there is handy to render pages).
3. `python -m pytest -q` on `REF`. At writing: 403 passed, 17 skipped. Record the exact count.
4. Data that is not in git (needed only for step 6): `features/` (about 645 MB) and `replay_results/` (about
   3.7 GB, subfolders `raw/`, `t14/`, `t16_fresh/`, `t16_baselines/`, `t17_shift/`). Copy them from Idan's
   machine. Regenerating them is possible but long: features with the Colab notebook (`docs/COLAB_EXTRACTION.md`),
   replays with `scripts/replay*.sh`.

## Step 1. Inventory the tasks in time order

Sources: `pm/tasks/T01..T14`, the dated entries of `pm/STATUS.md` (T15-T17 live there), `pm/handoffs/`,
`pm/DECISIONS.md`, `pm/KNOWLEDGE.md`, and `git log --reverse --name-status`.

Write a table (task, dates, files it introduced, what it found, in the final paper or not). The order in the
history, as a starting point to verify:

| order | task | what |
|---|---|---|
| 0 | import | upstream TDA code, v1 report and deck (gated fusion, margin admission, LOO temperature) |
| 1 | pm | project-management folder (plan, rules, knowledge, tasks) |
| 2 | T03, T04 | O(N) leave-one-out temperature search; label-free per-sample trace |
| 3 | T01, T02 | CLIP feature extraction and cache format; exact CPU replay of the stream |
| 4 | T13 | Colab extraction notebook and dataset downloader |
| 5 | T10 | draft deck v2 (later replaced) |
| 6 | T05, T07 | seed statistics; protocols P1 (per-dataset alpha, beta) vs P2 (single setting) |
| 7 | A11, T14 | echo analysis on val; development sets; full-vector trace; the agreement gate |
| 8 | T15 | fresh held-out sets; story figures; read-first TDA, DMN-weak, ECALP baselines |
| 9 | T16 | comparison table (ZAEC, temperature), ablation by agreement, no-information analysis, report and deck |
| 10 | T17 | shift data, near/far tiers, over/underconfidence, random control, final paper and deck v6 |

## Step 2. Decide what `final` keeps (ask Idan before deleting anything)

Keep the dependency closure of the final products: `report/main.tex` and everything it inputs, the regeneration
scripts of step 6, `presentation/build_pptx.py`, the method code (`online_method/`, `online_tta.py`, `utils/`,
`data/`), extraction and replay scripts, the tests, `docs/`, and `pm/` (the research record, including negative
results in KNOWLEDGE and DECISIONS).

Candidates to drop, each only if `git grep` finds no reference from a kept file, and each listed with a one-line
reason in the commit that drops it:
- the v2 report pipeline leftovers: `report/tables/`, older `report/figures/*.png` written by `scripts/run_all.*`
  (`report/README.md` already says `main.tex` does not use them);
- `analysis/switch.py` and `results/analysis/t16/switch*.csv` (the confidence switch was taken out of the paper);
- code for abandoned paths (the SUN397 registration, sets that are in no tier), and stale deck drafts.

If Idan prefers, keep everything and make `final` differ from `REF` only in history. Then `git diff REF final`
must be empty.

## Step 3. Build the branch

1. Start `final` at the first commit (`3c79fc0`, the upstream import and v1), so the TDA import stays commit one:
   `git switch -c final 3c79fc0`.
2. For each task in the step 1 order: bring in the FINAL version of the files that task introduced
   (`git checkout REF -- <paths>`), together with their tests and docs. Commit as `T0x: <what it did>` and put in
   the body what it found, with a pointer to `pm/tasks/T0x.md` or the STATUS entry. A file that several tasks
   changed goes in with the first task that created it, in its final form; say in the body that later tasks
   refined it.
3. After each commit, at least import the touched modules, and run the tests whose files are present
   (`python -m pytest -q tests/<those files>`). Failures caused only by a later task's missing file are fine
   until that task's commit.
4. Last commits: the T16-T17 analyses, `results/analysis/**` CSVs, `report/` (tex, generated, figures, pdf),
   `presentation/` (builder, pptx, slides.md), and `pm/STATUS.md`.
5. Check `git diff --stat REF final`: empty, or exactly the agreed drops.

## Step 4. Top-level README on `final`

What the project is (one paragraph from the abstract), how to reproduce (environments, data, the step-6 order),
where the paper and deck are, and the task-to-commit map.

## Step 5. Tests

`python -m pytest -q` on `final`: same counts as step 0, minus the tests of dropped files.

## Step 6. Reproduction check: regenerate everything the paper and deck show

Run in this order from the repo root of `final`, with `features/` and `replay_results/` in place. Each script
rewrites tracked files; nothing should change.

| # | command | produces | used for |
|---|---|---|---|
| 1 | `python analysis/protocols.py` | `results/protocols.csv`, `results/analysis/protocols_paired.csv` (3 seeds) | App. A |
| 2 | `python analysis/protocols.py --seeds 0 1 2 3 4 5 6 7 8 9 --out-csv results/protocols_10seeds.csv --paired-csv results/analysis/protocols_paired_10seeds.csv` | 10-seed tables | Table 1 |
| 3 | `python analysis/t14_dev.py` | `results/analysis/t14/dev_protocols.csv` | near sets in Table 3 |
| 4 | `python analysis/t14_results.py` | `results/analysis/t14/alpha_curve_test.csv`, `matched_test.csv` | alpha-sweep control |
| 5 | `python analysis/story.py` | `results/analysis/story/summary_{val,test}.csv` | section 3, Fig. 2 |
| 6 | `python analysis/comparison.py` | `results/analysis/t16/comparison*.csv` | Table 2, App. B |
| 7 | `python analysis/fresh.py` | `results/analysis/t16/fresh*.csv`, `gaps.csv`, `where_it_helps*.csv` | Table 3, Fig. 3, section 6 |
| 8 | `python analysis/no_info.py` | `results/analysis/t16/no_info_*.csv`, Figs. 4-5 | Table 4, section 6, App. C |
| 9 | `python analysis/p1_appendix.py` | `results/analysis/p1_appendix_*.csv` | App. A |
| 10 | `python analysis/paper_figures.py` | Figs. 2-3 | paper |
| 11 | `python analysis/report_assets.py` | `report/generated/numbers.tex`, `tab_*.tex` | every number in the paper |
| 12 | `scripts/build_report.ps1` (PowerShell) | `report/main.pdf` | paper |
| 13 | `python presentation/build_pptx.py` | `presentation/calibration_aware_tta.pptx`, `slides.md` | deck |

Step 2's arguments were inferred from the script's options; confirm them against the committed
`results/protocols_10seeds.csv` (same columns, `n` = 10).

Pass criteria:
- `report/generated/numbers.tex` and every `report/generated/tab_*.tex`: byte-identical (`git diff --exit-code`).
- Every regenerated CSV: identical, or equal within float noise. Check with
  `pandas.testing.assert_frame_equal(new, old, rtol=1e-6, atol=1e-3)` for any CSV that `git diff` flags.
- The paper's text: `pdftotext -layout report/main.pdf` before and after, `diff` empty.
- `presentation/slides.md` (all numbers in the deck notes): identical.
- Figures: PNG bytes can differ between matplotlib versions. If they do, open old and new side by side; the data
  behind them is already covered by the CSV check.
- `python -m pytest -q`: all pass.

Optional deeper check, if time allows: rerun one replay from the features, for example DTD seed 0, with
`scripts/replay.py` (see `docs/` and `pm/handoffs/2026-11-10-T07-protocols.md` for the arguments), and compare its
record with the one in `replay_results/raw/`. This shows the replays themselves reproduce, not only the analysis.

Write the outcome to `pm/handoffs/<date>-final-branch-verification.md`: one row per artifact (identical / within
tolerance / differs, with the diff). Any "differs" is a finding to report to Idan before pushing.

## Step 7. Hand back

Tell Idan: the branch name and its last commit, the verification report, the list of dropped files (if any), and
the test counts. Idan reviews and pushes.
