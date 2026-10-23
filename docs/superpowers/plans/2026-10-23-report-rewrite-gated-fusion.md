# Report Rewrite: Gated Fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `report/main.pdf` around the final finding (gated fusion wins; margin dropped; temperature does not stack), with every number regenerated from committed records.

**Architecture:** The headline sweep is redefined as five methods and re-run; `aggregate.py` and `make_figures.py` learn the new method set so Table 3, Figure 1 and Figure 2 regenerate; a new `make_contribution_tables.py` emits the four per-contribution LaTeX tables so Sections 5.1–5.3 carry no hand-typed numbers; then the LaTeX sources are rewritten to the agreed structure and the PDF is built.

**Tech Stack:** Python 3.14 venv (`.venv`), pandas, matplotlib, pytest; PowerShell scripts; MiKTeX `pdflatex` via `scripts/build_report.ps1`.

**Spec:** `docs/superpowers/specs/2026-10-23-report-rewrite-gated-fusion-design.md`

## Global Constraints

- Method keys and display order, verbatim from the spec: `clipzs`, `tda`, `cal_tda_gate_temp`, `cal_tda_gate`, `cal_tda_temp`.
- Gate thresholds: `fusion_gate_max_gap: 0.2`, `fusion_gate_min_mass: 0.6`, `fusion_weight: 0.5`; margin admission off in every headline `cal_tda_*` config.
- Every run: `-j 0`, seed 0, `ViT-B/16`, batch size 1, `$env:DATA_ROOT = "D:\second_degree\deep\data_root"`, venv activated.
- No hand-typed numbers in the report: per-dataset tables come from `analysis/make_contribution_tables.py`; the headline table from `aggregate.py`; Table 4 from `make_ablation_table.py --study gate`.
- `python report/verify_report.py` and `.\scripts\build_report.ps1` both pass before the PDF is committed.
- Never `git add` `presentation/~$calibration_aware_tta.pptx`.
- Run every Python command through `.\.venv\Scripts\python.exe` (or the activated venv).

## Review Focus

1. A stale `results/raw/<dataset>_cal_tda.json` left behind would resurrect the bundle as a sixth headline row in `summary.csv` and the figures. Pinned by `test_no_bundle_records_in_headline_dir` (Task 2).
2. A `cal_tda_*` record missing for one dataset must make `plot_reliability` warn, not silently draw four curves. Pinned by `test_missing_headline_record_warns` (Task 3).
3. The new method names carry two underscores; an unescaped one breaks the LaTeX build. Pinned by `test_summary_table_escapes_new_method_names` (Task 2) and by `make_contribution_tables` tests asserting `\_` (Task 4).
4. A `\ref{sec:discussion}` or `\ref{tab:temp-per-dataset}` surviving the rewrite renders as `??`. Pinned by running `report/verify_report.py` (Task 5, Step 7), which checks every `\ref` has a `\label`.
5. Figures 1 and 2 must list the same five series in the same order. Pinned structurally: both read `METHOD_STYLE`; `test_legend_lists_every_headline_method` (Task 3) asserts the legend is built from that dict.

---

### Task 1: Commit the in-flight work in two logical commits

**Files:**
- Modify: `.gitignore` (add `presentation/~$*.pptx` under the existing presentation block)
- Stage, commit 1 (deck): `presentation/build_pptx.py`, `presentation/calibration_aware_tta.pptx`, `presentation/slides.md`, `requirements.txt`, `.gitignore`
- Stage, commit 2 (gated fusion + second ablation): `online_method/cal_tda.py`, `tests/test_cal_tda_ops.py`, `configs/method/exploration/`, `configs/method/ablation_gate/`, `results/raw/exploration/`, `results/raw/ablation_gate/`, `results/ablation_gate.csv`, `report/tables/ablation_gate.tex`, `scripts/run_ablation_gate.ps1`, `analysis/make_ablation_table.py`, `analysis/make_figures.py`, `tests/test_make_figures.py`, `analysis/figures/ablation_ece.png`, `analysis/figures/ablation_gate_ece.png`

**Interfaces:**
- Produces: a clean working tree on branch `publish` from which Tasks 2–6 start.

- [ ] **Step 1: Add the lock-file ignore**

Append `presentation/~$*.pptx` after the `presentation/assets/` line in `.gitignore`.

- [ ] **Step 2: Verify the whole suite passes on the current tree**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all 105 collected tests pass.

- [ ] **Step 3: Commit the deck work**

```bash
git add .gitignore requirements.txt presentation/build_pptx.py presentation/calibration_aware_tta.pptx presentation/slides.md
git commit -m "feat(deck): data-driven 19-slide builder with TDA-by-example slide"
```

- [ ] **Step 4: Commit the gated-fusion method work and second ablation**

```bash
git add online_method/cal_tda.py tests/test_cal_tda_ops.py configs/method/exploration configs/method/ablation_gate results/raw/exploration results/raw/ablation_gate results/ablation_gate.csv report/tables/ablation_gate.tex scripts/run_ablation_gate.ps1 analysis/make_ablation_table.py analysis/make_figures.py tests/test_make_figures.py analysis/figures/ablation_ece.png analysis/figures/ablation_gate_ece.png
git commit -m "feat(cal_tda): gated probabilistic fusion, exploration runs and second ablation"
```

- [ ] **Step 5: Confirm the tree is clean**

Run: `git status --short`
Expected: no output.

---

### Task 2: Redefine the headline sweep as five methods and re-run it

**Files:**
- Create: `configs/method/cal_tda_gate.yaml`, `configs/method/cal_tda_temp.yaml`, `configs/method/cal_tda_gate_temp.yaml` (byte-identical to `configs/method/ablation_gate/{gate,temp,gate_temp}.yaml` apart from the header comment, which names them as headline configs)
- Modify: `scripts/run_all.ps1`, `scripts/run_all.sh` (method lists), `analysis/aggregate.py:20-28` (`METHOD_ORDER`, `METHOD_CELLS`)
- Delete: `results/raw/dtd_cal_tda.json`, `results/raw/flower102_cal_tda.json`, `results/raw/pets_cal_tda.json`, `results/raw/eurosat_cal_tda.json`, `results/raw/aircraft_cal_tda.json`
- Test: `tests/test_aggregate.py`, `tests/test_configs.py`

**Interfaces:**
- Produces: `results/summary.csv` with 25 rows, `method` ∈ {`clipzs`, `tda`, `cal_tda_gate_temp`, `cal_tda_gate`, `cal_tda_temp`}; `results/summary_table.tex` (`\label{tab:main}`); `results/raw/<dataset>_<method>.json` for the 15 new runs. `analysis.aggregate.METHOD_ORDER` is the display order Tasks 3 and 5 rely on.

- [ ] **Step 1: Write the failing tests**

In `tests/test_aggregate.py`, replace the method list in `test_rows_are_sorted_by_dataset_then_method_order` and add two tests:

```python
def test_rows_are_sorted_by_dataset_then_method_order(tmp_path):
    ...
    for method in ["cal_tda_temp", "cal_tda_gate", "clipzs", "cal_tda_gate_temp", "tda"]:
        ...
    assert list(frame["method"])[:5] == [
        "clipzs", "tda", "cal_tda_gate_temp", "cal_tda_gate", "cal_tda_temp"
    ]


def test_summary_table_escapes_new_method_names():
    from analysis.aggregate import METHOD_CELLS
    assert r"cal\_tda\_gate" in METHOD_CELLS
    assert r"cal\_tda\_gate\_temp" in METHOD_CELLS
    assert r"cal\_tda\_temp" in METHOD_CELLS


def test_no_bundle_records_in_headline_dir():
    # The three-contribution bundle is no longer a headline method; a leftover
    # record would silently add a sixth row to summary.csv and both figures.
    assert not list(Path("results/raw").glob("*_cal_tda.json"))
```

In `tests/test_configs.py`, extend `METHOD_CONFIGS` (the list that parametrises `test_method_configs_construct_a_valid_config_object`) with `cal_tda_gate`, `cal_tda_temp`, `cal_tda_gate_temp`, and add:

```python
@pytest.mark.parametrize("name,gate,temp", [
    ("cal_tda_gate", True, False), ("cal_tda_temp", False, True), ("cal_tda_gate_temp", True, True),
])
def test_headline_configs_have_margin_off_and_the_named_contributions_on(name, gate, temp):
    config = CalTdaConfig(**yaml.safe_load(Path(f"configs/method/{name}.yaml").read_text()))
    assert config.use_margin_admission is False
    assert config.use_prob_fusion is gate and config.fusion_gate is gate
    assert config.use_loo_temperature is temp
    assert (config.fusion_gate_max_gap, config.fusion_gate_min_mass) == (0.2, 0.6)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_aggregate.py tests/test_configs.py -q`
Expected: the four new/changed tests FAIL (missing configs, old `METHOD_ORDER`, five `*_cal_tda.json` present).

- [ ] **Step 3: Create the three headline configs and update `aggregate.py`**

`METHOD_ORDER = ["clipzs", "tda", "cal_tda_gate_temp", "cal_tda_gate", "cal_tda_temp"]`; `METHOD_CELLS` holds the five escaped names. Update the comment above `METHOD_ORDER` ("baseline, prior work, ours").

- [ ] **Step 4: Delete the bundle's headline records and update the sweep scripts**

`git rm results/raw/*_cal_tda.json`. In `run_all.ps1` the `$methods` array becomes `clipzs` (no config), `tda` (`configs/method/tda.yaml`), and the three `cal_tda_*` entries, each run as `-Method cal_tda -Config configs/method/<name>.yaml -RunName <name>`. Mirror in `run_all.sh`. Update both scripts' header comment ("5 datasets x 5 methods").

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_aggregate.py tests/test_configs.py -q`
Expected: PASS.

- [ ] **Step 6: Run the 15 new headline runs**

`clipzs` and `tda` records are unchanged; only the `cal_tda_*` runs are needed. Run in the background, one dataset at a time (the harness has killed long sweeps under memory pressure; resume with the remaining pairs if that happens):

```powershell
$env:DATA_ROOT = "D:\second_degree\deep\data_root"; .\.venv\Scripts\Activate.ps1
foreach ($d in @("dtd","pets","flower102","aircraft","eurosat")) {
  foreach ($m in @("cal_tda_gate","cal_tda_temp","cal_tda_gate_temp")) {
    .\scripts\run_one.ps1 -Dataset $d -Method cal_tda -Config "configs/method/$m.yaml" -RunName $m
  }
}
```

Expected: 15 new files `results/raw/<dataset>_<method>.json`; roughly 60 minutes total.

- [ ] **Step 7: Aggregate and cross-check against the earlier records**

Run: `.\.venv\Scripts\python.exe analysis/aggregate.py`
Expected: 25 rows printed; for each dataset the `cal_tda_gate` row equals `results/ablation_gate.csv`'s `gate` row, `cal_tda_temp` equals its `temp` row and `cal_tda_gate_temp` its `gate_temp` row, to two decimals (the spec's reference table). A mismatch larger than 0.13 points means a config differs; stop and investigate.

- [ ] **Step 8: Commit**

```bash
git add configs/method/cal_tda_gate.yaml configs/method/cal_tda_temp.yaml configs/method/cal_tda_gate_temp.yaml scripts/run_all.ps1 scripts/run_all.sh analysis/aggregate.py tests/test_aggregate.py tests/test_configs.py results/raw results/summary.csv results/summary_table.tex
git commit -m "feat(sweep): five headline methods (gate, temp, gate+temp replace the bundle)"
```

---

### Task 3: Figures 1 and 2 over the five headline methods

**Files:**
- Modify: `analysis/make_figures.py:19-40` (`METHOD_STYLE`, drop `TEMP_VARIANT_STYLE`), `:45-120` (`plot_accuracy_vs_ece`), `:123-197` (`plot_reliability`), `:200-223` (`plot_reliability_legend`), `make_all_figures`/`main` (drop `ablation_raw_dir` plumbing; keep `ablation_csv` and `ablation_gate_csv` for the bar charts)
- Modify: `scripts/run_all.ps1`, `scripts/run_all.sh`, `scripts/run_ablation.ps1`, `scripts/run_ablation_gate.ps1` only if they pass `--ablation-raw-dir` (they do not today; verify)
- Test: `tests/test_make_figures.py`

**Interfaces:**
- Consumes: `results/summary.csv` method keys from Task 2.
- Produces: `METHOD_STYLE: dict[str, dict]` with keys in `METHOD_ORDER` order and labels verbatim from the spec table; `plot_accuracy_vs_ece(summary, out_path)`; `plot_reliability(raw_dir, dataset, out_path)`; `plot_reliability_legend(out_path)`; PNGs `accuracy_vs_ece.png`, `reliability_<dataset>.png`, `reliability_legend.png` in `analysis/figures/` and copied to `report/figures/`.

- [ ] **Step 1: Write the failing tests**

Update the `fake_results` fixture to write the five methods (`clipzs`, `tda`, `cal_tda_gate_temp`, `cal_tda_gate`, `cal_tda_temp`) and add:

```python
def test_legend_lists_every_headline_method():
    from analysis.make_figures import METHOD_STYLE
    from analysis.aggregate import METHOD_ORDER
    assert list(METHOD_STYLE) == METHOD_ORDER


def test_missing_headline_record_warns(fake_results):
    summary, raw, figures = fake_results
    (raw / "dtd_cal_tda_gate.json").unlink()
    with pytest.warns(UserWarning, match="cal_tda_gate"):
        make_all_figures(summary, raw, figures)


def test_make_all_figures_no_longer_takes_an_ablation_raw_dir(fake_results):
    import inspect
    assert "ablation_raw_dir" not in inspect.signature(make_all_figures).parameters
```

Keep `test_reruns_are_idempotent`'s count at 4 PNGs for the fixture (scatter + legend + two reliability panels).

- [ ] **Step 2: Run them to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_make_figures.py -q`
Expected: the three new tests FAIL.

- [ ] **Step 3: Rewrite `METHOD_STYLE` and the three plot functions**

Styles (my call; keep the four existing hues, add one): `clipzs` o `#4C72B0`; `tda` s `#DD8452`; `cal_tda_gate_temp` ^ `#CCB974`; `cal_tda_gate` D `#8172B2`; `cal_tda_temp` * `#55A868`. Labels verbatim from the spec. `plot_accuracy_vs_ece` loses its `temp_variant` parameter and joins each dataset's five points sorted by accuracy, as today. `plot_reliability` reads only `raw_dir/<dataset>_<method>.json` for the five methods, warning per missing record with the method key in the message. Update the module docstring and the `METHOD_STYLE` comment block; remove the `TEMP_VARIANT_STYLE` block and its justification comment.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_make_figures.py -q`
Expected: PASS.

- [ ] **Step 5: Regenerate and copy the figures, then look at them**

Run: `.\.venv\Scripts\python.exe analysis/make_figures.py; Copy-Item analysis\figures\*.png report\figures\ -Force`
Then open `analysis/figures/accuracy_vs_ece.png`, `reliability_pets.png` and `reliability_legend.png` with the Read tool. Expected: five series per dataset in the scatter, five curves per reliability panel, legend lists five methods plus the identity line, no label clipped at the right spine.

- [ ] **Step 6: Commit**

```bash
git add analysis/make_figures.py tests/test_make_figures.py analysis/figures report/figures
git commit -m "feat(figures): headline scatter and reliability grid over the five final methods"
```

---

### Task 4: Generated per-contribution tables for Sections 5.1–5.3

**Files:**
- Create: `analysis/make_contribution_tables.py`
- Create: `tests/test_contribution_tables.py`
- Modify: `scripts/run_ablation.ps1`, `scripts/run_ablation_gate.ps1` (call the new script after their table step), `report/README.md` (list the four generated tables)

**Interfaces:**
- Consumes: `results/ablation.csv` (columns `dataset,variant,accuracy,ece`), `results/ablation_gate.csv` (same), `results/raw/ablation/<dataset>_margin.json` (`hyperparams.admission.admission_rate`), `analysis.latex_table.wrap_table`, `analysis.aggregate.DATASET_ORDER` and `DATASET_LABEL`.
- Produces: `build_contribution_tables(ablation_csv, ablation_gate_csv, ablation_raw_dir, out_dir) -> dict[str, Path]` writing four files into `report/tables/`:
  - `margin_vs_none.tex`, label `tab:margin`, columns: Dataset | none acc | margin acc | none ECE | margin ECE | Admission rate (%)
  - `fusion_vs_none.tex`, label `tab:fusion`, columns: Dataset | none acc | fusion acc | none ECE | fusion ECE
  - `gate_vs_none.tex`, label `tab:gate`, same shape, from `ablation_gate.csv` variants `none`/`gate`
  - `temp_vs_none.tex`, label `tab:temp`, same shape, from `ablation_gate.csv` variants `none`/`temp`
  Rows in `DATASET_ORDER`, display names from `DATASET_LABEL`, a final `Mean` row, values `%.2f`, `escape=True`.

- [ ] **Step 1: Write the failing tests**

```python
def test_writes_four_tables_with_expected_labels(fake_study):
    paths = build_contribution_tables(*fake_study, out_dir)
    assert set(paths) == {"margin", "fusion", "gate", "temp"}
    assert r"\label{tab:margin}" in paths["margin"].read_text()


def test_margin_table_carries_admission_rate_as_percent(fake_study):
    # fixture writes dtd_margin.json with admission_rate 0.53 -> "53.00"
    text = build_contribution_tables(*fake_study, out_dir)["margin"].read_text()
    assert "53.00" in text and "DTD" in text


def test_tables_end_with_a_mean_row(fake_study):
    text = build_contribution_tables(*fake_study, out_dir)["gate"].read_text()
    assert "Mean" in text.split(r"\bottomrule")[0].splitlines()[-1]


def test_variant_names_are_escaped_for_latex(fake_study):
    text = build_contribution_tables(*fake_study, out_dir)["temp"].read_text()
    assert "_" not in text.replace(r"\_", "")
```

`fake_study` writes two-dataset CSVs (`dtd`, `pets`) for both studies and a `dtd_margin.json`/`pets_margin.json` with admission rates 0.53 / 0.82.

- [ ] **Step 2: Run them to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contribution_tables.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `analysis/make_contribution_tables.py`**

Follow `make_ablation_table.py`'s import pattern (`try: from latex_table ...`). `main()` takes `--ablation-csv`, `--ablation-gate-csv`, `--ablation-raw-dir`, `--out-dir` with the repo defaults. Captions name the comparison and that `none` is the TDA-equivalent baseline run in the same sweep.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contribution_tables.py -q`
Expected: PASS.

- [ ] **Step 5: Generate the real tables and wire the scripts**

Run: `.\.venv\Scripts\python.exe analysis/make_contribution_tables.py`
Expected: four files under `report/tables/`; `margin_vs_none.tex` shows Aircraft 15.90 and Pets 81.90 in the admission column. Add `python analysis/make_contribution_tables.py` to both ablation scripts after their table step, and list the four files in `report/README.md`'s generated-tables bullets.

- [ ] **Step 6: Commit**

```bash
git add analysis/make_contribution_tables.py tests/test_contribution_tables.py report/tables scripts/run_ablation.ps1 scripts/run_ablation_gate.ps1 report/README.md
git commit -m "feat(report): generated per-contribution tables for the results section"
```

---

### Task 5: Rewrite the report sources

**Files:**
- Modify: `report/main.tex` (abstract; remove `\input{sections/discussion}`)
- Modify: `report/sections/introduction.tex` (final paragraph only)
- Modify: `report/sections/experiments.tex` (keep 4.1; delete 4.2; add the sweeps paragraph)
- Rewrite: `report/sections/results.tex`
- Rewrite: `report/sections/conclusion.tex`
- Delete: `report/sections/discussion.tex`

**Interfaces:**
- Consumes: `\input{../results/summary_table.tex}` (Task 2), `figures/accuracy_vs_ece.png`, `figures/reliability_*.png`, `figures/reliability_legend.png` (Task 3), `\input{tables/margin_vs_none.tex}` etc. (Task 4), `\input{tables/ablation_gate.tex}` (`tab:ablation-gate`, already generated).
- Produces: labels `sec:results`, `sec:margin`, `sec:fusion`, `sec:temperature`, `sec:together`, `sec:limitations`; `fig:scatter`, `fig:reliability` unchanged.

- [ ] **Step 1: Abstract and introduction**

Rewrite the abstract to the spec's story with the five-dataset means (CLIP 53.16/5.57, TDA 56.63/11.24, gate 55.01/4.61, temp 56.60/8.54, gate+temp 55.01/6.59). Rewrite the Introduction's final paragraph ("Each change is independently switchable…") to say: margin admission fails and is dropped; fusion fails as proposed but succeeds once gated to two-way ties; temperature helps alone but does not stack; the recommended method is TDA's cache plus gated fusion. Keep the three-item contribution list as proposed.

- [ ] **Step 2: Leave Section 3 untouched**

`method.tex` is not edited. The gate is not mentioned anywhere before Section 5.2: the report presents the three ideas as first proposed, shows un-gated fusion failing, and only then introduces the gate.

- [ ] **Step 3: Experiments**

Delete subsection 4.2 and its `\label{sec:admission-rates}`. Append to 4.1 one paragraph listing the sweeps: headline 5 datasets × 5 methods (25 runs), first ablation 8 variants × 5 (40 runs, `scripts/run_ablation.ps1`), second ablation 4 variants × 5 (20 runs, `scripts/run_ablation_gate.ps1`), and that the second ablation's `none`/`temp` records are the first ablation's runs (same code path, seed and effective config), as `run_ablation_gate.ps1`'s header states.

- [ ] **Step 4: Results, in the spec's order**

Write `results.tex` with these subsections and content; cite only generated tables and figures. Do not carry over the old opening sentence ("Table~\ref{tab:main} is generated directly by \texttt{analysis/aggregate.py} from \texttt{results/summary.csv}; no number in it is hand-typed") or write any sentence like it; table provenance stays in `report/README.md`:
- 5.1 `sec:margin` — `\input{tables/margin_vs_none.tex}`; the admission-rate paragraph moved from old 4.2 (threshold = 0.01 cosine gap; rates track zero-shot accuracy); the three failure mechanisms from the spec; closing sentence: every variant from here on has margin admission off.
- 5.2 `sec:fusion` — `\input{tables/fusion_vs_none.tex}`; the flat-cache ceiling mechanism and the Aircraft accident; then a paragraph "Gating the fusion" that *defines* the gate here, as an equation: tie condition $p_{(1)} - p_{(2)} < 0.2 \land p_{(1)} + p_{(2)} > 0.6$ on $p = \sigma(z_{\text{clip}})$, Eq. (fusion) with $w = 0.5$ on a tie, $\sigma(z_{\text{clip}})$ otherwise, thresholds fixed a priori in probability space; config flag names in `\texttt{}` with `\allowbreak` after each `\_` as `method.tex`'s comment instructs; `\input{tables/gate_vs_none.tex}`; the same-quantity-different-moment explanation; where it costs accuracy (EuroSAT −5.92, DTD −1.35 vs `none`) and where it loses calibration (Pets).
- 5.3 `sec:temperature` — `\input{tables/temp_vs_none.tex}`; accuracy identical by construction; Pets mechanism (no stopping rule, LOO underestimates on fine-grained data).
- 5.4 `sec:together` — `\input{../results/summary_table.tex}` introduced by a sentence that says what it compares, not how it was made; Figure 1 (`fig:scatter`, caption listing the five series, drop the "purple diamond" wording); `\input{tables/ablation_gate.tex}`; why gate+temp do not stack with the pinning fractions (EuroSAT 99.9%, Aircraft 92.5%, Pets 0%); Figure 2 (`fig:reliability`, keep the existing subfigure block, update the caption to the five methods); the strict-criterion count (gate: Aircraft only) stated plainly.
- 5.5 `sec:limitations` — one seed, one backbone; thresholds a priori and untested elsewhere; EuroSAT accuracy cost; LOO estimate quality on fine-grained data; the `none`-vs-`tda` floating-point drift sentence from the old text (keep it, shortened).

- [ ] **Step 5: Conclusion and main.tex**

Rewrite the conclusion (half a page): what was proposed, what failed and why, what is recommended, the honest criterion count. In `main.tex` remove `\input{sections/discussion}`; `git rm report/sections/discussion.tex`.

- [ ] **Step 6: Grep for dangling references**

Run: `grep -n "sec:discussion\|sec:admission-rates\|tab:temp-per-dataset\|tab:ablation}" report/main.tex report/sections/*.tex`
Expected: no output (`tab:ablation}` must not appear; the second study is `tab:ablation-gate`).

- [ ] **Step 7: Structural verification**

Run: `.\.venv\Scripts\python.exe report/verify_report.py`
Expected: exits 0, no undefined reference, input, graphic, citation or command.

- [ ] **Step 8: Build the PDF**

Run: `.\scripts\build_report.ps1`
Expected: `OK  report/main.pdf  (N pages)`. Open `report/main.pdf` pages with the Read tool and check: Table 4 is the four-row second ablation, Figure 1 shows five series, Figure 2's legend matches Figure 1, no `??`.

- [ ] **Step 9: Commit**

```bash
git add report/main.tex report/sections report/main.pdf
git commit -m "docs(report): rewrite results around gated fusion; drop margin and the discussion section"
```

---

### Task 6: Whole-tree verification

**Files:** none new.

- [ ] **Step 1: Full test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 2: Regenerate everything from records and confirm nothing drifts**

Run: `.\.venv\Scripts\python.exe analysis/aggregate.py; .\.venv\Scripts\python.exe analysis/make_ablation_table.py --study gate; .\.venv\Scripts\python.exe analysis/make_contribution_tables.py; .\.venv\Scripts\python.exe analysis/make_figures.py; Copy-Item analysis\figures\*.png report\figures\ -Force; git status --short`
Expected: `git status` shows only PNG files, if anything (matplotlib embeds no timestamp by default, so usually nothing). Any changed `.csv` or `.tex` means a generator and its committed output disagree; fix before finishing.

- [ ] **Step 3: Report what is stale**

State in the handoff that `README.md` and the deck still describe the old finding and are covered by a follow-up plan.
