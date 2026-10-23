# Report rewrite: gated fusion — design

Date: 2026-10-23. Decided in conversation with the project owner; this document records
those decisions so the plan can be executed without the conversation.

## Intent

Rewrite `report/main.pdf` so it tells the project's final story: of the three proposed
contributions, margin admission failed and is dropped; probabilistic fusion failed as
proposed but succeeds once *gated* to two-way ties; LOO temperature scaling helps alone but
does not stack on the gate. The recommended method is **TDA's cache + gated probabilistic
fusion**, margin admission off, temperature off.

## What stays, what changes

- Sections 1 (Introduction), 2 (Related Work), 3 (Method) stay as written, with one
  exception decided here: the Introduction's final paragraph and the abstract are
  rewritten because they state the old finding. Section 3 deliberately does **not**
  mention the gate: the report shows the three ideas as first proposed, and the gate is
  introduced in Section 5.2 only after the un-gated fusion has been shown to fail.
- Section 4 (Experiments) keeps 4.1 "Datasets and protocol". Its 4.2 "Cache admission
  rates" moves into the margin-admission results subsection, since margin is dropped after
  that point. 4.1 gains one paragraph listing the three sweeps (headline, first ablation,
  second ablation) with run counts.
- Section 5 (Results) is restructured, in this order:
  1. **Margin admission**: `none` vs `margin` per dataset (accuracy, ECE, admission
     rate); why it failed; "from here on every variant has margin admission off".
  2. **Probabilistic fusion**: `none` vs `fusion` per dataset; why it failed; the gated
     variant; `none` vs `gate` per dataset; where the gate costs accuracy.
  3. **LOO temperature**: `none` vs `temp` per dataset; where and why it fails (Pets).
  4. **Putting it together**: headline table (five methods), Figure 1 (scatter), Table 4
     (second ablation: none / gate / temp / gate_temp), Figure 2 (reliability grid).
  5. **Limitations** (short; replaces the Discussion section).
- Section "Discussion" is removed. Its mechanism explanations move into 5.1–5.3; its
  caveats become 5.5. Owner asked for an opinion: a separate Discussion would now repeat
  5.1–5.3, so dropping it is right, but the limitations content must survive somewhere.
- Conclusion is rewritten.

## The five headline methods

Names are the `method` values in `results/summary.csv`, the record filenames under
`results/raw/`, and the keys of `METHOD_STYLE` in `analysis/make_figures.py`, in this
display order everywhere:

| method key | display label | config |
|---|---|---|
| `clipzs` | CLIP zero-shot | — |
| `tda` | TDA (CVPR'24) | `configs/method/tda.yaml` |
| `cal_tda_gate_temp` | Ours: gated fusion + LOO temperature | `configs/method/cal_tda_gate_temp.yaml` |
| `cal_tda_gate` | Ours: gated fusion (recommended) | `configs/method/cal_tda_gate.yaml` |
| `cal_tda_temp` | Ours: LOO temperature | `configs/method/cal_tda_temp.yaml` |

The three-contribution bundle (`cal_tda`, `configs/method/cal_tda.yaml`) is no longer a
headline method. Its five records under `results/raw/*_cal_tda.json` are deleted; the
identical numbers survive as the first ablation's `all` rows. The config file stays (the
first ablation's `all.yaml` and `tests/test_configs.py` depend on it).

Figure 1 (accuracy vs ECE) and Figure 2 (reliability diagrams + shared legend) show exactly
these five series, with one legend derived from `METHOD_STYLE` so the two cannot differ.
The headline runs are re-run under these names (not copied from exploration records) so
`scripts/run_all.ps1` reproduces `summary.csv` from scratch.

## The gated variant (defined in Section 5.2, where it is introduced)

With $p = \sigma(z_{\text{clip}})$ and $p_{(1)} \ge p_{(2)}$ its two largest entries, a
sample is a *two-way tie* when $p_{(1)} - p_{(2)} < 0.2$ and $p_{(1)} + p_{(2)} > 0.6$.
On a tie the output is Eq. (fusion) with $w = 0.5$; otherwise it is $\sigma(z_{\text{clip}})$
(CLIP corrected by TDA's negative cache, positive cache not consulted). Thresholds are in
probability space so one global pair means the same thing on 10 and 100 classes; both
values were fixed a priori, not tuned. Implemented as `fusion_gate`,
`fusion_gate_max_gap`, `fusion_gate_min_mass` on `CalTdaConfig`
(`online_method/cal_tda.py`), functions `is_two_way_tie` and `fusion_probs`.

## Mechanisms the text must state (from the conversation)

**Why margin admission failed** (5.1):
1. A raw-logit threshold of 1.0 is not portable: admission ran 15.9% (Aircraft) to 81.9%
   (Pets), and damage tracks admission rate monotonically.
2. A starving cache is a *biased* cache: only classes CLIP finds easy clear the threshold,
   so the cache's neighbours can only ever vote for easy classes. That is a prior pulling
   every prediction toward them — accuracy drops, ECE rises (Aircraft 15.56 → 32.64).
3. It filters on the wrong signal: a confidently wrong sample has a large margin too.
   Margin and entropy both measure loudness, not correctness; TDA's per-class eviction by
   entropy already removes near-ties where traffic allows.

**Why fusion failed as proposed** (5.2): cache logits are sums of a few exponential
similarity weights, topping out near 2; CLIP's logits are scaled by ≈100 and peaked.
Softmaxed over 100 classes the cache is nearly flat, so a 50/50 mixture halves CLIP's
confidence on every sample, right or wrong. ECE punishes underconfidence as hard as
overconfidence: Pets at 88.77% accuracy reported ≈45% confidence (ECE 43.78). Aircraft's
apparent gain is the same ceiling landing near that dataset's low accuracy by accident.

**Why gating fixes it** (5.2): on a clear winner the cache cannot change the argmax and can
only sharpen (TDA) or dilute (fusion), so the right move is to leave it out; on a two-way
tie the cache can flip the answer, which is where TDA's accuracy gain lives, and a
confidence near one half is honest there. The gate uses the same top-two-gap quantity
margin admission used, applied at prediction time (when to consult the cache) instead of
storage time (what the cache remembers), so every class stays represented.

**Why temperature fails on Pets** (5.3): no stopping rule — it always smooths toward the
LOO estimate. On fine-grained datasets the LOO 1-NN agreement underestimates accuracy
(a cached Birman's nearest neighbour is often a Ragdoll even when both pseudo-labels are
right), so it smooths an already-honest model.

**Why gate + temperature do not stack** (5.4): both target the overconfidence TDA's cache
manufactures; the gate removes it at the source, leaving CLIP's own ≈5% ECE, which the
label-free estimate cannot improve on. On EuroSAT/Aircraft the search pins at T = 1
(99.9% / 92.5%, no-op); on Pets/Flowers it acts in the wrong direction (Pets 5.68 → 13.90).

## Reference numbers (all from committed records; generated tables are authoritative)

Headline, mean over five datasets: CLIP 53.16 / 5.57; TDA 56.63 / 11.24;
gate 55.01 / 4.61; temp 56.60 / 8.54; gate+temp 55.01 / 6.59 (accuracy % / ECE %).

Second ablation (`results/ablation_gate.csv`), per dataset:

| dataset | none acc | none ECE | gate acc | gate ECE | temp acc | temp ECE | gate_temp acc | gate_temp ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dtd | 47.04 | 17.41 | 45.69 | 6.24 | 47.04 | 4.68 | 45.69 | 6.21 |
| flower102 | 69.79 | 6.51 | 69.02 | 4.08 | 69.79 | 4.85 | 69.02 | 5.89 |
| pets | 88.74 | 2.48 | 88.77 | 5.68 | 88.74 | 15.54 | 88.77 | 13.90 |
| eurosat | 53.22 | 14.43 | 47.30 | 3.56 | 53.22 | 13.43 | 47.30 | 3.56 |
| aircraft | 24.21 | 15.56 | 24.27 | 3.48 | 24.21 | 4.21 | 24.27 | 3.37 |

First ablation rows used in 5.1/5.2 (`results/ablation.csv`): `margin` and `fusion`
per dataset; admission rates from `results/raw/ablation/<dataset>_margin.json`
(`hyperparams.admission.admission_rate`): aircraft 0.159, eurosat 0.265, dtd 0.530,
flower102 0.657, pets 0.819.

Strict success criterion (accuracy ≥ TDA and ECE < TDA, headline rows): the gate meets it
on Aircraft only; report this count honestly alongside the per-dataset deltas.

## Non-goals

- README.md and the slide deck are not updated by this work (they will be stale; a
  follow-up plan covers them).
- The `fusion_sharpen_cache` flag and the `results/raw/exploration/` records stay as a
  recorded side experiment; the report does not discuss them.
- No new GPU experiments beyond re-running the 15 headline `cal_tda_*` runs.

## Constraints

- Every number in the report comes from a generated table or a committed record; new
  per-dataset tables are generated by a script, never hand-typed. The report text does
  not say so: sentences of the form "Table N is generated directly by
  `analysis/aggregate.py` from `results/summary.csv`; no number in it is hand-typed" are
  removed, not carried over. That provenance belongs in `report/README.md` and the
  scripts, not in the paper.
- Windows runs pass `-j 0`; `DATA_ROOT` must be set; seed 0, ViT-B/16, batch size 1.
- `python report/verify_report.py` and `.\scripts\build_report.ps1` must both pass;
  `report/main.pdf` is committed.
- Commit after each task; never commit `presentation/~$*.pptx`.
