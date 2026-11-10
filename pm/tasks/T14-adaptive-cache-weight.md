# T14 — Dataset-independent adaptive cache weight α(x)

Status: open · Owner: local session (lead) + agents · Depends on: echo analysis (A11,
`analysis/echo_analysis.py`, patch `echo-analysis.patch`) · Approved by Idan 2026-11-10

## Why
A11 (val, 3 seeds; `analysis/figures/echo_val.png`) shows: TDA's cache fixes decisions only
when CLIP is unsure (CLIP confidence ≲ 0.6; EuroSAT up to +28% of a bin), while its push on
CLIP's top-2 gap and the self-vote share grow with CLIP confidence, and TDA inflates
confidence in every bin — including bins with zero flips. The current gate (two-way ties)
misses EuroSAT's gains, which sit in *diffuse* low-confidence samples. Goal: a per-sample
weight `softmax(z + α·w(φ(x))·c − n)` that keeps the useful flips and removes the useless
inflation, **without depending on the dataset**.

## Key fact that makes it cheap
Cache contents depend only on raw CLIP (KNOWLEDGE K7), never on α. So with z, c, n stored per
sample (full vectors), any α-rule is evaluated exactly offline, without re-running the stream.

## Guardrails for dataset independence (Idan's requirement)
1. Signals must be scale-free: probabilities not logits; entropy / log C; confidence relative
   to 1/C; divergences / log C; cosines not β-dependent kernels. Cache opinion as a real
   distribution: kernel-normalized class weights `p_cache(k) = Σ_{i∈k} w_i / Σ_i w_i`
   (not softmax(c), which is near-flat).
2. Fit under the **clean protocol** (α = 2, β = 5 everywhere). The rule outputs a multiplier on
   one global α — never a per-dataset value.
3. Tiny, theory-shaped rule: ≤ 2 inputs, ≤ 3 parameters, monotone where theory says so.
   Motivating argument: adding logits treats CLIP and cache as independent experts; the cache
   is built from CLIP's own labels, so when they agree it double-counts (ideal α → 0); when they
   disagree and CLIP is unsure it carries new information (α high).
4. Transfer must be shown: leave-one-dataset-out (LODO), parameter stability across folds,
   overlay of per-dataset curves, class-subset invariance.
5. Final rule fitted on **development datasets only**; the five project datasets are the test
   set, evaluated once (RULES §1).

## Steps
1. **Trace extension (small)**: store full-length `c` and `n` per step (opt-in, default off;
   equivalence tests as in T04). Wire `--trace` through the CPU replay (T05 prerequisite).
2. **Signals** (label-free), per sample: CLIP confidence, normalized entropy, probability
   margin; cache confidence, evidence mass Σw (per slot), max cosine to the cache; argmax
   agreement, KL(p_clip‖p_cache)/log C, JS, rank of CLIP's top class in p_cache; self-vote flag.
3. **Oracle map** (labels in analysis only): per sample, α* = argmin_α NLL on a grid
   α ∈ [0, 4] (step 0.1). Plot α* and benefit = NLL(0) − NLL(α=2) vs each signal, all
   datasets overlaid on the same axes. Also NLL with α* = ceiling for any rule.
4. **Signal ranking**: AUROC of each signal for "benefit > 0", Spearman with α*, per dataset
   and pooled. Two-expert heatmap: P(CLIP right) and P(cache right) over (CLIP conf × cache
   conf), split agree/disagree.
5. **Rule ladder**, each fitted by NLL: constant α (TDA) · step on best signal · logistic on best
   signal · 2-D table on the best pair · small monotone logistic on all signals · oracle.
   Choose the simplest rule within a small NLL margin of the best (decide the margin before
   looking at test).
6. **Generalization tests**: (a) proof of concept now — LODO across the five **val** splits;
   (b) overlay test; (c) class-subset test (one dataset restricted to random 10 / 50 / all
   classes: curves must coincide); report fold-to-fold parameter spread and transfer gap
   (LODO NLL vs in-dataset fit).
7. **Development datasets** (if 6a/6b pass): one more Colab run extracting ImageNet-val
   (subset ≥ 10k OK), Caltech101, UCF101, Food101 with the T01 pipeline (needs dataset
   support in the extractor/downloader; ImageNet requires a Kaggle or image-net.org account —
   ask Idan). Skip SUN397 (37 GB) and Stanford Cars (download unreliable).
8. **Fit and freeze** the rule on the four development datasets with LODO among them; record
   the frozen formula and constants in DECISIONS.
9. **Test once** on the five project test sets, P1 and P2, 3 seeds (10 at wrap-up), vs CLIP,
   TDA, gate, temperature, baselines. Then the deck story changes (PLAN "Storyline").

## Acceptance
- Rule + constants frozen before step 9, logged in DECISIONS with the dev-set evidence.
- LODO table, overlay figure, class-subset figure, oracle ceiling, rule-ladder table — all
  from tested scripts, CSVs under `results/analysis/adaptive_alpha/`.
- Test-set results with paired-by-seed differences vs TDA and vs the gate.

## Decisions to log when starting (lead)
- 2026-11-10 (Idan): the method must be dataset-independent; formula designed on separate
  development datasets (ImageNet + others), tested once on the five datasets; val splits only
  for the proof of concept.
- 2026-11-10 (Idan): consolidated order — dev-set extraction → diagnosis evidence → adaptive
  cache weight → calibration inside the active region (R13 or R4) → fair evaluation
  (P1/P2, seeds, baselines) → deck rebuild → report. Original changes become findings;
  the gate is reframed as "right idea, wrong criterion".
