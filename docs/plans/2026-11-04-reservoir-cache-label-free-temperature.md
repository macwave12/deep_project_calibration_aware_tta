# Plan: sampled cache + unbiased label-free temperature

Date: 2026-11-04. Status: **plan only — do not implement** until the walkthrough
(sessions 1–6) is finished. Base branch: `publish` (bdd57c1).

## 1. Why

Two observations from the existing results drive this plan.

**TDA's miscalibration is almost entirely a uniform overconfidence shift.** Mean
confidence minus accuracy (from the `reliability` field of `results/raw/*.json`):

| dataset | CLIP gap | TDA gap | TDA ECE |
|---|---:|---:|---:|
| DTD | +7.9 | +17.2 | 17.4 |
| Flowers | +0.3 | +6.4 | 6.4 |
| Pets | −3.5 | −1.3 | 2.5 |
| EuroSAT | +5.3 | +13.1 | 14.4 |
| Aircraft | +4.7 | +15.5 | 15.5 |

ECE ≈ |global gap| on every dataset, so a single well-chosen temperature should
remove most of it. The open problem is choosing T **without labels**.

**The current label-free temperature (`cal_tda_temp`) is limited by its accuracy
estimate `â`, which errs in both directions:**

| dataset | TDA ECE | temp ECE | mean T | % pinned at T=1 | reading |
|---|---:|---:|---:|---:|---|
| DTD | 17.41 | 4.68 | 1.65 | 1% | `â` about right |
| Aircraft | 15.56 | 4.21 | 1.49 | 0.5% | `â` about right |
| Flowers | 6.51 | 4.85 | 1.27 | 65% | `â` partly too high |
| EuroSAT | 14.43 | 13.43 | 1.05 | 88% | `â` too high — method barely acts |
| Pets | 2.48 | 15.54 | 1.81 | 0.2% | `â` too low — softens an underconfident model |

Suspected causes:
1. **Cache too small for leave-one-out.** 3 items/class ⇒ each item has only 2
   same-class neighbours; the 1-NN is often another class even when labels are
   right ⇒ `â` biased low (Pets).
2. **Cache is a biased sample.** TDA keeps the 3 lowest-entropy images per class, the
   easiest ones; agreement among them overstates stream accuracy (EuroSAT, Flowers).
3. **Grid floor T ≥ 1** cannot fix underconfidence (Pets). The floor was introduced
   because margin admission inflated `â`; margin admission is now dropped.
4. **Correlated errors** — tight clusters of identically mislabeled images agree with
   each other. Not fixable by reweighting; measured, reported as a limitation.

## 2. Idea

The cache is a sample of the test stream. Make it a statistically sound sample, and
both the prediction vote and the label-free accuracy estimate improve.

- **Sample** biased toward confidence (purity of pseudo-labels).
- **Estimate** with that bias undone (representativeness of `â`).

Why a fit by NLL is not used: with pseudo-labels `ŷ = argmax p` (positive or
negative), the likelihood is monotone in 1/T and is minimized at T → 0. Any label
derived from the model's own scores says "be more confident". An independent
estimate of the error rate is required; that is what `â` provides.

## 3. Design

All behind `CalTdaConfig` flags; defaults reproduce current behaviour exactly.

### 3.1 Cache policy — `cache_policy: topk | reservoir`, `cache_capacity: K`
- `topk` (default, K=3): upstream TDA rule (keep K lowest-entropy per pseudo-class).
- `reservoir`: weighted reservoir sampling per pseudo-class (Efraimidis–Spirakis
  A-Res). Each arriving image gets key `u^(1/w)`, `u ~ U(0,1)`, `w` = CLIP top-1
  probability; keep the K largest keys. Store `w` with the item. Uses a seeded RNG
  so runs are reproducible.

### 3.2 Vote normalization — `cache_vote_norm: sum | mean`
Cache logit is a **sum** over items; raising K from 3 to 16 multiplies the vote by
~5 and inflates confidence, confounding the capacity sweep. `mean` divides each
class's vote by its item count (×3 to keep α on the original scale at K=3).
Default `sum`.

### 3.3 No self-vote — `read_before_write: false | true`
Compute cache logits before inserting the current image, so an admitted image does
not vote for itself with weight `exp(0)=1`. Required in every new configuration so
sampling effects are not mixed with the self-vote. Default `false`.

### 3.4 Accuracy estimator — `loo_estimator: nn1 | knn_ipw`, `loo_k`
- `nn1` (default): current 1-NN agreement, unweighted.
- `knn_ipw`: each item's agreement = its pseudo-label equals the majority label of
  its k nearest other cached items; aggregate with inverse-probability weights
  `1/w_i` (Horvitz–Thompson):
  `â = Σ (1/w_i)·agree_i / Σ (1/w_i)`.
  Only meaningful with `cache_policy: reservoir` (under `topk` inclusion
  probability is not ∝ w); config validation rejects the combination.

### 3.5 Temperature grid — `temp_grid_min`
Default `1.0` (current). New configs use `0.5` to allow sharpening.

### 3.6 Diagnostics written to each run record
- `â` and true running accuracy at every step (labels used **only** in the scorer,
  after the run, as for ECE) → **estimator error** `mean |â − acc|`, and its sign.
- Fraction of the stream admitted to the cache, and fraction that received a
  self-vote (for `read_before_write: false`).
- Existing temperature diagnostics (mean T, boundary fractions).

## 4. Reference point: oracle temperature (no rerun)

Fit one T per dataset by NLL on TDA's saved logits using the **true test labels**.
Reported only as an oracle ceiling ("best any single T can do"), never as a method.
Runs from `online_results/**/save_results.pt` on the local machine (those files are
not in git). Note: `tda` saves logits, `cal_tda` saves probabilities.

Also from the same files, at no cost: NLL and Brier score for every existing run,
and a running-T vs fixed-T comparison (does the best T drift as the cache fills?).

## 5. Experiments (staged)

Stage A — screen (5 datasets × 2 seeds = 30 runs, ~1.5–2 GPU h):

| config | policy | K | vote | read-before-write | estimator | T grid |
|---|---|---:|---|---|---|---|
| A0 = current `temp` | topk | 3 | sum | no | nn1 | ≥1 |
| A1 | reservoir | 16 | mean | yes | knn_ipw | ≥1 |
| A2 | reservoir | 16 | mean | yes | knn_ipw | ≥0.5 |

Proceed to stage B only if A1 or A2 reduces mean estimator error vs A0.

Stage B — attribution (which ingredient matters):
- K ∈ {3, 8, 16} × policy ∈ {topk, reservoir}, estimator and grid fixed at the
  stage-A winner.
- `read_before_write` on/off at the winning K/policy, temperature off — isolates the
  self-vote's share of TDA's overconfidence gap.

Seeds: reservoir sampling and stream order are both random; report mean ± std over
≥2 seeds (3 if time allows) for every stage-B number.

## 6. Success criteria (fixed before running)

1. Estimator error `mean |â − acc|` lower than A0 on ≥4/5 datasets.
2. ECE lower than `cal_tda_temp` on average, without Pets degrading below TDA's
   2.48%, and EuroSAT ECE clearly below TDA's 14.43%.
3. Accuracy within seed noise of TDA (temperature cannot change argmax; any accuracy
   change comes from the cache policy and is reported separately).
4. Gap to the oracle-T ceiling reported for every dataset.

## 7. Risks

- Correlated errors keep `â` high on clustered datasets (likely EuroSAT). Diagnostic
  in 3.6 will show it; report rather than tune around it.
- Larger caches change accuracy, not only calibration — must be separated (stage B).
- A-Res inclusion probabilities are only ∝ w when K ≪ images per class; Flowers has
  ~24 test images/class, so at K=16 the IPW correction is approximate there.
- Any new threshold/weight tuned on these test sets is test-set tuning; all new
  knobs use fixed, a-priori values listed above.

## 8. Implementation checklist

- `online_method/cal_tda.py`: new `CalTdaConfig` fields; reservoir update; vote
  normalization; read-before-write ordering; `inert_fields()` updated.
- `utils/loo_temperature.py`: `knn_ipw_accuracy(features, labels, weights, k)`;
  grid min parameter.
- `utils/evaluation.py`: estimator-error diagnostics from the logged `â` trace.
- `configs/method/plan_reservoir/{a0,a1,a2}.yaml`.
- Tests: defaults still reduce to upstream TDA (extend `test_cal_tda_runner.py`);
  reservoir keeps ≤K per class and is seed-deterministic; IPW estimator recovers a
  known accuracy on a synthetic biased sample; no label reaches the method
  (`test_no_label_leak.py` covers new code paths).
- Oracle-T + NLL/Brier script under `analysis/` reading local `save_results.pt`.

## 9. Backlog from the walkthrough (not in this plan's scope)

- Negative-cache ablation (`use_neg_cache=False`): does TDA's one damping term matter?
- LOO-consistency eviction of suspected mislabeled cache items.
- Selective-prediction metric (risk–coverage / AURC).
- Related work: add C-TPT (ICLR'24), O-TPT (CVPR'25), Tip-Adapter; narrow the
  novelty claim to label-free calibration of online, backprop-free, cache-based TTA.
- Backbone mismatch: proposal motivates with RN50 (benchmark Table 7), experiments
  use ViT-B/16 — state it or add RN50 runs.
- README Results section still describes the original three-contribution bundle.
