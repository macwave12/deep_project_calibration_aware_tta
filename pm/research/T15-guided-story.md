# Proposed research structure: from TDA's self-vote to the agreement gate

Draft for agreement with Idan, 2026-11-11. This is a structure to agree on before redoing it
"the right way". Numbers are from runs we already have. Most come from tracked CSVs
(`results/analysis/t14/*`, `results/protocols_10seeds.csv`). The ones marked *prelim.* come
from `analysis/story_stats_preliminary.py`, which is untested
(`results/analysis/t14/story_stats_preliminary.csv`). Setting: CLIP ViT-B/16, zero-shot, P2
(α = 2, β = 5), seeds 0/1/2 unless stated otherwise.

## The argument in one paragraph

TDA builds its cache from CLIP's own pseudo-labels, and every image is written to the cache
before it is read (K8). So the cache often **echoes CLIP back to itself**. The extreme case is
the **self-vote**, where the image votes for its own label. We show three things:

1. Where the cache agrees with CLIP, it **provably cannot change the decision and can only raise
   confidence**.
2. That inflation is large and does not depend on how calibrated CLIP was, so on most datasets
   it overshoots.
3. All of TDA's accuracy gain comes from the samples where the cache disagrees with CLIP.

Hence the method: use the cache only where it disagrees. A label-free test of how much of the
stream the cache cannot affect predicts the datasets where this is not worth doing.

## Step 1: the self-vote and its cost

- **Claim.** An admitted image is in the cache when it is read, so it contributes exp(0) = 1 to
  its own pseudo-label: +α to that logit, carrying no information about the true label.
- **Math (Lemma 1).** c_k(x) = Σ_{i ∈ cache_k} exp(−β(1 − ⟨f(x), f_i⟩)), so each term is ≤ 1.
  If x is cached under ŷ₀, then c_ŷ₀(x) ≥ 1. The cache can disagree with CLIP only if another
  class has more kernel mass than that.
- **Evidence** (*prelim.*):
  - The self-vote share is 2% (EuroSAT) to 60% (ImageNetV2).
  - It makes agreement almost certain: P(cache agrees with CLIP | self-vote) is **89–99%**,
    against 28–90% without one.
  - Removing only the self term cuts TDA's ECE: ImageNetV2 15.1 → 5.0, DTD 17.4 → 11.7,
    Aircraft 18.1 → 13.3, UCF101 8.5 → 5.3, Flowers 9.6 → 6.8.
  - …but it costs accuracy: ImageNetV2 −1.3, Flowers −1.1, Aircraft −0.9. So deleting
    self-votes is not the fix. Self-votes are the most visible symptom of a broader echo.
- **Figure 1** (to make): self-vote share and P(agree | self) per dataset; TDA's ECE with and
  without the self term.

## Step 2: when CLIP is confident, the cache cannot change its mind; it only inflates

- **Math (Lemma 2, no-flip region).** TDA changes CLIP's decision only if, for some class j,
  z_ŷ₀ − z_j < α(c_j − c_ŷ₀) + (n_ŷ₀ − n_j). Since 0 ≤ c_k ≤ S (shot capacity 3), no flip is
  possible once CLIP's top-2 logit gap exceeds αS + range(n). Under P2 that is roughly
  p₁/p₂ > e⁶ ≈ 400 from the cache alone. The condition is label-free.
- **Evidence** (*prelim.*):
  - 0 flips in the region on all 9 datasets. Its share is 1–12% on most datasets, but 37%
    (Pets), 42% (Food101) and 68% (Caltech101).
  - Inside it TDA moves only confidence: the confidence − accuracy gap rises on every dataset
    (e.g. ImageNetV2 +1.7 → +4.9, Flowers +2.2 → +4.9).
  - The bin view (A11) shows net decisions fixed only at CLIP confidence ≲ 0.6, but TDA above
    CLIP's confidence in every bin (`analysis/figures/echo_val.png`; to be regenerated under P2).
- **Figure 2:** the A11 three-row figure under P2, with the no-flip boundary marked.

## Step 3: where the gain is — agreement vs disagreement

Step 2 is about confidence. The decisive split is whether the cache agrees with CLIP.

- **Math (Lemma 3, agreement ⇒ same decision, higher confidence).** Let b = z − n. If
  argmax c = argmax b = ŷ₀, then for every j:
  o_ŷ₀ − o_j = (b_ŷ₀ − b_j) + α(c_ŷ₀ − c_j) ≥ b_ŷ₀ − b_j > 0.
  So the decision is unchanged and every margin grows, and therefore
  max softmax(o) ≥ max softmax(b). ∎
  - **Corollary:** on agreeing samples, TDA's accuracy equals CLIP + negative cache's, and its
    mean confidence − accuracy gap can only increase. Whether ECE gets worse depends on the
    starting gap (Step 5).
  - Lemma 2 is a special case: in the no-flip region the cache can only agree in effect.
- **Evidence** (test, mean of 5; `mechanism_test.csv`):

  | samples | share | acc: CLIP + neg → TDA | gap: CLIP + neg → TDA |
  |---|---:|---|---|
  | cache agrees | 56% | 68.79 → 68.96 (no gain) | −2.6 → **+11.3** |
  | cache disagrees | 44% | 36.72 → **41.08** (all the gain) | +7.9 → +8.4 |

  The dev sets show the same split (`mechanism_dev.csv`): agreeing 84.20 → 84.20, gap
  −2.7 → +6.9.
- **Figure 3:** two panels (agree / disagree): accuracy and gap for CLIP, CLIP + negative
  cache and TDA.
- **This leads directly to the method:** keep the disagreeing branch, drop the agreeing one.

## Step 4: the method and its results

**Agreement gate:** α(x) = α·[argmax c ≠ argmax z]. It adds no parameters.

**Five test sets, P2, 10 seeds, 90% CI** (`protocols_10seeds.csv`, `protocols_paired_10seeds.csv`):

| method | accuracy | ECE |
|---|---:|---:|
| CLIP | 53.12 | 5.65 |
| TDA | 56.75 | 10.96 |
| **agreement gate** | **56.66** | **5.51** |

- Against TDA: Δ accuracy −0.09 [−0.13, −0.04], Δ ECE −5.45 [−5.54, −5.37], 10/10 seeds.
- On the 5-dataset mean the gate also beats CLIP on both metrics. Per dataset its ECE is worse
  than CLIP's on Aircraft, Flowers and Pets.
- **Matched comparison:** at the gate's accuracy, the best ECE TDA reaches along its α-curve is
  10.39 (`matched_test.csv`; figure `t14_acc_ece_test.png`).
- **Dev sets:** mean Δ −0.01 / −4.07. 2 wins, 1 tie, 1 loss (Caltech101).
- **Appendix:** P1 (the gate gives −0.06 / −4.83) and the P1 reproduction of the published
  numbers.

## Step 5: why "CLIP is calibrated" is the right premise, stated precisely

This step refines the premise rather than asserting it.

- **What the data say.** CLIP is near-calibrated overall (test ECE 5.65; dev 3.32). On
  agreeing samples it is calibrated to mildly underconfident: the gap is +2.6 on Aircraft and
  +0.8 on DTD, −0.3 to −0.7 on ImageNetV2 and Food101, and −5 to −12 on UCF101, Flowers,
  Pets, Caltech101 and EuroSAT (*prelim.*).
- **What the math gives** (Lemma 3). The echo only adds confidence on those samples.
- **What the data add.** How much it adds is not controlled by how calibrated CLIP was. On
  agreeing samples, TDA raises the gap over CLIP + negative cache by +3.8 (Pets) to +23.4
  (Aircraft) on the test sets (`mechanism_test.csv`). UCF101 and Caltech101 start from the same
  gap (−5.8 and −5.5) but end at +7.3 and −0.7 (`mechanism_dev.csv`).
- **Consequence:**
  - on 7 of 9 datasets the echo overshoots into overconfidence;
  - it helps only where it happens to land near zero (Caltech101, Pets).
- So the precise premise is: **"the echo adds confidence and no accuracy, by an amount that is
  unrelated to how calibrated CLIP was; so it overshoots unless CLIP was underconfident by
  about the right amount."** It is not true that CLIP is always calibrated.

## Step 6: when not to apply it — a label-free "ban" test (hypothesis)

The datasets where the gate loses or ties (Caltech101, Pets, Food101) are the ones where most
of the stream is in the no-flip region (Lemma 2). There the cache has almost no decision power
and CLIP is in its easy, typically underconfident regime.

| no-flip share (*prelim.*) | datasets | gate vs TDA |
|---|---|---|
| ≥ 37% | Caltech101 68%, Food101 42%, Pets 37% | loss, tie, loss |
| ≤ 12% | UCF101, ImageNetV2, DTD, EuroSAT, Flowers, Aircraft | wins |

- **Proposed rule.** Estimate the no-flip share on the first few hundred stream samples. This
  is label-free and online. Apply the gate only if the share is below τ; otherwise keep TDA.
- **Alternative signal:** CLIP's mean confidence separates the same groups (83–88% vs ≤ 68%).
- **Caveats:**
  - this is 9 datasets, and τ would be chosen after looking at them;
  - the agreement-rate accuracy estimate I also tried does *not* work (Pets 66% vs 88%
    accuracy);
  - so τ must be pre-registered and tested on fresh datasets (below).
- **Figure 6:** scatter of no-flip share against Δ ECE (gate − TDA), one point per dataset,
  with τ marked.

## Doing it "the right way" (proposal)

1. **Freeze now:** this structure, the figures, the method, and the ban rule with τ fixed in
   advance (e.g. 25%, between 12% and 37%).
2. **Fresh evaluation datasets that have never been looked at:** ImageNet-val (50k, the
   canonical set) plus the public ImageNet-A, -R and -Sketch. Optionally SUN397 or Stanford
   Cars subsets.
3. **Run once:** CLIP, TDA, the gate, and the gate + ban rule. Report the per-dataset ban
   decision and whether it was right.
4. **Regenerate** every figure above under P2 from tested scripts. The preliminary script
   becomes a tested module.
5. **Deck (T10)** following steps 1–6, with P1 and the appendix tables at the end.

## Questions to agree on

- Is this order right? An alternative leads with Step 3 (agreement) and treats the self-vote
  (Step 1) as its most visible special case.
- Should the ban test be part of the method ("gate + abstain"), or a separate diagnostic?
- ImageNet-val: will you run the Colab extraction? It needs ImageNet access (Kaggle or
  Hugging Face terms). ImageNet-A, -R and -Sketch I can run locally on CPU.
