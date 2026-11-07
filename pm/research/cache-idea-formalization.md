# Soft-label tie-breaking cache — formalization of Idan's idea (2026-11-07)

Status: research note. **Not part of the wrap-up**; to be developed after T12, as an
extension of the gated method.

## 1. Intuition (Idan)
TDA's cache stores the most confident image per class with a one-hot pseudo-label: it
represents the *centre* of each class. The gate only uses the cache on hard decisions
(two-way ties), exactly where those centres are least informative. Instead, store images
whose CLIP distribution is concentrated on a *few* classes, keep their *soft* distribution,
and let similar cached images vote with it. The cache then describes the decision
boundaries ("an orthogonal basis over the probabilities"), complemented by decisive anchors.

## 2. Notation
- f(x): L2-normalized image feature; p(x) = softmax(z(x)) ∈ Δ^{C−1}: CLIP's posterior.
- Significant set S_τ(x) = {k : p_k(x) > τ}; multiplicity m(x) = |S_τ(x)|;
  concentration κ_τ(x) = Σ_{k∈S_τ(x)} p_k(x).
- Soft label (tail truncated): q(x) = p(x) ⊙ 1[S_τ(x)] / κ_τ(x).
- Kernel: w(x, x') = exp(−β (1 − f(x)·f(x'))), as in TDA.

## 3. Cache construction (write rule)
- Eligible: m(x) ≤ n and κ_τ(x) ≥ ρ ("most of the mass on at most n classes").
- One bucket B_k per class with capacity c. An eligible item is offered to the bucket of
  every class in S_τ(x) (stored once, referenced from several buckets).
- Strata by multiplicity: c = c_1 + c_2 + … + c_n slots. m = 1 items are **anchors**
  (TDA's role); m ≥ 2 items are **boundary items**.
- Priority inside a stratum (higher is better):
  π(x) = −H(q(x))/log m(x) − λ_div · max_{j∈B_k} f(x)·f_j − λ_neg · max_{j∈N} f(x)·f_j,
  where N is the negative ("null") cache. The first term prefers decisive items
  within their support; the second spreads the bucket (diversity); the third keeps away
  from the null region.
- A-priori defaults (no tuning on test): τ = 0.1, n = 3, ρ = 0.8, c = 6 with strata 2/2/2,
  λ_div = λ_neg = 0 in the first version (then ablated). c is swept on **val**.

## 4. Read rule and combination
Evidence mass W(x) = Σ_i w(x, x_i); soft-label vote V(x) = Σ_i w(x, x_i) q_i.

**Dirichlet-mean combination** (CLIP as a prior worth λ₀ pseudo-counts):

  p̂(x) = (λ₀ p(x) + V(x)) / (λ₀ + W(x)).

The cache weight ω(x) = W/(λ₀ + W) adapts by itself: no similar cached image → CLIP;
many similar, agreeing images → the cache. No softmax of cache logits, hence **no scale
mismatch** (the failure of probabilistic fusion) and no arbitrary mixing weight.

**Tie-restricted output** (extension of the gate). Let T(x) = top-2 classes of p(x) when
the gate fires (or S_τ(x) when m ≥ 2), M_T = Σ_{k∈T} p_k(x):

  p_out,k = M_T · p̂_k / Σ_{j∈T} p̂_j  for k ∈ T;   p_out,k = p_k(x) otherwise.

CLIP decides *how much* mass the tie gets; the cache decides only *how to split it*.
An optional temperature on the within-tie split is a single scalar, chosen on val.

## 5. Properties (the mathematical backing)
- **P1 Bounded confidence.** p_out keeps CLIP's distribution outside T and max_k p_out,k ≤ M_T.
  The method can never be more confident than "CLIP's mass on the tie". Contrast TDA, whose
  additive rule is a product of experts (softmax(z + αc) ∝ p_clip · p_cache^α) and sharpens
  beyond either input.
- **P2 Convex bound.** (V/W)_k ≤ max_i q_{i,k}: the cache vote cannot exceed its most
  confident neighbour on any class (Idan's convex-hull point: the vote can *amplify the
  strongest pick relative to others*, not beyond the inputs).
- **P3 Ambiguous items can still break ties decisively.** What matters on a tie (a, b) is the
  ratio r_ab = (λ₀p_a + V_a)/(λ₀p_b + V_b), not the maximum. Idan's example (4 classes, tie
  between classes 2 and 3, 1-indexed): 0.4·[0,0,.5,.5] + 0.3·[0,.5,.5,0] + 0.3·[.5,0,.5,0]
  = [0.15, 0.15, 0.50, 0.20] (the note had classes 2 and 4 swapped; conclusion unchanged).
  Restricted to {2, 3}: 0.15 vs 0.50 → split 0.23 / 0.77, a decisive tie-break built from
  items none of which is class-specific.
- **P4 Limit: systematic confusion is not corrected.** If every nearby cached item carries
  the same ratio as x on (a, b), r_ab ≈ p_a(x)/p_b(x) and nothing changes. Boundary items
  help where CLIP's confusion is *local noise* (varies between nearby images); anchors are
  needed where it is *bias*. This is why anchors (m = 1) stay in the cache.
- **P5 Coverage objective ("orthogonal basis").** Let G be the confusion graph: an edge (a, b)
  weighted by how often CLIP's top-2 is {a, b} on the stream (label-free). A good cache covers
  every heavy edge with items that are *discriminative for the pair*: q_a + q_b large and
  |q_a − q_b| large, in both directions. This replaces "orthogonal basis" with a measurable
  criterion: edge coverage of G.

## 6. Risks and how to test them
- Boundary items are CLIP's least reliable predictions → measure soft-label quality
  (cross-entropy of q against the true label) per stratum, in analysis only.
- Bias vs noise (P4) → per confused pair, measure how consistent neighbours' ratios are.
- Too many knobs → fix all but c a priori; sweep c and λ₀ on val only.

## 7. Variants to compare (after wrap-up, P2 protocol, val selection)
- V1 tie-restricted (above): direct extension of the gate.
- V2 global Dirichlet mean p̂ on every sample: a principled replacement of probabilistic
  fusion; may recover TDA's flips outside ties (the gate's EuroSAT loss) with bounded
  confidence.
- Ablations: one-hot vs soft labels; anchors only vs anchors + boundary; strata split; c.

## 8. Related ideas to check before claiming novelty
Soft-label k-NN / Nadaraya–Watson; label propagation with soft labels (ECALP, ICLR'25);
prototype evolution (DPE, NeurIPS'24); cache-noise mitigation (ICLRW'25); Bayesian
class-embedding updates (BCA, CVPR'25); uncertainty sampling in active learning;
evidential / Dirichlet classifiers.
