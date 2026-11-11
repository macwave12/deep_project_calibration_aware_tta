# T16 — Wrap-up plan: final research, paper and presentation

Draft for agreement with Idan, 2026-11-11. When agreed, it replaces the phase table in PLAN.md.
The decisions it needs are listed at the end; the rest of the plan does not depend on them.

## 0. What the project now claims

1. **Diagnosis.** TDA's cache is built from CLIP's own pseudo-labels and is read after it is
   written. So wherever the cache agrees with CLIP it only echoes CLIP: the decision is the same
   and confidence is higher, which is proven (Lemmas 1–3). TDA's whole accuracy gain comes from
   the samples where the cache disagrees.
2. **Method.** The agreement gate: use the cache only where it disagrees. It has no parameters
   and is always on.
3. **Evidence.** It keeps TDA's accuracy and halves its ECE. It holds across about 19 datasets,
   and the gain is largest where CLIP is not confident.

Everything below serves these three claims. Anything that does not serve them moves to an
appendix or "earlier attempts".

## 1. Paper structure (`report/`, rewritten)

| § | Section | Content | Main figures and tables |
|---|---|---|---|
| — | Abstract | the three claims, the headline numbers | — |
| 1 | Introduction | TTA makes CLIP overconfident; why TDA; contributions | teaser: accuracy–ECE plane (CLIP, TDA, gate) |
| 2 | Background | zero-shot CLIP, TDA (positive and negative cache, write-before-read), ECE and reliability | TDA pipeline diagram |
| 3 | **Diagnosis** | 3.1 self-vote; 3.2 unsure vs confident CLIP; 3.3 agreement vs disagreement; 3.4 the lemmas | step 1, 2 and 3 figures (val + test panels) |
| 4 | Method | the gate, its cost (one comparison), implementation | — |
| 5 | Setup | the datasets in three tiers: 5 core test sets, 4 development, ~10 fresh; protocol; seeds and CIs; CPU replay | dataset table |
| 6 | **Results** | 6.1 main table (10 seeds, CIs); 6.2 matched comparison (α-curve, TDA + T); 6.3 generalization (dev + fresh); 6.4 where it helps (ρ); 6.5 TDA-paper analyses with ECE; 6.6 baselines; 6.7 cost | main table, accuracy–ECE figure, generalization table, ρ figure, Fig. 3/4 and Table 6 analogues |
| 7 | Earlier attempts | margin admission, probability fusion, two-way-tie gate, LOO temperature, NLL-fitted α(x): what each did and why it falls short | one compact table |
| 8 | Limitations | underconfident CLIP (Pets, Caltech101), the gated branch's remaining overconfidence, single backbone, design on val, selection history | — |
| 9 | Conclusion | | |
| App. | | A protocol P1 (reproduction + P1 vs P2); B proofs; C per-dataset tables; D reliability diagrams; E selection-history disclosures | |

## 2. Presentation structure (`presentation/build_pptx.py`, deck v3, ~26 slides)

The deck follows the paper and keeps the PLAN requirement that at least 25% is about TDA.

1. Title, agenda (2)
2. Problem and setting: CLIP zero-shot, why TTA, the landscape, why TDA (4). Reuse v2's
   `slide_clip`, `slide_landscape`, `slide_why_tda`.
3. TDA in depth: idea, write/read, α/β, negative cache, product of experts, worked example,
   gains (7). Reuse v2's `slide_tda_*`.
4. Calibration: ECE, reliability, CLIP vs TDA (3). Reuse.
5. **Diagnosis:** self-vote; unsure vs confident; agreement split; the lemmas (4). New.
6. **Method:** the gate (1). New.
7. **Results:** main table; matched comparison; generalization; where it helps; TDA-paper
   ablations; baselines (6). New.
8. Earlier attempts (1); limitations and next steps (1); conclusion (1); backup: P1, proofs,
   per-dataset (3+).

## 3. What gets recalculated

| Item | Datasets / seeds | Status |
|---|---|---|
| Main table: CLIP, TDA, gate (+ TDA + T, temperature) | 5 core × 10 seeds | **done (P2)**; redo the gate under P1 for the appendix (3 seeds, done) |
| Matched comparison (α-curve, TDA + T) | 5 core × 3 seeds | done; extend the α-curve to 10 seeds |
| Diagnosis figures: val panels + test/dev panels | 5 val, 5 test, 4 dev | val done; **add test/dev**, merge steps 2+3 |
| Generalization | 4 dev (done) + ImageNet-val (running) + 9 fresh (Colab) | in progress |
| Where it helps (ρ over all datasets) | ~19 | redo when the fresh sets are in |
| Reliability diagrams | 5 core | **redo** for CLIP / TDA / gate |
| TDA Fig. 3 analogue: positive-only / negative-only / both, + ECE | 5 core | **new**. Key: the negative cache gives half the gain (K39) |
| TDA Fig. 4 analogue: shot capacity 1–6, + ECE, TDA vs gate | 5 core | **new**. Does more cache mean more echo? |
| TDA Table 6 analogue: α × β heatmaps, + ECE, P1/P2 points | 5 core | **new**. Echo grows with α; covers the protocol question |
| TDA Fig. 1 analogue: per-image cost | 5 core | **new**, cheap |
| Baselines (see §5) | 5 core × 3 seeds (+ ImageNet) | **new** |
| Earlier-attempts table (margin, fusion, two-way-tie gate, temp, NLL-fitted α(x)) | 5 core, P2 | numbers exist; **regenerate** in one table |
| Proofs written out | — | **new** (appendix B) |
| Fresh-set P1 | ImageNet family only (TDA has no α/β for the others) | after the Colab run |

Skipped, with a sentence why: TDA Fig. 5 (negative mask p_l) and Table 5 (entropy band). They
are about the negative cache's thresholds and do not touch the echo argument.

## 4. What is no longer central (moves to "earlier attempts" or the appendix)

- **Margin admission, probability fusion (gated or not), `fusion_sharpen`, gate + temp:** one
  row each in §7. Their separate v2 slides (`slide_margin`, `slide_fusion_*`, `slide_gate_*`,
  `slide_temp_*`) collapse into one slide.
- **The LOO temperature:** a finding (it calibrates TDA, but Pets ECE is 15.4). The story keeps
  it as "what a calibrator alone does", next to TDA + T.
- **The 2×2×2 ablation table** (`tables/ablation*.tex`): moves to the appendix.
- **`results/summary.csv`** (GPU, P1, seed 0) is no longer a headline. It stays the
  reproduction check in appendix A.
- **The ban rule:** a record only (DECISIONS). The "where it helps" relationship replaces it.

## 5. Comparison methods

| Method | Type | Plan |
|---|---|---|
| CLIP zero-shot | base | run |
| TDA | base | run |
| **TDA + global temperature** (fitted on val, with labels) | calibration reference | run (strongest simple competitor) |
| CLIP + negative cache (α = 0) | ablation | run (shows where TDA's gain comes from) |
| **ECALP** (ICLR'25) | online, backprop-free | replay exists; add scoring |
| **DMN-weak** (CVPR'24) | online memory cache, backprop-free | replay exists; add scoring |
| OnZeta | online, backprop-free | replay exists; add scoring if cheap |
| TPT, C-TPT, O-TPT | episodic, backprop | cite their ViT-B/16 ECE; different setting, stated |
| Tip-Adapter (16-shot) | few-shot reference | optional; appendix only |

**Optional extension:** apply the gate to DMN-weak's memory, which is also built from
pseudo-labels. If the echo is generic to pseudo-label caches, that is a second piece of
evidence. About half a day.

## 6. Prior-art check (before anything is written as "ours")

Search for agreement- or disagreement-gated caches in TTA (TDA follow-ups: BoostAdapter,
DPE, DOTA, FreeTTA, BCA, CRG and others), for "self-voting" or "confirmation bias" in
pseudo-label caches, and for co-training / disagreement-based adaptation. Output: a table in
`research/related-work.md` and one honest sentence in §1. This is T09's third pass.

## 7. Order of work (checkpoints with Idan marked ◆)

1. Prior-art check (§6) ◆ (if something close exists, the framing changes).
2. Fresh-set Colab run (Idan) → runs, generalization table, ρ figure.
3. Recompute (§3): TDA-paper analogues, reliability diagrams, baselines, earlier-attempts
   table, diagnosis test/dev panels. All from tested scripts into tracked CSVs.
4. Reviewer agent on all new analysis ◆.
5. Paper rewrite (§1) → build the PDF → reviewer pass ◆.
6. Deck v3 (§2) from the same CSVs → render → reviewer pass ◆.
7. Final 10-seed reruns where a headline number still has 3 seeds; freeze; tag a release.

## 8. Decisions (Idan, 2026-11-11)

1. **Headline protocol: P2.** P1 goes to the appendix, as the reproduction of the published
   numbers plus the P1-vs-P2 check.
2. **Baselines:** ECALP, DMN-weak and TDA + T (val-fitted temperature). TPT, C-TPT and O-TPT are
   cited, not run. OnZeta and Tip-Adapter are not run.
3. **Gate on DMN-weak's memory:** a stretch step, only if time allows.
4. **TDA-paper analyses:**
   - **Kept:** Fig. 3 as one compact panel (CLIP / positive-only / negative-only / both, with
     ECE; exact and free from the traces, and it shows the negative cache's share of the gain);
     Fig. 1 (per-image cost).
   - **Skipped:** Fig. 4 and 5, Table 5. The accuracy gain lives in the disagreeing samples;
     one limitations sentence covers it.
   - **Table 6:** its α part is our α-curve (matched comparison); no β sweep.
5. **Novelty check:** a separate research agent, before the paper is written (§6).

## 9. After the novelty check (2026-11-11)

Full findings: `related-work.md`, third pass. Consequences for the plan:

- **Framing.** Lead with the mechanism: a cache of CLIP's own pseudo-labels, written before it is
  read, echoes CLIP. Then the lemmas, then the cause-level fix. Credit ZAEC for the empirical
  observation ("sharpening without changing the prediction"), Malach & Shalev-Shwartz (2017) for
  the disagreement principle, and MARC for margin-style no-flip conditions. "All the gain is in the
  disagreeing samples" is stated as a consequence of Lemma 2, not as a finding.
- **Baselines (added):** ZAEC, exact and label-free on our traces, as the mandatory one. SaLS
  logit-range rescaling if a faithful re-implementation is clear from the paper. Both also on top
  of the gate, because the gate and post-hoc calibration combine.
- **Results to report:**
  - gate vs TDA + ZAEC (test: 5.51 vs 5.95 at −0.09 accuracy; dev: 3.30 vs 3.70 at equal
    accuracy);
  - gate + ZAEC (test 5.27);
  - the per-dataset mix, where ZAEC wins on Flowers, UCF101 and Caltech101.
- **Still to do:** the Semantic Scholar scan of papers citing TDA (rate-limited today), and a
  check of the full texts of DMN, BoostAdapter, DPE and COSMIC for agreement-dependent fusion.

## 10. Format constraints (Idan, 2026-11-11), overriding §1–§2 where they differ

- **Paper: about 4 pages** (course report), plus references and an appendix if allowed. Audience: a professor, so it
  stays rigorous rather than simplistic. Main text, in order:
  1. intro, with the contribution stated against ZAEC;
  2. TDA background, compact;
  3. diagnosis (self-vote → unsure vs confident → agreement, with the lemma stated; proofs in the appendix);
  4. method;
  5. results: main table with CIs; comparison table (TDA, gate, read-first, TDA+T, ZAEC, gate+ZAEC, DMN-weak,
     ECALP); generalization over the dev and fresh sets with the "where it helps" ρ; cache ablation;
  6. limitations.

  About 3–4 figures in total: diagnosis (val), agreement split (val), accuracy–ECE plane or reliability (test), and
  where-it-helps. P1, earlier attempts, per-dataset tables and proofs go to the appendix.
- **Presentation: 20 slides at most**, professor-level. Suggested split:
  - title (1);
  - setting and the problem (2);
  - **TDA (4–5), using the TDA paper's Fig. 1 and/or Fig. 2** (Fig. 2 = TDA's pipeline with positive and negative
    caches; Fig. 1 = accuracy vs efficiency), cited, from the arXiv source of 2403.18293;
  - calibration and TDA's overconfidence (1–2);
  - diagnosis (3–4);
  - method (1);
  - results (3–4);
  - limitations and conclusion (1).

  Backup slides after the 20 are fine (P1, proofs, per-dataset).
