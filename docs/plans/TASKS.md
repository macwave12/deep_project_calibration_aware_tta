# Project task list — calibration-aware TTA

Last updated: 2026-11-04. Base: `publish` @ 4def98a, plus the audit branch
`audit/tda-baseline-hyperparameters` (ecee9fe). Companion plan:
`docs/plans/2026-11-04-reservoir-cache-label-free-temperature.md`.

Rules for everything below:
- **No selection on test.** Anything that chooses between variants, thresholds or
  capacities is decided on the CoOp **val** split (`--dataset_mode val`); test is run once,
  for the final numbers. Earlier selections made on test are disclosed in the report.
- **Label-free method.** Labels are read only by the scorer (`utils/evaluation.py`), for
  metrics and diagnostics.
- **≥3 seeds** for every number that enters a final table (stream order is random).

Status: `[ ]` open · `[~]` in progress · `[x]` done.

---

## 0. Resuming on another machine

- [ ] Clone the repo, check out `publish`, apply this branch's patches (`git am`), push
      `plan/reservoir-cache-temperature`.
- [ ] Push access for Claude from the cloud: macwave installs the Claude GitHub App on the
      `macwave12` account for this repo (github.com/apps/claude/installations/select_target).
      Not needed for local Claude Code / desktop work.
- [ ] Locally, point Claude at this file: "read `docs/plans/TASKS.md` and the plan next to it".

## 1. Walkthrough — Idan getting up to date (no code changes)

- [x] S1 Methods and measures (CLIP, Tip-Adapter, TDA, TPT/C-TPT/O-TPT, ECE, temperature).
- [x] S2 Proposal vs results; per-contribution mean-confidence gaps.
- [x] Phase 3/4 review: exploration variants, `main_v2.pdf` critique.
- [ ] S3 Code trace: one image through `CalTDA.adaptation_process`; `utils/loo_temperature.py`;
      `utils/calibration.py`; `tests/test_no_label_leak.py`, `tests/test_cal_tda_runner.py`.
- [ ] S4 Latest material: the audit branch (done reading) and the 22-slide deck (4def98a).
- [ ] S5/S6 Agree the final change list with macwave (sections 3–5 below).

## 2. Report and README fixes (no new runs)

- [ ] Restore the Limitations subsection (removed in 851f8bf): one seed, one backbone,
      gate chosen among four variants **evaluated on the test sets** (Phase 3 exploration),
      EuroSAT accuracy cost, LOO estimate quality.
- [ ] Disclose model selection: list the four exploration variants and their numbers
      (`results/raw/exploration/`); note `fusion_gate_sharpen` was never run on EuroSAT.
- [ ] Qualify "gated fusion is better calibrated than CLIP": true for the mean (4.61 vs
      5.57), but on 3/5 datasets only (worse on Flowers 4.08 vs 2.63, Pets 5.68 vs 4.37).
- [ ] Benchmark citation: wherever 5.70 → 9.21 is quoted, say "fine-grained average,
      CLIP-ResNet-50 (Table 7)"; the full average is 6.07 → 11.53. Our ECE is ViT-B/16, which
      the benchmark does not report.
- [ ] Add the audit's two sentences (per-dataset α/β provenance; fairness caveat).
- [ ] Describe TDA precisely: every image is offered to the cache; 3 lowest-entropy items
      per class are kept; the image is written **before** the cache is read.
- [ ] Rewrite the proposal's diagnosis: no feedback loop exists (pseudo-labels and entropy
      come from raw CLIP); the inflation is a self-vote (+α on CLIP's own argmax for
      admitted images) plus a near-uniform confidence shift (TDA ECE ≈ |mean conf − acc|).
- [ ] README Results section: still describes the original three-contribution bundle.
- [ ] Related work: see section 5.

## 3. Diagnostics (cheap; most need no GPU)

From saved outputs (`online_results/**/save_results.pt`, local only; `tda` saves logits,
`cal_tda` saves probabilities) or a counter added to the run record:

- [ ] Mean-confidence − accuracy gap for every run, plus sign, in the results tables
      (already computable from the `reliability` field of each JSON).
- [ ] NLL, Brier score, and risk–coverage / AURC for every headline run.
- [ ] **Which uncertainty score best predicts CLIP's correctness?** AUROC of entropy, max
      probability, probability margin and logit margin for "CLIP is right", per dataset.
      Answers whether anything should replace entropy for cache admission at all.
- [ ] **TDA reliability on the val split** (stream the CoOp val set through `tda`), to get a
      confidence–accuracy curve that is not the test set.
- [ ] Oracle temperature: one T per dataset fitted by NLL on TDA's logits with the true
      labels, reported only as a ceiling. Also fixed-T vs running-T.
- [ ] Counters per run: gate firing rate; self-vote fraction (admitted images); number of
      classes with ≥1 cache entry; `â` and running true accuracy at each step.

## 4. Experiments (GPU; select on val, final on test, ≥3 seeds)

E1 — Self-vote
- [ ] `read_before_write` flag; TDA with and without self-vote. Tests the corrected
      diagnosis directly.

E2 — Fusion rules (combine stage)
- [ ] Two-hypothesis mixture: `p = 0.5·softmax(z) + 0.5·softmax(z + α·c)` (CLIP vs TDA
      posterior). Peaked on both sides, so no flat-cache dilution.
- [ ] Uncertainty-conditioned weight: `p = (1−w)·softmax(z) + w·p_cache`, with
      `w = g(H(z)/log C)` increasing with CLIP's entropy (soft version of the gate that,
      unlike the gate, also uses the cache on diffuse samples).
- [ ] Kernel-normalized cache posterior instead of `softmax(c)`:
      `p_cache(k) = Σ_{i∈k} w_i / Σ_i w_i` (Nadaraya–Watson); removes the flatness problem
      without an ad-hoc rescale.
- [ ] Finish `fusion_gate_sharpen` on EuroSAT; keep `fusion_sharpen` as a candidate (only
      variant above TDA's accuracy anywhere: Flowers 70.93 vs 69.91).

E3 — Sampled cache + label-free temperature (the plan file)
- [ ] Weighted reservoir cache (A-Res, w = CLIP top-1 prob), vote normalized by K.
- [ ] Inverse-probability-weighted k-NN accuracy estimate; allow T < 1.
- [ ] Track estimator error `|â − acc|` as the primary diagnostic.

E4 — Capacity study (TDA paper Fig. 4, for our method)
- [ ] Positive and negative shot capacity sweeps on the chosen method, reporting accuracy,
      ECE **and** estimator error. Run on val (selection), report test as sensitivity.

E5 — Negative cache
- [ ] `use_neg_cache=False` ablation (does TDA's one damping term matter for ECE?).

E6 — More baselines on ViT-B/16 ECE
- [ ] Run the online methods already in `online_method/` (DMN, DPE, BoostAdapter, ECALP,
      OnZeta) through our scorer. The benchmark reports ECE only for ResNet-50, so a
      ViT-B/16 ECE table for these methods is new information in itself.

E7 — Seeds
- [ ] Re-run final configs with seeds 0/1/2; mean ± std; bootstrap CIs on ECE.

## 5. Related-work audit (find anyone who tried our ideas)

Verify each (venue, claim, whether it is calibration-aware, needs backprop, online):
- [ ] Prompt-tuning TTA: TPT (NeurIPS'22), DiffTPT (ICCV'23), C-TPT (ICLR'24),
      O-TPT (CVPR'25), DynaPrompt.
- [ ] Cache/memory TTA: Tip-Adapter (ECCV'22), TDA (CVPR'24), DMN (CVPR'24),
      DPE (NeurIPS'24), BoostAdapter (NeurIPS'24), ECALP (ICLR'25), OnZeta.
- [ ] CLIP calibration without TTA: calibration of fine-tuned / adapted CLIP (e.g.
      open-vocabulary calibration, adapter calibration work, 2024).
- [ ] Label-free accuracy estimation: ATC (ICLR'22), difference-of-confidence,
      agreement-on-the-line — prior art for our `â`.
- [ ] Search specifically for: test-time temperature scaling without labels; probability-
      space fusion of CLIP and cache; entropy-gated / confidence-gated cache fusion;
      reservoir or diversity-aware TTA caches.
- [ ] Rewrite the novelty claim to what survives: label-free calibration of an online,
      backprop-free, cache-based TTA method.
