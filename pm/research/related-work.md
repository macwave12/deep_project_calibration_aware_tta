# Related work — landscape and prior-art audit (T09, first pass 2026-11-07)

Status: second pass 2026-11-07 (venues verified for D-TPT, TCA, BCA, CRG). Every row marked *verify* must be checked against the paper before it
goes on a slide.

## Landscape (for the method-landscape slide)

| Method | Venue | Setting | Online? | Backprop? | Calibration-aware? | Relation to us |
|---|---|---|---|---|---|---|
| CLIP zero-shot | ICML'21 | zero-shot | — | no | no | base model |
| Tip-Adapter | ECCV'22 | **few-shot** (labeled cache) | no | no (Tip-Adapter-F: yes) | no | origin of TDA's cache; our few-shot reference (T08) |
| TPT | NeurIPS'22 | zero-shot TTA | episodic | yes (prompt) | no (entropy-min → overconfident) | canonical TTA baseline |
| C-TPT | ICLR'24 | zero-shot TTA | episodic | yes | **yes** (text-feature dispersion) | closest goal; cite ECE |
| O-TPT | CVPR'25 | zero-shot TTA | episodic | yes | **yes** (orthogonality) | cite ECE |
| D-TPT | arXiv 2510.09473 (Oct 2025; no venue found) | zero-shot TTA | episodic | yes | **yes** (dimensional entropy) | best reported ViT-B/16 fine-grained ECE (4.18%) |
| TCA (attribute-aware test-time calibration) | arXiv 2506.22819 (Jun 2025; no venue found) | zero-shot TTA | episodic | yes | **yes** | cite |
| TDA | CVPR'24 | zero-shot TTA | **online** | **no** | no | our base |
| DMN (DMN-weak) | CVPR'24 | zero-/few-shot | online | no | no | run (T08) |
| DPE | NeurIPS'24 | zero-shot | online | yes (residuals) | no | cite |
| BoostAdapter | NeurIPS'24 | zero-shot | online | no | no | cite (benchmark ECE 17%) |
| ECALP | ICLR'25 | zero-/few-shot | online | no | no | run (T08); most accurate, worst calibrated |
| FreeTTA (online EM) | CVPR'25 | zero-shot | online | no | *verify* | cite |
| BCA (Bayesian TTA) | CVPR'25 | zero-shot | online (continual class-embedding and prior updates) | no | no (abstract claims accuracy/speed only) | Bayesian likelihood + prior updates; cite as probability-space relative |
| DOTA / OGA (Gaussian TTA) | arXiv'24 / '25 | zero-shot | online | no | *verify* | cite |
| UL-TTA | arXiv 2511.09101 (Nov 2025) | zero-shot | online | no | **yes**: decoupled prediction/calibration temperatures, reports 20–30% ECE reduction; does not compare to TDA | **prior art for label-free test-time temperature** — must be cited |
| Robust calibration of VLM adapters (ZS-Norm) | ECCV'24 | adapters | — | — | **yes**: logit-range normalization | prior art for our scale/range argument (`match_logit_range`) |
| Illusion of Progress (benchmark) | NeurIPS'25 | benchmark | — | — | measures ECE | motivation and codebase |

## Prior-art questions for our ideas

| Our idea | Closest found so far | Status |
|---|---|---|
| Probability-space fusion instead of logit addition | — | search |
| Gate: consult the cache only on two-way ties | — | search |
| Label-free temperature for online TTA | UL-TTA (2025): decoupled τ_cal | **found** — differentiate: ours uses the cache's LOO agreement as the accuracy target |
| Cache noise / wrong pseudo-labels | CRG, "Mitigating Cache Noise in TTA for Large VLMs" (ICLRW'25, arXiv 2503.18334): learnable residuals aligning cache prototypes to text + Gaussian discriminant analysis; accuracy only, no calibration; "Adaptive Cache Enhancement" (arXiv'25) still to read | **found** — our purity analysis is diagnostic (why calibration breaks), not a new denoiser; cite CRG |
| Logit-range mismatch between CLIP and adapters | ZS-Norm (ECCV'24) | **found** |

## Selection for the deck (see DECISIONS.md)
Run: ECALP, DMN-weak (zero-shot, single view, in repo, backprop-free); Tip-Adapter 16-shot
(few-shot). Cite: TPT, C-TPT, O-TPT, D-TPT ECE; BoostAdapter/DPE from the benchmark.

## Sources
- Awesome-LabelFree-VLMs, online TTA list: https://github.com/tim-learn/Awesome-LabelFree-VLMs/blob/main/assets/online_test_time_adaptation.md
- UL-TTA: https://arxiv.org/html/2511.09101
- D-TPT: https://arxiv.org/pdf/2510.09473v1
- TCA: https://arxiv.org/pdf/2506.22819
- C-TPT: https://arxiv.org/html/2403.14119v3
- Robust calibration of VLM adapters: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3461_ECCV_2024_paper.php
- Survey "Adapting VLMs Without Labels": https://arxiv.org/pdf/2508.05547
- BCA (CVPR'25): https://arxiv.org/abs/2503.09248v2
- CRG / cache noise (ICLRW'25): https://arxiv.org/abs/2503.18334v2
