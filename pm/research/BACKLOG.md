# Research ideas — index and status

Every idea raised during the walkthrough (2026-11-03 → 11-10), where it is written up, and
whether it is part of the wrap-up or parked for after it (decision 2026-11-04).

## Folded into the wrap-up (deck v2 evidence)
| Idea | Now part of | Notes |
|---|---|---|
| Global α = 2, β = 5 protocol ("is the zero-shot claim honest?") | T07 (P1 vs P2) | was backlog E0 |
| Negative-cache ablation | T06 (TDA Fig. 3 with ECE) | was E5 |
| Shot-capacity study (TDA Fig. 4 for our method) | T06 | was E4 |
| Oracle temperature ceiling; fixed vs running T | T05 A9 | |
| NLL, Brier, risk–coverage / AURC | T05 A10 | |
| Self-vote share, cache purity, "poisoned" cached image | T05 A3, A4 | diagnosis, not a fix |
| More baselines incl. ViT-B/16 ECE for other online methods | T08 | was E6 |
| Seeds and paired statistics | DECISIONS 2026-11-10 | was E7 |

## Parked — to develop after the wrap-up
| # | Idea | Origin | Written up in |
|---|---|---|---|
| R1 | **No self-vote**: read the caches before writing the current image (both caches) | proposal review | `docs/plans/TASKS.md` E1 |
| R2 | **Confidence-weighted reservoir cache** (A-Res), capacity sweep, vote normalized by K | Idan | `docs/plans/2026-11-04-reservoir-cache-label-free-temperature.md` |
| R3 | **Unbiased accuracy estimate for the temperature**: inverse-probability-weighted k-NN agreement; allow T < 1 | Idan + lead | same plan doc |
| R4 | **Per-branch temperatures** (tie rows vs non-tie rows of the gate) | checkpoint discussion | this file (new) |
| R5 | **Two-hypothesis mixture** `0.5·softmax(z) + 0.5·softmax(z + αc)` | Idan | `docs/plans/TASKS.md` E2 |
| R6 | **Uncertainty-conditioned fusion weight** w = g(H(z)) (soft gate that also uses the cache on diffuse samples) | Idan | `docs/plans/TASKS.md` E2 |
| R7 | **Kernel-normalized cache posterior** (Nadaraya–Watson) instead of softmax(c) | lead | `docs/plans/TASKS.md` E2 |
| R8 | **Revisit `fusion_sharpen`** (only variant above TDA's accuracy anywhere) | Phase 3 review | `docs/plans/TASKS.md` E2 |
| R9 | **Soft-label tie-breaking cache**: soft labels, strata by significant classes, Dirichlet-mean combination, confusion-graph coverage | Idan | `pm/research/cache-idea-formalization.md` |
| R10 | **LOO-consistency eviction** of suspected mislabeled cache items | lead | plan doc §9 backlog |
| R11 | **Which uncertainty score should gate the cache?** AUROC of entropy / max-prob / margin for CLIP correctness | Idan's margin-vs-entropy question | `docs/plans/TASKS.md` §3 |
| R12 | Few-shot validation anchors / calibration on train images | Idan | discussed; **changes the setting to few-shot** — only if the project is reframed |

Suggested order after the wrap-up: R1 (one flag, tests the corrected diagnosis) → R2+R3
(they fix each other) → R9 (Idan's idea, builds on the gate) → R5–R8 as comparisons.
