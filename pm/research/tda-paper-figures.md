# TDA paper (CVPR'24) figures and tables → what we compute (T06)

| TDA item | What it shows there | Our version (5 datasets, ViT-B/16) | Why it matters for us |
|---|---|---|---|
| Fig. 1 / Table 2 | accuracy vs efficiency | per-image CPU time: CLIP, TDA, gate, temp (O(N) after T03) | gate costs one comparison; temperature cost was an implementation artifact |
| Table 3 | cross-domain accuracy | CLIP / TDA / ours: accuracy **and ECE**, P1 and P2 | headline table with calibration added |
| Fig. 3 | CLIP vs positive-only vs negative-only vs TDA (accuracy) | same + ECE + confidence gap | which cache causes the overconfidence (expected: positive) |
| Fig. 4 | shot capacity 1–6, positive and negative (accuracy) | same + ECE, for TDA and for the gate | capacity trades purity vs coverage; does calibration follow accuracy? |
| Fig. 5 | negative mask threshold p_l | same + ECE | does the damping term matter for calibration? |
| Table 5 | entropy band [τ_l, τ_h] | same + ECE | sensitivity of the negative cache |
| Table 6 | residual α × sharpness β (ImageNet) | α × β heatmaps per dataset, accuracy and ECE, P1/P2 points marked | the per-dataset α/β question: how much of TDA's gain and of its overconfidence comes from tuning |
| Table 1 | ImageNet OOD variants | not reproduced (different datasets) | — |

Test-split sweeps are sensitivity analyses and are labelled as such; any choice made from
them is made on val (RULES §1).
