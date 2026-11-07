# T08 — Baselines: ECALP, DMN-weak (zero-shot), Tip-Adapter (few-shot)

Status: open · Owner: engineering agent · Depends on: T02 (+ features)

- ECALP (ICLR'25) and DMN-weak (CVPR'24): replay from features with the benchmark's
  hyperparameters; accuracy, ECE, mean-confidence gap; P1 only (their own published
  settings), noted as such.
- Tip-Adapter (ECCV'22), 16-shot, training-free variant: implement on cached `train16`
  features; α, β chosen on val per its paper's protocol. Shown as the few-shot reference
  ("what labels buy"), not a competitor.
- Cited, not run (64 views + backprop): TPT, C-TPT, O-TPT, D-TPT ViT-B/16 ECE on our five
  datasets from their papers, with protocol differences stated.
Acceptance: numbers in `results/baselines.csv`; replay sanity check against the benchmark's
reported ViT-B/16 accuracies where available.
