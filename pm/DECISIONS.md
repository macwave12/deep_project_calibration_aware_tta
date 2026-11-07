# Decision log

| Date | Decision | Why | By |
|---|---|---|---|
| 2026-11-04 | Keep TDA's per-dataset α/β for comparability (P1) | They match the official configs and the benchmark exactly | macwave (audit) |
| 2026-11-04 | New methods (reservoir cache, self-vote removal) deferred until wrap-up is done | Finish and present the existing state first | Idan |
| 2026-11-07 | Report two protocols: P1 published, P2 clean (α = 2, β = 5 everywhere) | Per-dataset tuning uses test information and undermines the zero-shot claim | Idan |
| 2026-11-07 | Headline protocol = P2 if P2-TDA beats zero-shot CLIP, else P1; both always shown | Prefer the truly zero-shot result when it is not degenerate | Idan |
| 2026-11-07 | Project goal stated as zero-shot CLIP classification; compare to ≤2 more zero-shot methods (run) and 1 few-shot method | Answer "why only TDA?" without scope creep | Idan |
| 2026-11-07 | CPU-only; one GPU job (feature extraction) by macwave | Idan has no GPU; frozen encoder + single view makes features reusable | Idan, lead |
| 2026-11-07 | Zero-shot baselines to run: ECALP (ICLR'25), DMN-weak (CVPR'24). Few-shot: Tip-Adapter (ECCV'22). Calibration-specific episodic methods (C-TPT, O-TPT, D-TPT) cited from their papers, not run | ECALP = most accurate/worst calibrated online method; DMN = widely used memory method; both single-view and in repo. Tip-Adapter is the few-shot origin of TDA's cache. TPT-family needs 64 views + backprop, infeasible on CPU | lead (proposed) |
| 2026-11-07 | Reproduce TDA paper's ablations (Fig. 3, 4, 5, Tables 5, 6) with ECE added | Show how TDA's own design choices trade accuracy against calibration | Idan |
| 2026-11-07 | PM folder `pm/` tracked in git is the source of truth for tasks and handoffs | Continuity across machines and agents | Idan |
| 2026-11-07 | T01/T02 review: no blocking defects; RNG state at loop start verified identical to `online_tta.main()`. Before the GPU job: torch-version guard, split preflight, exact per-sample order check against the original `save_results.pt`, resume fixes, deterministic autocast emulation | One-shot GPU job must be robust and self-verifying | lead (reviewer findings) |
| 2026-11-07 | GPU job also extracts 10 augmented views per train16 image (Tip-Adapter's official augmentation) and val permutations for seeds 0/1/2 | Faithful few-shot baseline without a second GPU session | lead |
