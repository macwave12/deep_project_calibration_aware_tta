# Idan's review of the 22-slide deck (2026-11-07) — checklist for T10

1. S3: say both encoders are frozen (weights not updated during TTA), not only the image encoder.
2. S3: show the prompt "a photo of a {class}." and that it is a single template.
3. S4: failure is also fine-grained separation (caption granularity), not only domain shift.
4. S4: state the problem: known class set, unlabeled images, no labeled set.
5. Add a method-landscape slide building up to TDA.
6. S5: high-level TDA idea only (several image features describe the class region more
   finely than one text feature); remove the jargon example; weaknesses later.
7. Before S7: TDA math (logits, parameters); TDA walkthrough ≥ 25% of the deck.
8. S7: unclear; caveats only after the full walkthrough.
9. Before "accuracy is not the only…": show what TDA achieves over CLIP; then accuracy vs
   calibration + ECE; then CLIP vs TDA accuracy, confidence, ECE, reliability curves.
10. S9: explain why calibration matters and what the Illusion of Progress paper found.
11. Add a slide with our proposal and the list of methods.
12. S11: keep; add per-dataset accuracy and ECE for CLIP and TDA.
13. Show TDA's overconfidence concretely: the cache, confident correct and wrong images,
    a wrongly cached image pulling others (CLIP vs cache contributions).
14. Each change: motive + example of why it should work; otherwise consider dropping it.
15. S13: show wrongly pseudo-labeled cached images before/after margin admission.
16. S13: define "admitted" and "none".
17. S14: fairness — per-dataset α/β vs untuned w; protocols P1/P2; test-set separation.
18. S15: support the dilution claim with subgroup confidence/accuracy (CLIP-confident
    samples, reliability drift toward underconfidence) and the scale issue with a graph.
19. Explain the scale issue (softmax is not scale-invariant).
20. Support every assumption and conclusion with examples and graphs.
21. Explain why fusion is not a normalized product (it is TDA).
22. "CLIP has a clear winner": support with math or a CLIP-vs-cache score analysis.
23. Temperature: discuss cost and whether to keep it.
General: thorough background, other methods, evidence-backed conclusions.
