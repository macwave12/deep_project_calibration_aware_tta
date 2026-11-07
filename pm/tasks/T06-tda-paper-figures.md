# T06 — TDA paper ablations, redone with calibration

Status: open · Owner: analysis agent · Depends on: T02 (+ features)

See `pm/research/tda-paper-figures.md` for the mapping. For each, run on our five datasets
(test split, labelled as sensitivity; selection-type conclusions only from val), report
accuracy **and** ECE (and mean-confidence gap), under P1 and P2:

- Fig. 3: CLIP vs positive-only vs negative-only vs TDA.
- Fig. 4: positive and negative shot capacity 1–6.
- Fig. 5: negative mask threshold p_l.
- Table 5: entropy band [τ_l, τ_h].
- Table 6: α × β grid (heatmaps per dataset; mark P1 and P2 points).
- Fig. 1 / Table 2: per-image time of each method on the same CPU (relative overhead only).
- Table 3 analogue: our five datasets, CLIP / TDA / ours, accuracy + ECE.

Acceptance: one script regenerates all; CSVs tracked; figures legible at slide size.
