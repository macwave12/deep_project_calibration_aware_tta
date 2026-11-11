# T07 — Protocols P1 (published) and P2 (clean)

Status: review (2026-11-10, lead) · Owner: analysis agent · Depends on: T02 (+ features)

Run clipzs, tda, cal_tda {gate, temp, gate+temp, margin, fusion} under P1 and P2, seeds
0/1/2. Produce `results/protocols.csv` (mean ± std). Apply the decision rule in
`DECISIONS.md` (P2 headline iff P2-TDA beats CLIP accuracy) and record the outcome there.

## Outcome (2026-11-10)
`analysis/protocols.py` → `results/protocols.csv`, `results/analysis/protocols_paired.csv`.
Rule: P2-TDA − CLIP = +3.66 acc (dataset mean, 3/3 seeds) → headline P2 (DECISIONS, K36–K37).
Handoff: `pm/handoffs/2026-11-10-T07-protocols.md`.
