# T04 — Label-free per-sample trace

Status: review · Owner: engineering agent · Depends on: —

## Goal
Opt-in trace (`trace: true`, default off) so analyses can explain *why* each prediction
came out as it did, without the method ever seeing a label.

## Per stream step
CLIP logits (top-5 values and indices) · pseudo-label `pred0` and entropy · admitted to
positive cache? · entered negative cache? · self-vote (image present in its own class's
cache at read time) · cache logits `c` for the top-5 CLIP classes and `c`'s argmax/max ·
negative-cache logits for the same classes · gate fired? · `â` and T · final top-5
probabilities · argmax.

## Cache snapshots
Every 500 steps and at the end: positive and negative cache contents as (class key, entropy,
feature). Analysis maps each feature back to its dataset index by exact match against the
replayed features, then joins labels (purity, coverage).

## Acceptance
- Trace off: outputs bit-identical to the current code (test).
- Trace on: outputs identical to trace off (test); trace file written next to the run record.
- `tests/test_no_label_leak.py` passes; no label reaches `online_method/`.
- Joiner in `analysis/` with a synthetic-data test.
