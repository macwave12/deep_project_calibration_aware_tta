# 2026-11-10 — T14 step 1: full-length trace vectors, trace through the replay (lead, local)

Branch `deck/refinement`, uncommitted. T14 status: in progress (step 1 done, review).

Also this session: patches `research-backlog`, `echo-analysis`, `t14-task` applied with
`git am` (df557f9, 7539b85, 20f50fb). T14's two decisions logged in DECISIONS. PLAN.md updated
with the Storyline section, the new scope and the revised phase order.

## What was done
- `online_method/cal_tda_trace.py`: `TraceRecorder(full_vectors=False)`. When on, `steps` also
  holds `FULL_FIELDS`, each (n_steps, n_classes) float32: `clip_logits` (z), `neg_logits` (n),
  `base_logits` (z − n exactly as computed, which keeps autocast rounding) and `cache_logits`
  (c, unscaled). The dict carries `full_vectors`. The top-k fields are unchanged, and
  `TRACE_VERSION` stays 1 (only keys were added).
- `CalTdaConfig.trace_full` (instrumentation, default off, implies a trace). It stays out of
  run records while off, like `trace`.
- `online_tta.py --trace-full`. `scripts/replay.py --trace {off,top,full}` →
  `replay_one(trace=...)`, which writes `<dataset>_<name>.trace.pt` next to the record. It is
  cal_tda only: TDA is traced through `configs/method/tda_equivalent.yaml`, and any other
  method raises.

## How it was verified
- New tests (`tests/test_cal_tda_trace.py`, `tests/test_feature_replay.py`):
  - full vectors on vs off give bit-identical outputs;
  - the top-k fields equal slices of the full vectors;
  - **offline re-evaluation**: a trace taken at α = 2 reproduces runs at α = 0, 0.7 and 3.5 bit
    for bit (`softmax(base + α·c)`), and c and z − n are identical across α. This is the T14
    premise (K7), now tested;
  - CLI flags and defaults; the replay writes the trace; the replay refuses non-cal_tda methods.
- Full suite: 322 passed / 14 skipped.
- Real data: DTD val, seed 0, P2, `tda_equivalent`. The offline accuracy from the trace equals
  the record's (0.4583), the argmax equals `out_argmax`, the file is 2.4 MB, and the run took
  1.9 s. Size scales as N × C × 16 bytes, so an ImageNet 10k subset would be about 160 MB per run.

## Notes for the next steps
- **α = 0 is not CLIP.** The negative cache stays subtracted: on DTD val, α = 0 gives 0.4495 vs
  CLIP 0.4326. A T14 rule that scales only the positive cache should state that its floor is
  "CLIP + negative cache". If the floor should be plain CLIP, the rule needs a second factor on n.
  Decide before step 3.
- The echo analysis (A11) ran under P1 (`protocol="published"` in `analysis/echo_analysis.py`),
  while T14 guardrail 2 fits under P2. A11 is diagnosis only, but its figure should be regenerated
  under P2 (or both) before the deck uses it.
- Order inconsistency in T14: the logged order starts with dev-set extraction, but step 7 makes
  extraction conditional on 6a/6b passing. PLAN.md resolves this as "extraction (Colab, Idan)
  in parallel with the val proof of concept; step 8 needs both". Idan to confirm.
- "R13" appears in the T14 decision and in PLAN, but `research/BACKLOG.md` lists only R1–R12.
  It needs an entry.
- Step 2 can start now: replay TDA-equivalent with `--trace full` on the five val splits, P2,
  seeds 0/1/2 (~1 min), then compute the signals from the trace files.
