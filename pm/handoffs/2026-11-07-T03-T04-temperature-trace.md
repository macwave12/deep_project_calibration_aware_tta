# 2026-11-07 — T03 O(N) temperature + T04 label-free trace (engineering agent)

Branch `worktree-agent-a123b17e7cf12612a`, on top of `deck/refinement` @ 77303a5.
(The worktree had been created at the initial commit 3c79fc0, with no `pm/`; I moved
it to `deck/refinement` with `git reset --hard deck/refinement` before doing any work.
It had no commits of its own. The other agent's worktree, `agent-ace5471a3fbddc65c`,
was also at 3c79fc0 when I looked. Worth checking that its base is right too.)

## T03 — what was done
- `utils/loo_temperature.py`: new `RunningConfidence`. It keeps one float64 running sum
  of `max softmax(s/T)` per grid temperature. `update` is O(|grid|·C) per sample and
  `search` is O(|grid|). Same contract as `search_temperature` (NaN/None → 1.0, ties → first
  grid point). `search_temperature` is unchanged and kept as the reference.
- `online_method/cal_tda.py`: `_score_history` (list) replaced by `self._confidence`. Two
  small hooks were added: `_record_scores` (updates the sums, and only when
  `use_loo_temperature`, because nothing else reads them) and `_search_temperature`. The
  LOO estimate (O(M²), M ≤ 3C) is unchanged.
- Exactness argument (also in the docstring): per-sample terms are bit-identical to the
  reference, since it uses the same float32 division and softmax (verified elementwise).
  A max-softmax value lies in [1/C, 1], so float64 sums of these float32 terms are exact
  while 24 + log2 C + log2 N ≤ 53. The reference reduces in float32 (error ~1e-7), so a
  *pure* exact argmin can still round a near-tie the other way from the reference. **This
  really happened:** the first 8,100-step benchmark (pure mode) differed at 1 step
  (step 7153: T = 11.00 vs the reference's 10.95; the two best gaps differed by 1.2e-10).
- Fix (now the default, `RunningConfidence(reference_tie_tol=1e-5)`): when more than one
  candidate's exact gap is within 1e-5 of the minimum, only those candidates are re-scored
  with the reference's own `mean_confidence` on the full history, and the reference's
  first-minimum rule picks among them. No other candidate can win in the reference unless
  its float32 error exceeds 5e-6 (expected error < 1e-6 at N < 65k). Cost: the history is
  kept (O(N·C) memory, as before), and a rescoring touches 2–3 temperatures. On the
  8,100-step stream it triggered on 237 / 8,100 steps (2.9%). `reference_tie_tol=None`
  gives the pure O(1)-memory mode.

## T04 — what was done
- `online_method/cal_tda_trace.py` (new): `TraceRecorder`. It is read-only with respect to
  the method: it only does `topk`/indexing/`.cpu()` copies plus its own bookkeeping.
- `CalTdaConfig.trace: bool = False`. While off, it is dropped from `as_record_dict`, so
  run records are unchanged. When on, the record carries `trace: true`. It is not
  listed under `inert`, because it is instrumentation, not a knob.
- `CalTDA`: `self.trace` (recorder or None), `trace_dict()`, `save_trace(path)`,
  `_record_trace_step`. â is kept in `self._last_estimate`.
- `online_tta.py`: `--trace` CLI flag. After `score_and_save`, it writes
  `<out_dir>/<dataset>_<method>.trace.pt` next to the JSON record.
- `analysis/trace_join.py` (new): `FeatureIndex` (exact byte match, dtype must match:
  fp16 on the GPU path), `match_indices` (duplicates resolved via the permutation, else
  `AMBIGUOUS`), `join_snapshots`, `snapshot_summary` (purity, classes_present,
  coverage_correct), and `join_steps` (labels via `perm`). CLI: `python3 -m analysis.trace_join`.

### Trace file format (`torch.save` dict, version 1; loads with `weights_only=True`)
- `version, topk (=min(5,C)), n_steps, n_classes, snapshot_every (500), config (as_dict)`
- `steps`: column tensors, one row per stream step:
  `clip_topk_logit f32[k]`, `clip_topk_index i32[k]`, `pred0 i32`, `entropy f32` (TDA's
  `Entropy`, nats), `admitted bool` (offered to the positive cache), `in_pos_cache bool`
  (self present in its own class's positive cache at read time = self-vote), `neg_band bool`,
  `in_neg_cache bool`, `has_pos_cache bool`, `cache_topk f32[k]` (unscaled cache logits c at
  CLIP's top-k; additive TDA uses α·c), `cache_argmax i32` (-1 if the cache is empty),
  `cache_max f32`, `neg_topk f32[k]` (the negative-cache logits that are subtracted),
  `gate_fired i8` (1/0, or -1 when not applicable), `fusion_fallback bool`, `a_hat f32` (NaN
  when not estimated), `temperature f32`, `out_topk_prob f32[k]`, `out_topk_index i32[k]`,
  `out_argmax i32`.
- `snapshots`: list of `{step, pos, neg}`. Each cache is `{class i64[M], entropy f32[M],
  feature [M,D] in the model's dtype, inserted_step i64[M]}`, taken after steps 499, 999, …
  and at the end. `inserted_step` is label-free: it is the stream position that inserted
  the item, tracked by tensor identity.
- Size (measured, C = 10, D = 512): per-step rows are **155 B/sample**. Each snapshot is
  ~103 KB with fp32 features (≤ 3C positive + ≤ 2C negative items × D × 4 B), or ~half that
  with the fp16 features the GPU path produces. A 1,000-step trace file was 372 KB
  (2 snapshots). Estimate for 8,100 steps: ≈ 1.3 MB rows + 17 snapshots ≈ 2–3 MB at C = 10.
  At C = 100 (fp16, D = 512), expect ≈ 0.5 MB per snapshot, so ≈ 10 MB per run.

## How it was verified
- `tests/test_loo_temperature_streaming.py`: property tests on random histories (N ∈ {1..300},
  C ∈ {2,10,100}, logit scale 0.1–30, random/0/1/1/C targets). Results are exact-equal to
  `search_temperature`. Targets placed on the reference's own grid-point confidences are also
  exact. Midpoint targets (built to be adversarial) agree except where the two candidates'
  reference gaps differ by < 1e-6, and the test asserts that bound. Also covers flat-history
  ties, custom grid, and NaN fallback.
- `tests/test_cal_tda_temperature_stream.py`: CalTDA over a 120-sample, 10-class stream for
  `cal_tda_temp`, `cal_tda`, `cal_tda_gate_temp`. Temperatures, outputs (`torch.equal`) and
  `temperature_summary` are identical to a subclass that runs the old O(N²) search.
- `tests/test_cal_tda_trace.py::test_trace_off_is_bit_identical_to_the_pre_change_code`: the
  current CalTDA (trace off) against `online_method/cal_tda.py` **as of 77303a5, loaded from
  git**, on all 22 configs under `configs/method/` (60-sample stream). Every output is
  bit-identical, and so are the temperatures and the admission/temperature summaries. This
  covers T03 and T04 together. The test skips if git or the commit is unavailable.
- Trace on vs off: identical outputs and temperatures (4 configs). Trace rows are checked
  against the outputs, CLIP logits, summaries and snapshots (features equal
  `features[inserted_step]`, keys equal `pred0[inserted_step]`). All flags off + trace still
  matches upstream `TDA` (argmax equal, atol 5e-3). Save/load round trip works. The CLI flag
  is wired.
- `tests/test_trace_join.py`: exact match (one fp16 ULP off does not match), duplicates,
  dtype mismatch error, and a shuffled fp16 synthetic stream where join indices = perm[inserted_step]
  and purity equals a direct lookup.
- `tests/test_no_label_leak.py`: the trace module was added to `METHOD_FILES`. New checks:
  trace/CalTDA trace entry points take no label parameters, and method files import neither
  `analysis` nor `utils.evaluation`.
- Full suite: `python3 -m pytest -q` → **165 passed, 19 skipped** in 53 s (was 111/19). The 19 skips are the pre-existing CUDA-only tests.

## Measured speedup (synthetic, CPU, 8,100 samples, 10 classes, D=512, `cal_tda_temp`, eurosat α/β)
`scripts/bench_temperature_search.py --n 8100 --classes 10` (full CalTDA stream, both paths):

| path | wall time | per sample | load |
|---|---|---|---|
| O(N²) reference | 3,190 s | 394 ms | shared box, contended part of the time |
| O(N) streaming, pure mode | 109 s | 13.5 ms | same run, heavily contended (test suite running alongside) |
| O(N) streaming, final (tie re-scoring) | **7.45 s** | **0.92 ms** | quiet box, re-measured 2026-11-08 |

- Same-run ratio: **29×** (both contended).
- Quiet-box ratio: ≈ **330×** (3,190 / 7.45). This pairs a contended quadratic time with a
  quiet streaming time, so treat it as rough.
- The 800-sample quiet runs extrapolate (×(8100/800)²) to ≈ 2,450 s for the quadratic path,
  which gives ≈ 330× too.
- I did not re-run the 53-minute quadratic path on a quiet box, to spare the shared cores.

Equivalence on the 8,100-step stream (final mode, `near_tie_check`): every step whose
runner-up gap was within 1e-4 of the best (1,809 steps, a superset of every step where the
reference could disagree) was checked against the full reference search. **0 differences.**
The pure mode differed at exactly the one step above.

## Shared lines touched (for the merge with the device / protocol / replay work)
- `CalTDA.prepare_model_and_optimization`: in the reset block *below* the α/β write-back,
  `self._score_history = []` became `self._confidence = RunningConfidence()`, and two lines
  were added (`self._last_estimate = None`, `self.trace = ...`). The same change was made in
  `__init__`. The α/β lines themselves are untouched. A conflict there should be trivial:
  keep both sides.
- `CalTDA.adaptation_process`: the negative-cache subtraction was split into `neg_logits = ...;
  base_logits -= neg_logits` (value-identical). The band test is bound to `neg_band`, and
  there is a trace call before `return`. No device handling was touched. `online_method/tda.py`
  is not modified.
- `online_tta.py`: `load_method_config` (+3 lines), after the record write (+3 lines), and
  `--trace` arg.
- Tests run on CPU by patching `Tensor.cuda` to the identity (`tests/cpu_stream.py::cpu_only`)
  because `compute_cache_logits` still calls `.cuda()`. Once T02's device fix lands, the
  patch is a no-op.

## Open questions / caveats
- Exactness rests on the reference's float32 error staying below 5e-6. This was checked
  empirically, not proven for every torch build or CPU ISA. The pure exact-argmin mode alone
  is *not* identical (1 step in 8,100), which is why tie re-scoring is on by default.
- The trace's `clip_topk_logit` is taken from the autocast logits (fp16 on GPU). `entropy` is
  TDA's natural-log entropy with its 1e-5 epsilon. The negative band uses log2 normalisation,
  as upstream does.
- `in_pos_cache` is defined as "present at read time". Because writes precede reads (K8), this
  equals "inserted this step".
- Snapshots do not store the negative cache's probability maps; they can be recomputed from
  the features.
- GPU path not exercised (no CUDA here): the fp16 snapshot features and per-step `.cpu()`
  syncs (trace on only) are untested on a GPU.
- The 8,100-sample timing ran on a shared 2-core box while another agent was also running
  jobs, so the absolute times are inflated. The ratio is the meaningful number.
