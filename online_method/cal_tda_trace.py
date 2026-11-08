"""Opt-in, label-free per-sample trace for `CalTDA` (T04).

Records, for every stream step, what the method saw and decided, so analysis
code can explain *why* each prediction came out as it did. Everything here is
computed from quantities `CalTDA.adaptation_process` already has; nothing reads,
receives or stores a ground-truth label. Labels are joined afterwards, in
`analysis/trace_join.py`, from the saved file.

Recording never feeds back into the method: the recorder only reads tensors
(`topk`, indexing, `.cpu()` copies) and keeps its own bookkeeping, so outputs
with the trace on equal outputs with it off (`tests/test_cal_tda_trace.py`).

File format (`torch.save` of a dict, written next to the run record as
`<dataset>_<method>.trace.pt`):

    version          int, TRACE_VERSION
    topk             int, k used for the top-k fields (min(5, C))
    n_steps          int
    n_classes        int
    snapshot_every   int
    steps            dict of column tensors, one row per stream step (see STEP_FIELDS)
    snapshots        list of {'step', 'pos', 'neg'}; 'pos'/'neg' each a dict of
                     'class' (M,) int64 cache key (the method's pseudo-label),
                     'entropy' (M,) float32, 'feature' (M, D) in the dtype the
                     model returned (fp16 on GPU; kept exact for matching),
                     'inserted_step' (M,) int64 stream step that inserted the item.

Snapshots are taken after the step's cache write and read, every
`snapshot_every` steps (steps 499, 999, ...) and once at the end.
"""

from __future__ import annotations

import torch

TRACE_VERSION = 1
DEFAULT_SNAPSHOT_EVERY = 500
DEFAULT_TOPK = 5

# name -> (dtype, per-row shape: () scalar or ('k',) top-k vector)
STEP_FIELDS = {
    'clip_topk_logit': (torch.float32, ('k',)),   # raw CLIP logits, top-k values
    'clip_topk_index': (torch.int32, ('k',)),     # ... and their class indices
    'pred0': (torch.int32, ()),                   # CLIP pseudo-label (argmax)
    'entropy': (torch.float32, ()),               # TDA `Entropy` of softmax(CLIP), nats
    'admitted': (torch.bool, ()),                 # offered to the positive cache (margin gate passed / off)
    'in_pos_cache': (torch.bool, ()),             # self present in pos cache at read time (self-vote)
    'neg_band': (torch.bool, ()),                 # normalized entropy in (0.2, 0.5): offered to neg cache
    'in_neg_cache': (torch.bool, ()),             # self present in neg cache at read time
    'has_pos_cache': (torch.bool, ()),            # positive cache non-empty at read time
    'cache_topk': (torch.float32, ('k',)),        # unscaled cache logits c at CLIP's top-k classes
    'cache_argmax': (torch.int32, ()),            # argmax of c (-1 when the cache is empty)
    'cache_max': (torch.float32, ()),             # max of c (0 when the cache is empty)
    'neg_topk': (torch.float32, ('k',)),          # negative-cache logits subtracted at CLIP's top-k
    'gate_fired': (torch.int8, ()),               # 1/0 when the fusion gate is active, -1 when not applicable
    'fusion_fallback': (torch.bool, ()),          # fusion configured but cache empty -> additive branch
    'a_hat': (torch.float32, ()),                 # LOO accuracy estimate (nan when not estimated)
    'temperature': (torch.float32, ()),           # temperature applied to this step
    'out_topk_prob': (torch.float32, ('k',)),     # final probabilities, top-k values
    'out_topk_index': (torch.int32, ('k',)),      # ... and their class indices
    'out_argmax': (torch.int32, ()),              # final prediction
}


class TraceRecorder:
    """Accumulates per-step rows and periodic cache snapshots for one run."""

    def __init__(self, snapshot_every: int = DEFAULT_SNAPSHOT_EVERY, topk: int = DEFAULT_TOPK):
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be >= 1")
        self.snapshot_every = snapshot_every
        self.topk = topk
        self.n_classes = None
        self.step = 0
        self._rows = {name: [] for name in STEP_FIELDS}
        self.snapshots = []
        # Per cache, per class key: {id(feature tensor): inserting step} for the
        # items currently held. Cached tensors stay alive while cached, so their
        # ids are unique among live items.
        self._inserted = {'pos': {}, 'neg': {}}

    # -- bookkeeping called right after each cache write -------------------
    def track_write(self, which: str, cache: dict, key: int, feature: torch.Tensor) -> bool:
        """Refresh insertion steps for `cache[key]`; return whether `feature`
        (this step's image) is now held there."""
        previous = self._inserted[which].get(key, {})
        current, present = {}, False
        for item in cache.get(key, []):
            ident = id(item[0])
            if item[0] is feature:
                present = True
                current[ident] = previous.get(ident, self.step)
            elif ident in previous:
                current[ident] = previous[ident]
            else:  # pragma: no cover - would mean the cache changed behind our back
                raise RuntimeError("cache item of unknown origin; trace bookkeeping out of sync")
        self._inserted[which][key] = current
        return present

    # -- one row per step ---------------------------------------------------
    def record_step(
        self,
        *,
        clip_logits,
        pred0,
        entropy,
        admitted,
        in_pos_cache,
        neg_band,
        in_neg_cache,
        has_pos_cache,
        cache_logits,
        neg_logits,
        gate_fired,
        fusion_fallback,
        a_hat,
        temperature,
        probs,
        pos_cache,
        neg_cache,
    ):
        with torch.no_grad():
            clip = clip_logits.detach().reshape(-1).float().cpu()
            n_classes = clip.numel()
            if self.n_classes is None:
                self.n_classes = n_classes
            k = min(self.topk, n_classes)
            top_values, top_index = clip.topk(k)
            cache = cache_logits.detach().reshape(-1).float().cpu()
            neg = (neg_logits.detach().reshape(-1).float().cpu()
                   if neg_logits is not None else torch.zeros(n_classes))
            out = probs.detach().reshape(-1).float().cpu()
            out_values, out_index = out.topk(k)

            row = {
                'clip_topk_logit': top_values,
                'clip_topk_index': top_index.to(torch.int32),
                'pred0': torch.tensor(int(pred0), dtype=torch.int32),
                'entropy': torch.tensor(float(entropy), dtype=torch.float32),
                'admitted': torch.tensor(bool(admitted)),
                'in_pos_cache': torch.tensor(bool(in_pos_cache)),
                'neg_band': torch.tensor(bool(neg_band)),
                'in_neg_cache': torch.tensor(bool(in_neg_cache)),
                'has_pos_cache': torch.tensor(bool(has_pos_cache)),
                'cache_topk': cache[top_index],
                'cache_argmax': torch.tensor(int(cache.argmax()) if has_pos_cache else -1, dtype=torch.int32),
                'cache_max': torch.tensor(float(cache.max()) if has_pos_cache else 0.0, dtype=torch.float32),
                'neg_topk': neg[top_index],
                'gate_fired': torch.tensor(-1 if gate_fired is None else int(bool(gate_fired)), dtype=torch.int8),
                'fusion_fallback': torch.tensor(bool(fusion_fallback)),
                'a_hat': torch.tensor(float('nan') if a_hat is None else float(a_hat), dtype=torch.float32),
                'temperature': torch.tensor(float(temperature), dtype=torch.float32),
                'out_topk_prob': out_values,
                'out_topk_index': out_index.to(torch.int32),
                'out_argmax': torch.tensor(int(out.argmax()), dtype=torch.int32),
            }
            for name, value in row.items():
                self._rows[name].append(value)

        if (self.step + 1) % self.snapshot_every == 0:
            self.snapshot(pos_cache, neg_cache)
        self.step += 1

    # -- cache snapshots ----------------------------------------------------
    def _snapshot_cache(self, which: str, cache: dict) -> dict:
        classes, entropies, features, inserted = [], [], [], []
        for key in sorted(cache.keys()):
            steps = self._inserted[which].get(key, {})
            for item in cache[key]:
                classes.append(int(key))
                entropies.append(float(item[1]))
                features.append(item[0].detach().reshape(1, -1).cpu().clone())
                inserted.append(steps.get(id(item[0]), -1))
        if features:
            feature = torch.cat(features, dim=0)
        else:
            feature = torch.zeros(0, 0)
        return {
            'class': torch.tensor(classes, dtype=torch.int64),
            'entropy': torch.tensor(entropies, dtype=torch.float32),
            'feature': feature,
            'inserted_step': torch.tensor(inserted, dtype=torch.int64),
        }

    def snapshot(self, pos_cache: dict, neg_cache: dict) -> None:
        """Snapshot both caches as they stand after the current step."""
        self.snapshots.append({
            'step': self.step,
            'pos': self._snapshot_cache('pos', pos_cache),
            'neg': self._snapshot_cache('neg', neg_cache),
        })

    # -- output ---------------------------------------------------------------
    def to_dict(self, pos_cache: dict, neg_cache: dict) -> dict:
        """The trace as a plain dict, with a final snapshot if the last step
        did not already end on one."""
        last_step = self.step - 1
        if self.step > 0 and (not self.snapshots or self.snapshots[-1]['step'] != last_step):
            # `snapshot` stamps `self.step`, which has already advanced.
            self.step -= 1
            self.snapshot(pos_cache, neg_cache)
            self.step += 1
        steps = {}
        for name, (dtype, shape) in STEP_FIELDS.items():
            values = self._rows[name]
            if values:
                steps[name] = torch.stack(values).to(dtype)
            else:
                k = 0 if self.n_classes is None else min(self.topk, self.n_classes)
                steps[name] = torch.zeros((0, k) if shape else (0,), dtype=dtype)
        return {
            'version': TRACE_VERSION,
            'topk': self.topk if self.n_classes is None else min(self.topk, self.n_classes),
            'n_steps': self.step,
            'n_classes': self.n_classes,
            'snapshot_every': self.snapshot_every,
            'steps': steps,
            'snapshots': list(self.snapshots),
        }

