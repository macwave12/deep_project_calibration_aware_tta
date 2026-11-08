"""Join a label-free CalTDA trace (T04) with dataset indices and labels.

The trace (`online_method/cal_tda_trace.py`) never contains a label. This is
the analysis side, where labels are allowed:

- per-step rows: stream step i was dataset index `perm[i]` (T01's
  `perm_<split>_seed<k>.pt`; identity when the stream ran in dataset order), so
  its label is `labels[perm[i]]`;
- cache snapshots: each cached feature is mapped back to its dataset index by
  **exact** (bitwise) match against the replayed features (T01's
  `<split>_feats.pt`, dataset order), then joined with labels to get cache
  purity and class coverage. When the stream permutation is known it is used
  to cross-check the match through the item's `inserted_step`, and to resolve
  bitwise-duplicate features (identical images).

    python3 -m analysis.trace_join --trace results/raw/dtd_cal_tda.trace.pt \
        --features features/dtd/test_feats.pt --labels features/dtd/labels_test.pt \
        --perm features/dtd/perm_test_seed0.pt --out results/analysis/dtd_cal_tda_cache.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

UNMATCHED = -1
AMBIGUOUS = -2


def load_trace(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _as_tensor(obj) -> torch.Tensor:
    if isinstance(obj, dict):  # tolerate {'features': tensor} style files
        for key in ("features", "feats", "image_features"):
            if key in obj:
                return torch.as_tensor(obj[key])
        raise KeyError(f"no feature tensor among keys {sorted(obj)}")
    return torch.as_tensor(obj)


def _row_keys(features: torch.Tensor) -> list[bytes]:
    """One hashable key per row: the row's exact bytes."""
    rows = features.detach().cpu().contiguous()
    rows = rows.reshape(rows.shape[0], -1)
    raw = rows.view(torch.uint8).numpy() if rows.numel() else np.zeros((rows.shape[0], 0), np.uint8)
    return [raw[i].tobytes() for i in range(raw.shape[0])]


class FeatureIndex:
    """Exact-match lookup from a feature row to its dataset index(es)."""

    def __init__(self, dataset_features):
        features = _as_tensor(dataset_features)
        if features.ndim != 2:
            features = features.reshape(features.shape[0], -1)
        self.dtype = features.dtype
        self.dim = features.shape[1]
        self._index: dict[bytes, list[int]] = {}
        for i, key in enumerate(_row_keys(features)):
            self._index.setdefault(key, []).append(i)

    def candidates(self, features: torch.Tensor) -> list[list[int]]:
        features = torch.as_tensor(features)
        if features.numel() == 0:
            return [[] for _ in range(features.shape[0])]
        features = features.reshape(features.shape[0], -1)
        if features.dtype != self.dtype or features.shape[1] != self.dim:
            raise ValueError(
                f"trace features are {features.dtype} x {features.shape[1]}, dataset features "
                f"{self.dtype} x {self.dim}; exact matching needs the dtype the model returned"
            )
        return [self._index.get(key, []) for key in _row_keys(features)]


def match_indices(candidates, inserted_step=None, perm=None) -> np.ndarray:
    """Resolve candidate lists to one dataset index per item.

    Unique match -> that index. Several (bitwise-duplicate features) -> the one
    the permutation names for the inserting step if available, else AMBIGUOUS.
    None -> UNMATCHED.
    """
    out = np.full(len(candidates), UNMATCHED, dtype=np.int64)
    for i, options in enumerate(candidates):
        if len(options) == 1:
            out[i] = options[0]
        elif len(options) > 1:
            out[i] = AMBIGUOUS
            if perm is not None and inserted_step is not None and inserted_step[i] >= 0:
                expected = int(perm[int(inserted_step[i])])
                if expected in options:
                    out[i] = expected
    return out


def join_snapshot_cache(cache: dict, index: FeatureIndex, labels, perm=None) -> pd.DataFrame:
    """One row per cached item: key, entropy, inserting step, dataset index, label."""
    labels = np.asarray(torch.as_tensor(labels)).reshape(-1)
    inserted = np.asarray(cache["inserted_step"]).reshape(-1)
    perm_np = None if perm is None else np.asarray(torch.as_tensor(perm)).reshape(-1)
    dataset_index = match_indices(index.candidates(cache["feature"]), inserted, perm_np)
    matched = dataset_index >= 0
    label = np.full(len(dataset_index), -1, dtype=np.int64)
    label[matched] = labels[dataset_index[matched]]
    frame = pd.DataFrame({
        "class": np.asarray(cache["class"]).reshape(-1),
        "entropy": np.asarray(cache["entropy"]).reshape(-1),
        "inserted_step": inserted,
        "dataset_index": dataset_index,
        "label": label,
    })
    frame["correct"] = matched & (frame["label"].to_numpy() == frame["class"].to_numpy())
    if perm_np is not None:
        ok = inserted >= 0
        expected = np.full(len(inserted), UNMATCHED, dtype=np.int64)
        expected[ok] = perm_np[inserted[ok]]
        frame["perm_consistent"] = matched & (expected == dataset_index)
    return frame


def join_snapshots(trace: dict, dataset_features, labels, perm=None) -> pd.DataFrame:
    """All snapshots of both caches as one long table (`cache` in {pos, neg})."""
    index = FeatureIndex(dataset_features)
    frames = []
    for snap in trace["snapshots"]:
        for which in ("pos", "neg"):
            frame = join_snapshot_cache(snap[which], index, labels, perm)
            frame.insert(0, "cache", which)
            frame.insert(0, "step", int(snap["step"]))
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def snapshot_summary(joined: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    """Per snapshot and cache: size, purity, classes present, correct coverage.

    purity            fraction of matched items whose label equals their key
    classes_present   distinct keys / C
    coverage_correct  classes with >= 1 correctly keyed item / C
    """
    rows = []
    for (step, which), group in joined.groupby(["step", "cache"], sort=True):
        matched = group[group["dataset_index"] >= 0]
        rows.append({
            "step": step,
            "cache": which,
            "n_items": len(group),
            "n_unmatched": int((group["dataset_index"] == UNMATCHED).sum()),
            "n_ambiguous": int((group["dataset_index"] == AMBIGUOUS).sum()),
            "purity": float(matched["correct"].mean()) if len(matched) else float("nan"),
            "classes_present": group["class"].nunique() / n_classes,
            "coverage_correct": matched.loc[matched["correct"], "class"].nunique() / n_classes,
        })
    return pd.DataFrame(rows)


def join_steps(trace: dict, labels, perm=None) -> pd.DataFrame:
    """Per-step trace as a table, with the step's dataset index and label.

    Top-k vector fields become columns `<field>_<j>`. Adds `clip_correct`
    (pred0 == label) and `correct` (out_argmax == label).
    """
    steps = trace["steps"]
    n = int(trace["n_steps"])
    labels = np.asarray(torch.as_tensor(labels)).reshape(-1)
    perm_np = np.arange(n) if perm is None else np.asarray(torch.as_tensor(perm)).reshape(-1)[:n]
    if len(perm_np) != n:
        raise ValueError(f"permutation covers {len(perm_np)} steps, trace has {n}")
    columns = {"step": np.arange(n), "dataset_index": perm_np}
    for name, values in steps.items():
        array = values.numpy()
        if array.ndim == 1:
            columns[name] = array
        else:
            for j in range(array.shape[1]):
                columns[f"{name}_{j}"] = array[:, j]
    frame = pd.DataFrame(columns)
    frame["label"] = labels[perm_np]
    frame["clip_correct"] = frame["pred0"].to_numpy() == frame["label"].to_numpy()
    frame["correct"] = frame["out_argmax"].to_numpy() == frame["label"].to_numpy()
    return frame


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True) if path else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", required=True)
    parser.add_argument("--features", required=True, help="dataset-order features (T01 <split>_feats.pt)")
    parser.add_argument("--labels", required=True, help="dataset-order labels (T01 labels_<split>.pt)")
    parser.add_argument("--perm", default=None, help="stream permutation (T01 perm_<split>_seed<k>.pt)")
    parser.add_argument("--out", required=True, help="CSV for the per-snapshot summary")
    parser.add_argument("--steps-out", default=None, help="optional CSV for the per-step table")
    opts = parser.parse_args(argv)

    trace = load_trace(opts.trace)
    labels, perm = _load(opts.labels), _load(opts.perm)
    joined = join_snapshots(trace, _load(opts.features), labels, perm)
    summary = snapshot_summary(joined, int(trace["n_classes"]))
    Path(opts.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(opts.out, index=False)
    if opts.steps_out:
        join_steps(trace, labels, perm).to_csv(opts.steps_out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
