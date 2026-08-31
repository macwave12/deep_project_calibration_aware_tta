"""The one and only place ground-truth labels are consumed.

Methods (clipzs / tda / cal_tda) produce per-sample probability vectors and never
see a label. At the end of a run, the driver hands those probabilities plus the
labels to `score_and_save`, which computes the metrics and writes a
self-describing JSON record. Every number in the report traces back to one of
these files.
"""

from __future__ import annotations

import datetime as _datetime
import json
from pathlib import Path

import numpy as np

from utils.calibration import compute_accuracy, compute_ece, reliability_curve


def score_and_save(
    probs,
    labels,
    *,
    dataset: str,
    method: str,
    backbone: str,
    seed: int,
    hyperparams: dict,
    out_dir,
    n_bins: int = 20,
    timestamp: str | None = None,
) -> Path:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)

    if probs.ndim != 2:
        raise ValueError(f"probs must be (N, C), got shape {probs.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs has {probs.shape[0]} rows but labels has {labels.shape[0]}"
        )
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(
            "probs rows must be probability distributions summing to 1; "
            f"worst row sum was {float(np.abs(row_sums - 1.0).max() + 1.0):.6f}"
        )

    record = {
        "dataset": dataset,
        "method": method,
        "backbone": backbone,
        "seed": seed,
        "n_samples": int(probs.shape[0]),
        "n_classes": int(probs.shape[1]),
        "n_bins": n_bins,
        "accuracy": compute_accuracy(probs, labels),
        "ece": compute_ece(probs, labels, n_bins=n_bins),
        "reliability": reliability_curve(probs, labels, n_bins=n_bins),
        "hyperparams": hyperparams,
        "timestamp": timestamp or _datetime.datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_{method}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
