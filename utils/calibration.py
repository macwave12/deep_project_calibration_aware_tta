"""Calibration metrics shared by every method in this project.

One implementation, used to score clipzs, tda and cal_tda alike, so the numbers
are comparable to Table 7 of the benchmark paper (arXiv:2506.24000).
Bins are equal-width and right-closed: bin b covers (b/n_bins, (b+1)/n_bins],
with confidence exactly 0.0 falling in bin 0.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (N, C), got shape {probs.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs has {probs.shape[0]} rows but labels has {labels.shape[0]}"
        )
    return probs, labels


def _confidence_and_correctness(probs, labels):
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)
    return confidence, correct


def _bin_indices(confidence, n_bins):
    # Interior edges only; digitize(right=True) gives bins[i-1] < x <= bins[i].
    interior = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.clip(np.digitize(confidence, interior, right=True), 0, n_bins - 1)


def compute_accuracy(probs, labels) -> float:
    probs, labels = _as_arrays(probs, labels)
    _, correct = _confidence_and_correctness(probs, labels)
    return float(correct.mean())


def compute_ece(probs, labels, n_bins: int = 20) -> float:
    probs, labels = _as_arrays(probs, labels)
    confidence, correct = _confidence_and_correctness(probs, labels)
    index = _bin_indices(confidence, n_bins)

    total = confidence.shape[0]
    ece = 0.0
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def reliability_curve(probs, labels, n_bins: int = 20) -> dict:
    probs, labels = _as_arrays(probs, labels)
    confidence, correct = _confidence_and_correctness(probs, labels)
    index = _bin_indices(confidence, n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    curve = {
        "bin_lower": edges[:-1].tolist(),
        "bin_upper": edges[1:].tolist(),
        "bin_conf": [],
        "bin_acc": [],
        "bin_count": [],
    }
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        curve["bin_count"].append(count)
        curve["bin_conf"].append(float(confidence[mask].mean()) if count else 0.0)
        curve["bin_acc"].append(float(correct[mask].mean()) if count else 0.0)
    return curve
