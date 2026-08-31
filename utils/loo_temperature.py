"""Estimate how accurate the cache is, without labels, and soften logits to match.

The idea: a TTA cache holds features together with the labels the method itself
assigned. Leave-one-out nearest-neighbour agreement inside that cache is a
label-free proxy for how often the method is right. A well-calibrated model's
mean top-1 confidence should equal its accuracy, so we pick the softmax
temperature whose mean confidence lands on that estimate.

The search is smoothing-only: `default_grid()`'s floor is 1.0 (the identity
temperature), so no candidate can sharpen a prediction. This is a design
constraint, not an oversight. Margin admission selectively caches high-margin,
likely-correct samples, which biases `loo_accuracy` upward -- a purer subset
agrees with itself more than the true accuracy would. An inflated target then
pulls the unconstrained search below 1.0, sharpening predictions and making
overconfidence *worse*, exactly the opposite of this module's purpose. Capping
the grid at 1.0 means the worst the search can do is a no-op; it can only push
the model toward better calibration, never away from it.

Nothing here ever touches a ground-truth label.
"""

from __future__ import annotations

import math

import torch


def loo_accuracy(features: torch.Tensor, pseudo_labels: torch.Tensor) -> float:
    """Leave-one-out 1-NN agreement among cached items.

    `features` is (M, D) and should be L2-normalized. `pseudo_labels` is (M,)
    and holds the labels the *method* assigned when caching each item.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be (M, D), got shape {tuple(features.shape)}")
    if features.shape[0] != pseudo_labels.shape[0]:
        raise ValueError("features and pseudo_labels must have the same length")
    if features.shape[0] < 2:
        return float("nan")

    similarity = features @ features.t()
    similarity.fill_diagonal_(float("-inf"))
    neighbour = similarity.argmax(dim=1)
    return float((pseudo_labels[neighbour] == pseudo_labels).float().mean())


def mean_confidence(logits: torch.Tensor, temperature: float) -> float:
    """Mean top-1 probability after dividing logits by `temperature`."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    probs = (logits / temperature).softmax(dim=-1)
    return float(probs.max(dim=-1).values.mean())


def default_grid() -> list[float]:
    # Floor is 1.0, not 0.5: smoothing-only, see module docstring -- the
    # search may push confidence down but never manufacture it. Ceiling must
    # still reach 8/ln(2) = 11.54, the temperature that maps the test's
    # 3-class logits [8, 0, 0] to a mean confidence of 0.50 (a grid capped at
    # 10.0 only reaches 0.5267 and misses the 0.02 tolerance).
    return [round(1.0 + 0.05 * i, 2) for i in range(381)]  # 1.00 ... 20.00


def search_temperature(
    logits: torch.Tensor,
    target_accuracy: float,
    grid: list[float] | None = None,
) -> float:
    """Grid-search the temperature whose mean confidence matches `target_accuracy`.

    Returns 1.0 (a no-op) when the estimate is unavailable, e.g. the cache was
    too small for leave-one-out.
    """
    if target_accuracy is None or math.isnan(target_accuracy):
        return 1.0

    candidates = grid if grid is not None else default_grid()
    best, best_gap = 1.0, float("inf")
    for temperature in candidates:
        gap = abs(mean_confidence(logits, temperature) - target_accuracy)
        if gap < best_gap:
            best, best_gap = temperature, gap
    return best
