"""T03: the O(N) streaming temperature search equals the O(N^2) reference.

`search_temperature(history, a)` (kept unchanged) is the reference; the
streaming `RunningConfidence` must return the same grid temperature on the same
history. See `RunningConfidence`'s docstring for the one place the two may
legitimately differ (float32 reduction noise in the reference, ~1e-7).
"""

import math

import pytest
import torch

from utils.loo_temperature import (
    RunningConfidence,
    default_grid,
    mean_confidence,
    search_temperature,
)

# The reference reduces in float32; its mean confidence is within this of the
# exact (float64) mean the streaming path keeps. Observed max is ~1e-7.
FLOAT32_MEAN_TOL = 1e-6


def _random_history(generator, n, c, scale):
    return torch.randn(n, c, generator=generator) * scale


def _stream(history, generator):
    """Feed `history` in random-size chunks, as the runner feeds single rows."""
    running = RunningConfidence()
    start = 0
    while start < history.shape[0]:
        size = int(torch.randint(1, 4, (1,), generator=generator))
        running.update(history[start : start + size])
        start += size
    return running


def _reference_means(history):
    return torch.tensor([mean_confidence(history, t) for t in default_grid()], dtype=torch.float64)


def test_mean_confidences_match_the_reference_per_temperature():
    generator = torch.Generator().manual_seed(0)
    history = _random_history(generator, 300, 10, 4.0)
    running = _stream(history, generator)
    assert running.count == 300
    diff = (running.mean_confidences() - _reference_means(history)).abs().max()
    assert float(diff) < FLOAT32_MEAN_TOL


def test_streaming_search_equals_the_reference_on_random_histories():
    # Random histories over a range of sizes, class counts and logit scales
    # (from nearly flat to fully saturated), with random targets.
    generator = torch.Generator().manual_seed(1)
    cases = 0
    for n in (1, 2, 7, 60, 300):
        for c in (2, 10, 100):
            for scale in (0.1, 1.0, 5.0, 30.0):
                history = _random_history(generator, n, c, scale)
                running = _stream(history, generator)
                targets = torch.rand(3, generator=generator).tolist() + [0.0, 1.0, 1.0 / c]
                for target in targets:
                    assert running.search(target) == search_temperature(history, target)
                    cases += 1
    assert cases == 5 * 3 * 4 * 6


def test_targets_placed_on_reference_grid_points_pick_the_same_temperature():
    # Adversarial: target equals the reference's own mean confidence at a grid
    # temperature, so the reference's gap there is exactly zero.
    generator = torch.Generator().manual_seed(2)
    history = _random_history(generator, 200, 10, 3.0)
    running = _stream(history, generator)
    grid = default_grid()
    for index in range(0, len(grid), 7):
        target = mean_confidence(history, grid[index])
        assert running.search(target) == search_temperature(history, target)


def test_midpoint_targets_agree_except_within_float32_noise():
    # The hardest case: target halfway between two adjacent grid points' mean
    # confidences, so both candidates' gaps are (nearly) equal. Any
    # disagreement must be a pair whose reference gaps differ by no more than
    # the reference's own float32 error.
    generator = torch.Generator().manual_seed(3)
    history = _random_history(generator, 300, 10, 3.0)
    running = _stream(history, generator)
    grid = default_grid()
    means = _reference_means(history)
    disagreements = 0
    indices = range(0, len(grid) - 1, 4)
    for index in indices:
        target = float((means[index] + means[index + 1]) / 2)
        ours, ref = running.search(target), search_temperature(history, target)
        if ours != ref:
            disagreements += 1
            gap_ours = abs(mean_confidence(history, ours) - target)
            gap_ref = abs(mean_confidence(history, ref) - target)
            assert abs(gap_ours - gap_ref) < FLOAT32_MEAN_TOL
    # Exact midpoints are where the two paths are *expected* to split (about
    # half the time: each rounds the tie its own way), so no bound on the count.
    assert disagreements < len(indices)


def test_flat_history_ties_resolve_to_the_first_grid_point():
    # Uniform logits: every temperature gives exactly 1/C, all gaps tie, and
    # the reference keeps the first (identity) temperature.
    history = torch.zeros(20, 5)
    running = RunningConfidence()
    running.update(history)
    for target in (0.0, 0.2, 0.7):
        assert running.search(target) == search_temperature(history, target) == 1.0


def test_unavailable_estimate_falls_back_to_identity():
    running = RunningConfidence()
    running.update(torch.randn(3, 4))
    assert running.search(float("nan")) == 1.0
    assert running.search(None) == 1.0


def test_search_before_any_update_is_an_error():
    with pytest.raises(ValueError):
        RunningConfidence().search(0.5)


def test_custom_grid_matches_the_reference():
    generator = torch.Generator().manual_seed(4)
    history = _random_history(generator, 100, 7, 2.0)
    grid = [0.5, 0.75, 1.0, 2.0, 4.0]
    running = RunningConfidence(grid)
    running.update(history)
    for target in torch.rand(20, generator=generator).tolist():
        assert running.search(target) == search_temperature(history, target, grid)


def test_float64_sums_are_exact_for_float32_terms():
    # The exactness argument in `RunningConfidence`: float32 values in
    # [1/C, 1] add without rounding in float64 at these sizes.
    generator = torch.Generator().manual_seed(5)
    values = (torch.rand(10_000, generator=generator) * 0.99 + 0.01).float()
    forward = values.double().sum()
    backward = values.double().flip(0).cumsum(0)[-1]
    assert float(forward) == float(backward)
    assert math.isfinite(float(forward))
