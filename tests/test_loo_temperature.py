# tests/test_loo_temperature.py
import math

import pytest
import torch

from utils.loo_temperature import (
    default_grid,
    loo_accuracy,
    mean_confidence,
    search_temperature,
)


def test_clean_clusters_give_perfect_loo_accuracy():
    # Two tight, well-separated clusters, each consistently pseudo-labelled.
    features = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    )
    features = torch.nn.functional.normalize(features, dim=-1)
    pseudo = torch.tensor([0, 0, 1, 1])
    assert loo_accuracy(features, pseudo) == pytest.approx(1.0)


def test_mislabelled_cache_lowers_loo_accuracy():
    features = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    )
    features = torch.nn.functional.normalize(features, dim=-1)
    pseudo = torch.tensor([0, 1, 1, 1])  # second item is a cache error
    assert loo_accuracy(features, pseudo) < 1.0


def test_loo_accuracy_is_nan_for_a_cache_too_small_to_leave_one_out():
    features = torch.tensor([[1.0, 0.0]])
    pseudo = torch.tensor([0])
    assert math.isnan(loo_accuracy(features, pseudo))


def test_higher_temperature_lowers_mean_confidence():
    logits = torch.tensor([[5.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
    assert mean_confidence(logits, 4.0) < mean_confidence(logits, 1.0)


def test_search_finds_temperature_matching_the_estimated_accuracy():
    logits = torch.tensor([[8.0, 0.0, 0.0]]).repeat(50, 1)
    temperature = search_temperature(logits, target_accuracy=0.5)
    assert mean_confidence(logits, temperature) == pytest.approx(0.5, abs=0.02)
    assert temperature > 1.0  # an overconfident model must be softened


def test_search_falls_back_to_identity_when_estimate_is_unavailable():
    logits = torch.tensor([[8.0, 0.0, 0.0]])
    assert search_temperature(logits, target_accuracy=float("nan")) == 1.0


def test_default_grid_is_smoothing_only():
    # The method may only smooth, never sharpen: the floor is the identity
    # temperature, not a sub-1.0 value that would tighten the distribution.
    grid = default_grid()
    assert grid[0] == 1.0
    assert min(grid) == 1.0
    assert grid[-1] == 20.0


def test_search_pins_to_identity_instead_of_sharpening():
    # At T=1.0 these logits already give ~0.9995 mean confidence. A target
    # above that is only reachable by sharpening (T < 1.0), which the grid no
    # longer offers, so the search must settle on the identity temperature
    # rather than reaching past the floor for a closer, forbidden match.
    logits = torch.tensor([[8.0, 0.0, 0.0]]).repeat(50, 1)
    assert mean_confidence(logits, 1.0) < 0.9999
    temperature = search_temperature(logits, target_accuracy=0.9999)
    assert temperature == 1.0
