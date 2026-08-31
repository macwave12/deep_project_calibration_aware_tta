import numpy as np
import pytest

from utils.calibration import compute_accuracy, compute_ece, reliability_curve


def test_accuracy_counts_argmax_matches():
    probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([0, 1, 1])
    assert compute_accuracy(probs, labels) == pytest.approx(2 / 3)


def test_confident_and_correct_gives_zero_ece():
    probs = np.zeros((100, 4))
    probs[:, 0] = 1.0
    labels = np.zeros(100, dtype=int)
    assert compute_ece(probs, labels, n_bins=20) == pytest.approx(0.0, abs=1e-9)


def test_confident_and_wrong_gives_maximal_ece():
    probs = np.zeros((100, 4))
    probs[:, 0] = 1.0
    labels = np.ones(100, dtype=int)
    assert compute_ece(probs, labels, n_bins=20) == pytest.approx(1.0, abs=1e-9)


def test_ece_matches_hand_computed_two_bin_example():
    # Two occupied bins, each holding half the samples.
    # bin (0.90, 0.95]: accuracy 0.5, mean confidence 0.95 -> gap 0.45
    # bin (0.50, 0.55]: accuracy 1.0, mean confidence 0.55 -> gap 0.45
    probs = np.array(
        [[0.95, 0.05], [0.95, 0.05], [0.55, 0.45], [0.55, 0.45]]
    )
    labels = np.array([0, 1, 0, 0])
    assert compute_ece(probs, labels, n_bins=20) == pytest.approx(0.45, abs=1e-9)


def test_reliability_curve_bins_account_for_every_sample():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 10))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, 10, size=500)

    curve = reliability_curve(probs, labels, n_bins=20)

    assert len(curve["bin_count"]) == 20
    assert sum(curve["bin_count"]) == 500


def test_reliability_curve_is_consistent_with_ece():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(300, 5)) * 3
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, 5, size=300)

    curve = reliability_curve(probs, labels, n_bins=20)
    manual = sum(
        (c / 300) * abs(a - f)
        for c, a, f in zip(curve["bin_count"], curve["bin_acc"], curve["bin_conf"])
        if c > 0
    )
    assert compute_ece(probs, labels, n_bins=20) == pytest.approx(manual, abs=1e-12)
