"""Derived deck statistics: exact against the run records, bounds checked by brute force."""

import json
import math
import random
from pathlib import Path

import pandas as pd
import pytest
import yaml

from analysis.deck_evidence import (
    PROTOCOL_P2, additive_overturn_log_odds, dataset_constants, flat_cache_top_probability,
    mixture_overturn_gap, reliability_stats, reliability_summary, vote_halving_cosine,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reliability_summary_recovers_the_recorded_accuracy_and_ece():
    for path in sorted((ROOT / "results/raw").rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "reliability" not in record:
            continue
        stats = reliability_summary(record)
        assert stats["accuracy"] == pytest.approx(100 * record["accuracy"], abs=1e-6), path
        assert stats["ece"] == pytest.approx(100 * record["ece"], abs=1e-6), path
        # ECE >= |gap| (triangle inequality), the fact the deck's argument uses
        assert stats["ece"] + 1e-9 >= abs(stats["gap"]), path


def test_reliability_summary_on_a_hand_built_record():
    record = {"reliability": {"bin_count": [2, 0, 2], "bin_conf": [0.2, 0.0, 0.9], "bin_acc": [0.5, 0.0, 0.5]}}
    stats = reliability_summary(record)
    assert stats["mean_confidence"] == pytest.approx(55.0)
    assert stats["accuracy"] == pytest.approx(50.0)
    assert stats["gap"] == pytest.approx(5.0)
    assert stats["ece"] == pytest.approx(35.0)


def test_tracked_csvs_are_up_to_date(tmp_path):
    fresh = reliability_stats(ROOT / "results/raw")
    tracked = pd.read_csv(ROOT / "results/analysis/reliability_stats.csv")
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), tracked, check_dtype=False)
    fresh = dataset_constants(ROOT / "results/raw", ROOT / "configs")
    tracked = pd.read_csv(ROOT / "results/analysis/dataset_constants.csv")
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), tracked, check_dtype=False)


def test_reliability_stats_matches_summary_csv():
    rel = reliability_stats(ROOT / "results/raw")
    head = rel[rel.study == "headline"].set_index(["dataset", "variant"])
    summary = pd.read_csv(ROOT / "results/summary.csv")
    for row in summary.itertuples():
        assert head.loc[(row.dataset, row.method), "accuracy"] == pytest.approx(row.accuracy, abs=0.01)
        assert head.loc[(row.dataset, row.method), "ece"] == pytest.approx(row.ece, abs=0.01)


def test_protocol_p2_is_tda_papers_single_setting():
    tda = yaml.safe_load((ROOT / "configs/method/tda.yaml").read_text(encoding="utf-8"))
    assert (PROTOCOL_P2["alpha"], PROTOCOL_P2["beta"]) == (tda["alpha"], tda["beta"])


def test_vote_halving_cosine():
    for beta in (1.0, 2.0, 5.0, 8.0):
        assert math.exp(-beta * (1 - vote_halving_cosine(beta))) == pytest.approx(0.5)


def test_additive_overturn_bound_is_tight():
    alpha, capacity = 2.0, 3
    bound = additive_overturn_log_odds(alpha, capacity)
    # class 2 full cache (three identical items), class 1 empty: shift is exactly alpha * capacity
    assert alpha * (capacity * math.exp(0) - 0.0) == pytest.approx(bound)


def test_mixture_overturn_gap_bounds_random_caches():
    rng = random.Random(0)
    for n_classes in (10, 37, 100):
        bound = mixture_overturn_gap(n_classes, 3)
        for _ in range(500):
            c = [rng.uniform(0, 3) if rng.random() < 0.3 else 0.0 for _ in range(n_classes)]
            e = [math.exp(v) for v in c]
            q = [v / sum(e) for v in e]
            assert max(q[i] - q[j] for i in range(3) for j in range(3)) <= bound + 1e-12
        # attained by one class at the cap and the rest at zero
        e = [math.exp(3)] + [1.0] * (n_classes - 1)
        q = [v / sum(e) for v in e]
        assert q[0] - q[1] == pytest.approx(bound)


def test_flat_cache_top_probability():
    assert flat_cache_top_probability(100, 0.0) == pytest.approx(0.01)
    assert flat_cache_top_probability(10, math.log(9)) == pytest.approx(0.5)
