import numpy as np
import pandas as pd
import torch

from analysis.adaptive_alpha import disagree_mask
from analysis.no_info import (MEASURES, ablation_rows, auc, filter_gain, fit_logistic, flip_records, info_tables,
                              oracle_rows, variant_logits)
from tests.test_story import _run


def test_variants_split_the_positive_cache_by_agreement():
    r = _run()
    v = variant_logits(r)
    assert torch.allclose(v["apos"] + v["gate"] - v["base"], v["tda"])
    dis = disagree_mask(r)
    assert torch.equal(v["gate"][~dis], v["base"][~dis]) and torch.equal(v["apos"][dis], v["base"][dis])


def test_ablation_rows_report_every_variant_and_both_groups():
    frame = ablation_rows([_run(), _run(seed=1)], "test")
    assert len(frame) == 2
    for k in ("clip", "base", "apos", "gate", "tda", "gpos"):
        assert frame[f"{k}_acc"].between(0, 100).all()
    assert np.allclose(frame.agree_share + frame.disagree_share, 100)


def test_flip_records_keep_only_decided_flips_with_label_free_measures():
    r = _run()
    f = flip_records(r)
    p0, p1 = r.clip.argmax(-1), (r.base + 2.0 * r.cache).argmax(-1)
    n_right = int(((p1 != p0) & (p1 == r.labels)).sum())
    n_wrong = int(((p1 != p0) & (p0 == r.labels)).sum())
    assert f.right.sum() == n_right and (~f.right).sum() == n_wrong
    assert set(MEASURES) <= set(f.columns)


def test_auc_is_the_probability_a_right_flip_scores_higher():
    assert auc([True, True, False, False], [3, 4, 1, 2]) == 1.0
    assert auc([True, False], [1, 1]) == 0.5
    assert auc([True, True, False, False], [1, 2, 3, 4]) == 0.0


def test_filter_gain_counts_blocked_wrong_minus_blocked_right():
    f = pd.DataFrame({"right": [True, False, False, True], "dataset": "a", "seed": 0, "n": 100})
    g = filter_gain(f, np.array([0.1, 0.2, 0.9, 0.9]), 0.5)  # blocks one right and one wrong
    assert g.accuracy.item() == 0.0
    g = filter_gain(f, np.array([0.9, 0.2, 0.3, 0.9]), 0.5)  # blocks two wrong
    assert g.accuracy.item() == 2.0


def test_logistic_scorer_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    right = rng.random(400) < 0.5
    f = pd.DataFrame({k: rng.normal(size=400) for k in MEASURES})
    f["clip_conf"] += 3 * right
    f["right"] = right
    assert auc(right, fit_logistic(f)(f)) > 0.9


def test_oracles_are_at_least_the_block_oracle_and_info_tables_cover_every_tier():
    runs = [_run(seed=s) for s in (0, 1)]
    orc = pd.DataFrame([oracle_rows(r, t) for t in ("val", "test") for r in runs])
    assert (orc.oracle_pick >= orc.oracle_block - 1e-12).all()
    flips = pd.concat([flip_records(r).assign(tier=t) for t in ("val", "test") for r in runs], ignore_index=True)
    aucs, rates, comb = info_tables(flips, orc)
    assert set(aucs.tier) == {"val", "test"} and "combined" in set(aucs.measure)
    assert aucs.auc.between(0, 1).all() and set(comb.tier) == {"val", "test"}
    assert (comb.filter_gain <= comb.oracle_block + 1e-9).all()
