import numpy as np
import pandas as pd
import torch

from analysis.adaptive_alpha import disagree_mask
from analysis.no_info import (MEASURES, ablation_rows, auc, corr_rows, decile_rows, filter_gain, fit_logistic,
                              flip_records, info_tables, oracle_rows, random_mask, rule_rows, variant_logits,
                              within_percentile)
from tests.test_story import _run


def test_variants_split_the_positive_cache_by_agreement():
    r = _run()
    v = variant_logits(r)
    assert torch.allclose(v["apos"] + v["gate"] - v["base"], v["tda"])
    dis = disagree_mask(r)
    assert torch.equal(v["gate"][~dis], v["base"][~dis]) and torch.equal(v["apos"][dis], v["base"][dis])
    # the control opens on as many samples as the gate, chosen at random
    opened = (v["rand"] != v["base"]).any(-1)
    assert int(opened.sum()) <= int(dis.sum()) and torch.equal(v["rand"], variant_logits(r)["rand"])


def test_random_mask_draws_the_requested_share_reproducibly():
    m = random_mask(1000, 0.42, seed=3)
    assert int(m.sum()) == 420 and torch.equal(m, random_mask(1000, 0.42, seed=3))
    assert not torch.equal(m, random_mask(1000, 0.42, seed=4))


def test_ablation_rows_report_every_variant_and_both_groups():
    frame = ablation_rows([_run(), _run(seed=1)], "test")
    assert len(frame) == 2
    for k in ("clip", "base", "apos", "gate", "rand", "tda"):
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


def test_rule_rows_choose_side_and_threshold_on_val_and_apply_them_unchanged():
    rng = np.random.default_rng(0)
    rows = []
    for tier, informative in (("val", True), ("fresh", False)):
        right = rng.random(400) < 0.5
        f = pd.DataFrame({k: rng.normal(size=400) for k in MEASURES})
        if informative:
            f["clip_conf"] += 3 * right  # low CLIP confidence marks a wrong flip, on val only
        rows.append(f.assign(right=right, tier=tier, dataset="a", seed=0, n=1000))
    r = rule_rows(pd.concat(rows, ignore_index=True)).set_index(["measure", "tier"])
    assert r.loc[("clip_conf", "val"), "side"] == "below" and r.loc[("clip_conf", "val"), "gain"] > 10
    assert abs(r.loc[("clip_conf", "fresh"), "gain"]) < 5
    assert r.loc[("clip_conf", "val"), "threshold"] == r.loc[("clip_conf", "fresh"), "threshold"]


def test_correlations_and_deciles_are_within_each_dataset():
    rng = np.random.default_rng(0)
    rows = []
    for ds, shift in (("a", 0.0), ("b", 100.0)):  # same within-dataset signal, very different score levels
        right = rng.random(500) < 0.6
        f = pd.DataFrame({k: rng.normal(size=500) + shift for k in MEASURES})
        f["clip_conf"] += 2 * right
        rows.append(f.assign(right=right, tier="near", dataset=ds, seed=0, n=1000))
    flips = pd.concat(rows, ignore_index=True)
    q = within_percentile(flips, "clip_conf")
    assert q.between(0, 1).all() and abs(q[flips.dataset == "b"].mean() - 0.5) < 0.01
    c = corr_rows(flips).set_index("measure")
    assert c.loc["clip_conf", "rho"] > 0.5 and abs(c.loc["cache_conf", "rho"]) < 0.15
    d = decile_rows(flips)
    cc = d[d.measure == "clip_conf"].sort_values("decile")
    assert list(cc.decile) == list(range(10)) and cc.n.sum() == 1000
    assert cc.right_pct.iloc[0] < 30 and cc.right_pct.iloc[-1] > 90
