import math

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.adaptive_alpha import (ALPHA_GRID, Run, agree_logits, const_rule, disagree_mask,
                                     fit_rule, nll_matrix, probs_at, roc_auc_score, signals)
from analysis.t14_results import fit_temperature, matched


def _run(n=300, c=6, seed=0, dataset="d"):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, c, (n,), generator=g)
    clip = torch.randn(n, c, generator=g, dtype=torch.float64) * 2
    clip[torch.arange(n), labels] += 2.0
    cache = torch.rand(n, c, generator=g, dtype=torch.float64)
    cache[torch.arange(n), labels] += 0.5
    neg = 0.1 * torch.rand(n, c, generator=g, dtype=torch.float64)
    run = Run(dataset, seed, clip - neg, cache, clip, labels, torch.zeros(n, dtype=torch.bool))
    signals(run)
    return run


def test_signals_are_bounded_and_label_free():
    r = _run()
    for name in ("clip_conf", "clip_conf_rel", "clip_ent", "clip_margin", "cache_conf",
                 "cache_support_clip_top", "rank_clip_top", "agree", "js"):
        v = r.s[name]
        assert torch.isfinite(v).all() and v.min() >= -1e-9 and v.max() <= 1 + 1e-9, name
    # p_cache is c / sum(c): the cache's argmax decides agreement.
    assert torch.equal(r.s["agree"].bool(), r.cache.argmax(-1) == r.clip.argmax(-1))


def test_nll_matrix_matches_probs_at():
    r = _run()
    nll = nll_matrix(r)
    j = int(torch.nonzero(ALPHA_GRID == 1.5)[0])
    p = probs_at(r, 1.5)
    assert torch.allclose(nll[:, j], -p.gather(1, r.labels[:, None]).squeeze(1).log())


def test_roc_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    pos = rng.random(200) < 0.4
    score = rng.normal(size=200) + pos
    pairs = np.mean([(a > b) + 0.5 * (a == b) for a in score[pos] for b in score[~pos]])
    assert roc_auc_score(pos, score) == pytest.approx(pairs)


def test_agree_logits_use_cache_only_where_it_disagrees():
    r = _run()
    dis = disagree_mask(r)
    lg = agree_logits(r)
    assert torch.equal(lg[~dis], r.base[~dis])
    assert torch.allclose(lg[dis], r.base[dis] + 2 * r.cache[dis])
    t = agree_logits(r, t_dis=2.0)
    assert torch.allclose(t[dis], lg[dis] / 2) and torch.equal(t[~dis], lg[~dis])


def test_const_rule_fit_reaches_the_grid_optimum():
    runs = [_run(seed=s) for s in range(2)]
    params = fit_rule(const_rule(), runs)
    alpha = float(torch.nn.functional.softplus(params[0]))
    # With T free too, the fit must be at least as good as the best constant alpha at T = 1.
    best_grid = min(float(torch.cat([nll_matrix(r) for r in runs])[:, i].mean())
                    for i in range(len(ALPHA_GRID)))
    from analysis.adaptive_alpha import dataset_loss
    assert float(dataset_loss(const_rule(), params, runs)) <= best_grid + 1e-6
    assert alpha >= 0


def test_fit_temperature_recovers_a_known_temperature():
    # Labels drawn from softmax(logits / 2): the NLL-optimal T is ~2.
    g = torch.Generator().manual_seed(1)
    n, c = 20000, 5
    logits = torch.randn(n, c, generator=g, dtype=torch.float64) * 4
    labels = torch.multinomial((logits / 2).softmax(-1), 1, generator=g).squeeze(1)
    run = Run("d", 0, logits, torch.zeros(n, c, dtype=torch.float64), logits, labels,
              torch.zeros(n, dtype=torch.bool))
    assert fit_temperature([run]) == pytest.approx(2.0, rel=0.05)


def test_matched_interpolates_on_the_curve():
    curve = pd.DataFrame({"dataset": "d", "alpha": [0, 1, 2], "acc": [50.0, 54.0, 56.0],
                          "ece": [4.0, 8.0, 12.0]})
    points = pd.DataFrame({"dataset": ["d", "d"], "method": ["m", "far"], "acc": [55.0, 70.0],
                           "ece": [5.0, 1.0]})
    row = matched(curve, points, "m").iloc[0]
    assert row["tda_best_ece_at_same_acc"] == pytest.approx(10.0)
    assert row["tda_best_acc_at_same_ece"] == pytest.approx(51.0)
    far = matched(curve, points, "far").iloc[0]
    assert math.isnan(far["tda_best_ece_at_same_acc"]) and math.isnan(far["tda_best_acc_at_same_ece"])


def test_every_rule_family_fits_with_the_temperature_parameter():
    from analysis.adaptive_alpha import (SIGNALS, dataset_loss, ladder_candidates, linear_all_rule,
                                         logistic2_rule)
    runs = [_run(seed=s, dataset=f"d{s}") for s in range(2)]
    rungs = ladder_candidates(runs)
    rules = [rungs[0][0], rungs[1][0], rungs[2][0], logistic2_rule("clip_conf", "js"), rungs[4][0]]
    for rule in rules:
        params = fit_rule(rule, runs, steps=5)
        assert params.numel() == rule.n_params + 1
        assert torch.isfinite(dataset_loss(rule, params, runs))
