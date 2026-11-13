import pandas as pd
import torch

from analysis.adaptive_alpha import disagree_mask
from analysis.switch import (ALPHA, mean_over_datasets, method_rows, running_confidence, select_threshold,
                             switch_logits)
from tests.test_story import _run


def test_running_confidence_is_causal_and_a_running_mean():
    r = _run()
    rc = running_confidence(r.clip)
    conf = r.clip.softmax(-1).max(-1).values
    assert torch.allclose(rc[9], conf[:10].mean())
    later = r.clip.clone()
    later[200:] *= 10  # change only the future
    assert torch.equal(running_confidence(later)[:200], rc[:200])


def test_switch_extremes_are_the_gate_and_tda():
    r = _run()
    tda = r.base + ALPHA * r.cache
    gate = torch.where(disagree_mask(r)[:, None], tda, r.base)
    assert torch.equal(switch_logits(r, float("inf")), gate)
    assert torch.equal(switch_logits(r, -1.0), tda)


def test_switch_keeps_tda_exactly_where_the_running_confidence_is_above_the_threshold():
    r = _run()
    t = float(running_confidence(r.clip).median())
    keep = running_confidence(r.clip) > t
    assert keep.any() and (~keep).any()
    out, tda = switch_logits(r, t), r.base + ALPHA * r.cache
    gate = switch_logits(r, float("inf"))
    assert torch.equal(out[keep], tda[keep])
    assert torch.equal(out[~keep], gate[~keep])


def test_select_threshold_takes_the_centre_of_the_near_minimal_plateau():
    curve = ((0.6, 4.0), (0.7, 2.05), (0.8, 2.08), (0.9, 2.0), (1.0, 5.0))
    rows = [{"dataset": d, "seed": s, "threshold": t, "accuracy": 50.0, "ece": e}
            for d in ("a", "b") for s in (0, 1) for t, e in curve]
    assert select_threshold(pd.DataFrame(rows)) == 0.8
    assert select_threshold(pd.DataFrame(rows), width=0.01) == 0.9


def test_mean_over_datasets_averages_datasets_per_seed_then_seeds():
    f = pd.DataFrame({"m": ["x"] * 4, "dataset": ["a", "b", "a", "b"], "seed": [0, 0, 1, 1],
                      "accuracy": [1.0, 3.0, 5.0, 7.0], "ece": [0.0, 2.0, 4.0, 6.0]})
    out = mean_over_datasets(f, ["m"])
    assert out.loc[0, "accuracy"] == 4.0 and out.loc[0, "ece"] == 3.0


def test_method_rows_report_every_method_and_the_switch_share():
    frame = method_rows([_run()], "test", 0.5)
    assert set(frame.method) == {"TDA", "agreement gate", "TDA + ZAEC", "confidence switch"}
    assert frame.switch_on_pct.between(0, 100).all()
    assert frame.clip_conf.between(0, 100).all()
    assert frame.clip_gap.between(-100, 100).all()
