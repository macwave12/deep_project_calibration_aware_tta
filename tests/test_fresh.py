import json

import pandas as pd
import pytest
import torch

from analysis.fresh import complete, fresh_records, fresh_table, head_signals, record_dece
from analysis.story import signal_row


def _write(root, ds, method, seed, acc, ece):
    d = root if seed == 0 else root / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{ds}_{method}.json").write_text(json.dumps(
        {"dataset": ds, "method": method, "seed": seed, "accuracy": acc, "ece": ece}), encoding="utf-8")


def test_fresh_table_pairs_gate_against_tda_by_seed(tmp_path):
    for seed, (t_acc, g_acc, t_ece, g_ece) in enumerate([(0.60, 0.61, 0.10, 0.04), (0.62, 0.61, 0.12, 0.05),
                                                       (0.58, 0.59, 0.11, 0.03)]):
        _write(tmp_path, "cars", "tda_eq_clean", seed, t_acc, t_ece)
        _write(tmp_path, "cars", "cal_tda_agree_clean", seed, g_acc, g_ece)
        _write(tmp_path, "cars", "clipzs", seed, 0.55, 0.05)
        _write(tmp_path, "cars", "tda_eq_clean_head500", seed, 0.1, 0.1)  # ignored
    frame = fresh_records([tmp_path], [0, 1, 2])
    assert set(frame.method) == {"CLIP", "TDA", "agreement gate"}
    t = fresh_table(frame)
    row = t[t.dataset == "cars"].iloc[0]
    assert row["d_accuracy"] == pytest.approx(100 * (0.01 - 0.01 + 0.01) / 3)
    assert row["d_ece"] == pytest.approx(100 * (-0.06 - 0.07 - 0.08) / 3)
    assert row["ece_wins"] == "3/3"
    assert row["clip_accuracy"] == pytest.approx(55.0)
    assert list(t.dataset) == ["cars", "mean"]


def test_head_signals_use_only_the_first_samples(tmp_path):
    n, c = 50, 5
    clip = torch.zeros(n, c)
    clip[:10, 0] = 100.0  # the first 10 are very confident; the rest uniform
    for seed in (0, 1):
        d = tmp_path if seed == 0 else tmp_path / f"seed{seed}"
        d.mkdir(exist_ok=True)
        torch.save({"steps": {"clip_logits": clip, "base_logits": clip}}, d / "x_tda_eq_clean.trace.pt")
    s = head_signals(tmp_path, "x", [0, 1], "tda_eq_clean", head=10)
    assert s["over06"] == pytest.approx(100.0)
    assert s["share"] == pytest.approx(100.0)
    s = head_signals(tmp_path, "x", [0, 1], "tda_eq_clean", head=20)
    assert s["over06"] == pytest.approx(50.0)
    assert s["conf"] == pytest.approx(50.0 + 50.0 / c)


def test_signal_row_negative_cache_range_widens_the_no_flip_bound():
    clip = torch.tensor([[10.0, 0.0, 0.0]])
    assert signal_row("x", clip, clip)["share"] == 100.0  # gap 10 > alpha * S = 6
    base = clip - torch.tensor([[0.0, 5.0, 0.0]])
    assert signal_row("x", clip, base)["share"] == 0.0  # 10 < 6 + 5
    assert isinstance(pd.DataFrame([signal_row("x", clip, base)]), pd.DataFrame)


def test_clip_gap_is_the_count_weighted_confidence_minus_accuracy(tmp_path):
    from analysis.fresh import clip_gap
    path = tmp_path / "x_clipzs.json"
    path.write_text(json.dumps({"reliability": {"bin_count": [1, 3], "bin_conf": [0.2, 0.9], "bin_acc": [0.0, 1.0]}}),
                    encoding="utf-8")
    assert clip_gap(path) == pytest.approx(100 * (1 * 0.2 + 3 * -0.1) / 4)


def test_incomplete_datasets_are_left_out_and_record_dece_pairs_seeds(tmp_path):
    for seed in (0, 1):
        _write(tmp_path, "cars", "tda_eq_clean", seed, 0.6, 0.10)
        _write(tmp_path, "cars", "cal_tda_agree_clean", seed, 0.6, 0.04 + 0.01 * seed)
        _write(tmp_path, "cars", "clipzs", seed, 0.5, 0.05)
    _write(tmp_path, "gtsrb", "tda_eq_clean", 0, 0.4, 0.1)
    _write(tmp_path, "gtsrb", "clipzs", 0, 0.4, 0.1)
    with pytest.warns(UserWarning, match="gtsrb"):
        frame = complete(fresh_records([tmp_path], [0, 1]), [0, 1])
    assert set(frame.dataset) == {"cars"}
    assert record_dece(tmp_path, "cars", [0, 1]) == pytest.approx(-5.5)


def test_record_gaps_average_tda_and_gate_over_seeds_and_take_clip_from_seed0(tmp_path):
    from analysis.fresh import record_gaps

    def rel(conf, acc, seed, name):
        d = tmp_path if seed == 0 else tmp_path / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"x_{name}.json").write_text(json.dumps(
            {"reliability": {"bin_count": [1], "bin_conf": [conf], "bin_acc": [acc]}}), encoding="utf-8")
    rel(0.5, 0.6, 0, "clipzs")
    for seed, (t, g) in enumerate(((0.9, 0.6), (0.7, 0.4))):
        rel(t, 0.5, seed, "tda_eq_clean")
        rel(g, 0.5, seed, "cal_tda_agree_clean")
    gaps = record_gaps(tmp_path, "x", [0, 1])
    assert gaps["clip_gap"] == pytest.approx(-10.0)
    assert gaps["tda_gap"] == pytest.approx(30.0) and gaps["gate_gap"] == pytest.approx(0.0)
