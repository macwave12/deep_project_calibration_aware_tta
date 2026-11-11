import pandas as pd
import pytest

from analysis.p1_appendix import DATASETS, p1_vs_p2, reproduction


def _frames():
    lit = []
    for m, base in (("clipzs", 40.0), ("tda", 45.0)):
        for i, d in enumerate(DATASETS):
            lit.append({"key": f"bench_vitb16_{m}_{d}_acc", "value": base + i})
            lit.append({"key": f"tda_paper_vitb16_{m}_{d}_acc", "value": base + i + 1})
    summary = pd.DataFrame([{"dataset": d, "method": m, "accuracy": b + i + 0.1}
                            for m, b in (("clipzs", 40.0), ("tda", 45.0)) for i, d in enumerate(DATASETS)])
    prot = []
    for proto, shift in (("P1", 0.0), ("P2", 0.5)):
        for m, b in (("clipzs", 40.0), ("tda", 45.0)):
            for i, d in enumerate(DATASETS + ["mean"]):
                for metric in ("accuracy", "ece"):
                    prot.append({"protocol": proto, "dataset": d, "method": m, "metric": metric,
                                 "n": 3, "mean": b + i + shift, "std": 0.1})
    paired = pd.DataFrame([{"comparison": "P2_minus_P1", "dataset": d, "method": "tda",
                            "metric": metric, "mean_diff": 0.5, "wins": 3, "n": 3}
                           for d in DATASETS + ["mean"] for metric in ("accuracy", "ece")])
    return pd.DataFrame(lit), summary, pd.DataFrame(prot), paired


def test_reproduction_compares_gpu_run_with_the_benchmark():
    lit, summ, prot, _ = _frames()
    rep = reproduction(lit, summ, prot)
    row = rep[(rep.method == "tda") & (rep.dataset == "dtd")].iloc[0]
    assert row["benchmark_table14"] == 45.0 and row["delta_gpu_vs_benchmark"] == pytest.approx(0.1)
    assert row["ours_replay_p2_mean"] == 45.5
    mean = rep[(rep.method == "clipzs") & (rep.dataset == "mean")].iloc[0]
    assert mean["benchmark_table14"] == pytest.approx(42.0)


def test_p1_vs_p2_joins_the_paired_difference():
    _, _, prot, paired = _frames()
    out = p1_vs_p2(prot, paired)
    row = out[(out.method == "tda") & (out.dataset == "mean") & (out.metric == "accuracy")].iloc[0]
    assert row["mean_P2"] - row["mean_P1"] == pytest.approx(0.5)
    assert row["p2_minus_p1"] == 0.5 and row["seeds_p2_better"] == 3
