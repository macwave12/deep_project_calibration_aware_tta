import numpy as np
import pandas as pd

from analysis.paper_figures import fig_agreement, fig_selfvote, fig_where


def _summary():
    rng = np.random.default_rng(0)
    cols = ["p_agree_given_self", "p_agree_given_no_self", "agree_share", "clip_ece", "tda_ece", "noself_ece"]
    cols += [f"{k}_{m}_{p}" for k in ("tda", "base") for m in ("acc", "gap") for p in ("agree", "disagree")]
    cols += [f"{k}_{m}" for k in ("clip", "base", "gpos", "gate", "tda") for m in ("acc", "ece")]
    cols = list(dict.fromkeys(cols))
    t = pd.DataFrame(rng.uniform(1, 90, (2, len(cols))), columns=cols)
    t.insert(0, "dataset", ["dtd", "pets"])
    mean = t.drop(columns="dataset").mean().to_frame().T.astype(float)
    mean.insert(0, "dataset", "mean")
    return pd.concat([t, mean], ignore_index=True)


def test_figures_draw_from_their_tables(tmp_path):
    s = _summary()
    fig_selfvote(s, tmp_path / "a.png")
    fig_agreement(s, tmp_path / "b.png")
    helps = pd.DataFrame({"dataset": ["dtd", "cars", "pets"], "group": ["test", "near", "far"],
                          "conf": [50.0, 60.0, 85.0], "clip_gap": [8.0, -1.0, -3.0],
                          "d_ece": [-9.0, -4.0, 2.0]})
    rho = pd.DataFrame({"signal": ["conf", "clip_gap"], "datasets": ["all", "all"], "n": [3, 3], "rho": [1.0, -1.0]})
    fig_where(helps, rho, tmp_path / "d.png")
    fig_where(helps, rho, tmp_path / "d2.png", signal="conf")
    assert all((tmp_path / f"{k}.png").stat().st_size > 1000 for k in ("a", "b", "d", "d2"))
