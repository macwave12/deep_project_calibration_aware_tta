"""A11 — where does TDA's cache term help, and where does it only inflate confidence?

Replays TDA (CalTDA with every flag off, which reproduces upstream TDA) from the T01
feature cache with the T04 trace on, then, per CLIP-confidence bin:

- decisions: flip rate, wrong->right, right->wrong, net gain per sample;
- calibration: CLIP vs TDA accuracy, mean confidence, ECE contribution;
- the cache's push on CLIP's top-2 gap (delta_gap = alpha*(c1-c2) - (n1-n2)) and how much
  of it is an "echo" of CLIP's own gap (linear fit of delta_gap on CLIP's gap).

Development data is the CoOp **val** split (decision 2026-11-10: design on val, apply to
test once). Labels are joined here, after the stream, never inside the method.

Usage: python analysis/echo_analysis.py [--split val] [--seeds 0 1 2] [--bins 10]
Outputs: results/analysis/echo/{bins,flips,summary}_<split>.csv, analysis/figures/echo_<split>.png
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from online_tta import load_method_config  # noqa: E402
from utils.feature_cache import DATASETS, UPSTREAM_SET_ID, FeatureStore  # noqa: E402
from utils.feature_replay import (FeatureReplayModel, build_trainer,  # noqa: E402
                                  cuda_autocast_emulation, run_stream)
from utils.tools import set_random_seed  # noqa: E402


def trace_tda(features_root: Path, dataset: str, split: str, seed: int) -> dict:
    """Run TDA-equivalent CalTDA with the trace on; return per-step arrays (label-free)."""
    store = FeatureStore(features_root, dataset)
    order = store.permutation(split, seed)
    feats = store.image_features(split)
    text = store.text()
    model = FeatureReplayModel.from_text_dict(feats, text)
    cfg = yaml.safe_load((REPO / "configs/method/tda_equivalent.yaml").read_text())
    cfg["trace"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh)
        cfg_path = fh.name
    args = SimpleNamespace(algorithm="cal_tda", config=cfg_path, test_sets=UPSTREAM_SET_ID[dataset],
                           protocol="published", batch_size=1, gpu=None, loader_len=len(order),
                           seed=seed, run_name=None, arch="ViT-B/16")
    method_config = load_method_config(args)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        warnings.filterwarnings("ignore", category=FutureWarning)
        set_random_seed(seed)
        trainer = build_trainer("cal_tda", model, method_config)
        trainer.prepare_model_and_optimization(args)
        with torch.no_grad(), cuda_autocast_emulation(True):
            run_stream(trainer, model, order, args)
    rows = {k: torch.stack(v).numpy() for k, v in trainer.trace._rows.items()}
    # Full CLIP distribution for the stream (needed for CLIP's own confidence).
    f = feats[torch.as_tensor(order)].float()
    tf = text["text_features"].float() if isinstance(text, dict) else text.float()
    scale = float(text.get("logit_scale", 100.0)) if isinstance(text, dict) else 100.0
    clip_probs = (scale * f @ tf.t()).softmax(-1).numpy()
    rows.update(order=np.asarray(order), clip_probs=clip_probs, alpha=float(trainer.config.alpha),
                beta=float(trainer.config.beta))
    return rows


def per_sample(rows: dict, labels: np.ndarray) -> pd.DataFrame:
    y = labels[rows["order"]]
    z = rows["clip_topk_logit"]          # (N, k) CLIP top-k logits, descending
    idx = rows["clip_topk_index"]
    c = rows["cache_topk"]               # unscaled cache logits on those classes
    n = rows["neg_topk"]                 # negative-cache logits (already weighted)
    alpha = rows["alpha"]
    p0 = rows["clip_probs"]
    yhat0 = idx[:, 0]
    yhat = rows["out_argmax"]
    return pd.DataFrame({
        "clip_conf": p0.max(1),
        "clip_gap": z[:, 0] - z[:, 1],
        "cache_push": alpha * (c[:, 0] - c[:, 1]) - (n[:, 0] - n[:, 1]),
        "cache_max": rows["cache_max"],
        "cache_agrees": rows["cache_argmax"] == yhat0,
        "self_vote": rows["in_pos_cache"].astype(bool),
        "clip_correct": yhat0 == y,
        "tda_correct": yhat == y,
        "tda_conf": rows["out_topk_prob"][:, 0],
        "flip": yhat != yhat0,
    })


def bin_table(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = pd.qcut(df["clip_conf"].rank(method="first"), bins, labels=False)
    N = len(df)
    g = df.groupby("bin")
    out = pd.DataFrame({
        "n": g.size(),
        "clip_conf_lo": g["clip_conf"].min(), "clip_conf_hi": g["clip_conf"].max(),
        "clip_acc": g["clip_correct"].mean(), "clip_conf": g["clip_conf"].mean(),
        "tda_acc": g["tda_correct"].mean(), "tda_conf": g["tda_conf"].mean(),
        "flip_rate": g["flip"].mean(),
        "wrong_to_right": g.apply(lambda d: (d.flip & d.tda_correct & ~d.clip_correct).mean()),
        "right_to_wrong": g.apply(lambda d: (d.flip & ~d.tda_correct & d.clip_correct).mean()),
        "cache_push": g["cache_push"].mean(),
        "self_vote_rate": g["self_vote"].mean(),
        "cache_agrees": g["cache_agrees"].mean(),
    })
    out["net_gain"] = out["wrong_to_right"] - out["right_to_wrong"]
    out["clip_gap_err"] = (out["clip_conf"] - out["clip_acc"])
    out["tda_gap_err"] = (out["tda_conf"] - out["tda_acc"])
    # Contribution of each bin to a confidence-mean-gap ECE proxy (weighted |conf-acc|).
    out["clip_ece_part"] = out["clip_gap_err"].abs() * out["n"] / N
    out["tda_ece_part"] = out["tda_gap_err"].abs() * out["n"] / N
    return out.reset_index()


def echo_fit(df: pd.DataFrame) -> dict:
    """How much of the cache's push on CLIP's top-2 gap is a function of that gap?"""
    x, yv = df["clip_gap"].to_numpy(), df["cache_push"].to_numpy()
    b, a = np.polyfit(x, yv, 1)
    resid = yv - (a + b * x)
    r2 = 1 - resid.var() / yv.var() if yv.var() > 0 else float("nan")
    return {"echo_slope": b, "echo_intercept": a, "echo_r2": r2,
            "corr_gap_push": float(np.corrcoef(x, yv)[0, 1]), "resid": resid}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", default=str(REPO / "features"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "results/analysis/echo"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bins_all, flips_all, summary = [], [], []
    for ds in DATASETS:
        labels = FeatureStore(Path(a.features), ds).labels(a.split).numpy()
        for seed in a.seeds:
            rows = trace_tda(Path(a.features), ds, a.split, seed)
            df = per_sample(rows, labels)
            fit = echo_fit(df)
            df["echo_resid"] = fit.pop("resid")
            bt = bin_table(df, a.bins)
            bt.insert(0, "seed", seed)
            bt.insert(0, "dataset", ds)
            bins_all.append(bt)
            fl = df[df.flip].assign(dataset=ds, seed=seed,
                                    kind=np.where(df[df.flip].tda_correct, "fixed", "broken"))
            flips_all.append(fl)
            summary.append({"dataset": ds, "seed": seed, "n": len(df), "alpha": rows["alpha"],
                            "beta": rows["beta"], "clip_acc": df.clip_correct.mean(),
                            "tda_acc": df.tda_correct.mean(), "flip_rate": df.flip.mean(),
                            "self_vote_rate": df.self_vote.mean(), **fit})
            print(f"[{ds}/{a.split}/seed{seed}] n={len(df)} clip={df.clip_correct.mean():.3f} "
                  f"tda={df.tda_correct.mean():.3f} echo_r2={fit['echo_r2']:.2f}", flush=True)
    pd.concat(bins_all).to_csv(out / f"bins_{a.split}.csv", index=False)
    pd.concat(flips_all).to_csv(out / f"flips_{a.split}.csv", index=False)
    pd.DataFrame(summary).to_csv(out / f"summary_{a.split}.csv", index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
