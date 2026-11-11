"""T14 results on the test splits: the agreement gate against TDA's own accuracy-ECE curve.

Idan's comparison rule (DECISIONS 2026-11-10): a method beats TDA only if it improves both
accuracy and ECE, or at matched accuracy / matched ECE. So besides the run records
(`analysis/protocols.py`), this script places every method on TDA's constant-alpha curve,
re-evaluated exactly from a `--trace full` TDA replay, and adds "TDA + one temperature"
fitted on data other than the test sets (val splits, or the dev sets).

    python analysis/t14_results.py [--fit-on val|dev]
Outputs: results/analysis/t14/{alpha_curve,matched,points}_test.csv, analysis/figures/t14_acc_ece_test.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.adaptive_alpha import disagree_mask, load_runs, metrics  # noqa: E402
from utils.feature_cache import DATASETS, DEV_DATASETS  # noqa: E402

CURVE_ALPHAS = [round(a, 2) for a in np.arange(0.0, 4.001, 0.25)]
OUT = REPO / "results" / "analysis" / "t14"
FIG = REPO / "analysis" / "figures" / "t14_acc_ece_test.png"
RECORD_METHODS = {"clipzs": "CLIP", "cal_tda_gate": "old gate (ties)", "cal_tda_temp": "LOO temperature"}


def fit_temperature(runs, alpha=2.0) -> float:
    """One global T for TDA (alpha=2), equal weight per dataset, by NLL (grid, then refine)."""
    def loss(t):
        per = {}
        for r in runs:
            lp = ((r.base + alpha * r.cache) / t).log_softmax(-1)
            per.setdefault(r.dataset, []).append(float(-lp.gather(1, r.labels[:, None]).mean()))
        return float(np.mean([np.mean(v) for v in per.values()]))
    grid = np.exp(np.linspace(math.log(0.5), math.log(4), 61))
    best = min(grid, key=loss)
    fine = np.exp(np.linspace(math.log(best) - 0.04, math.log(best) + 0.04, 21))
    return float(min(fine, key=loss))


def curve_rows(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        for a in CURVE_ALPHAS:
            rows.append({"dataset": r.dataset, "seed": r.seed, "alpha": a,
                         **metrics((r.base + a * r.cache).softmax(-1), r.labels)})
    return pd.DataFrame(rows)


def point_rows(runs, temperature) -> pd.DataFrame:
    rows = []
    for r in runs:
        dis = disagree_mask(r).double()
        pts = {
            "TDA": r.base + 2 * r.cache,
            f"TDA + T ({temperature:.2f})": (r.base + 2 * r.cache) / temperature,
            "CLIP + negative cache": r.base,
            "agreement gate (ours)": r.base + (2 * dis)[:, None] * r.cache,
        }
        for name, lg in pts.items():
            rows.append({"dataset": r.dataset, "seed": r.seed, "method": name,
                         "gate_open": float(dis.mean()), **metrics(lg.softmax(-1), r.labels)})
    return pd.DataFrame(rows)


def with_mean(frame, keys):
    frame = frame.drop(columns=["seed"], errors="ignore")
    per = frame.groupby(keys + ["dataset"], as_index=False).mean(numeric_only=True)
    mean = per.groupby(keys, as_index=False).mean(numeric_only=True)
    mean["dataset"] = "mean"
    return pd.concat([per, mean], ignore_index=True)


def matched(curve: pd.DataFrame, points: pd.DataFrame, method: str) -> pd.DataFrame:
    """TDA's best ECE at the method's accuracy (interpolated on the seed-mean alpha-curve)
    and TDA's best accuracy at the method's ECE. NaN when the curve never reaches it."""
    rows = []
    for dataset, c in curve.groupby("dataset"):
        c = c.sort_values("alpha")
        p = points[(points.method == method) & (points.dataset == dataset)].iloc[0]
        acc, ece = c["acc"].to_numpy(), c["ece"].to_numpy()
        ece_at, acc_at = [], []
        for i in range(len(c) - 1):
            for x, y, target, out in ((acc, ece, p["acc"], ece_at), (ece, acc, p["ece"], acc_at)):
                lo, hi = sorted((x[i], x[i + 1]))
                if lo <= target <= hi and hi > lo:
                    w = (target - x[i]) / (x[i + 1] - x[i])
                    out.append(y[i] + w * (y[i + 1] - y[i]))
        rows.append({"dataset": dataset, "method": method, "acc": p["acc"], "ece": p["ece"],
                     "tda_best_ece_at_same_acc": min(ece_at) if ece_at else np.nan,
                     "tda_best_acc_at_same_ece": max(acc_at) if acc_at else np.nan,
                     "tda_curve_max_acc": acc.max(), "tda_curve_min_ece": ece.min()})
    return pd.DataFrame(rows)


def figure(curve, points, records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(DATASETS) + ["mean"]
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 3.4))
    style = {"TDA": ("#d1495b", "s"), "CLIP + negative cache": ("#8d99ae", "v"),
             "agreement gate (ours)": ("#2a9d8f", "*")}
    for ax, ds in zip(axes, order):
        c = curve[curve.dataset == ds].sort_values("alpha")
        ax.plot(c["acc"], c["ece"], "-", color="#d1495b", alpha=0.5, lw=1.5, label="TDA, α from 0 to 4")
        for _, p in points[points.dataset == ds].iterrows():
            color, marker = style.get(p["method"], ("#edae49", "D"))
            ax.plot(p["acc"], p["ece"], marker, color=color, ms=12 if marker == "*" else 7,
                    label=p["method"], mec="black", mew=0.5)
        for _, rr in records[records.dataset == ds].iterrows():
            marker = {"CLIP": "o", "old gate (ties)": "P", "LOO temperature": "X"}[rr["method"]]
            ax.plot(rr["acc"], rr["ece"], marker, color="#555555", ms=7, label=rr["method"], mfc="white")
        ax.set_title(ds)
        ax.set_xlabel("accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ECE (%), lower is better")
    handles, labels = axes[-1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=len(uniq), fontsize=8, frameon=False)
    fig.suptitle("Test splits, P2, seeds 0/1/2: methods against TDA's own accuracy-ECE curve")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def mechanism_rows(runs) -> pd.DataFrame:
    """Why the gate helps or hurts: confidence - accuracy gap (sign = over/underconfidence),
    gate-open share, accuracy on agreeing vs disagreeing samples, decisions fixed/broken
    relative to CLIP + negative cache (alpha = 0)."""
    rows = []
    for r in runs:
        dis = disagree_mask(r)
        y = r.labels
        base_pred = r.base.argmax(-1)
        for name, lg in (("CLIP", r.clip), ("CLIP + negative cache", r.base),
                         ("TDA", r.base + 2 * r.cache),
                         ("agreement gate (ours)", r.base + (2 * dis.double())[:, None] * r.cache)):
            p = lg.softmax(-1)
            pred, conf = p.argmax(-1), p.max(-1).values
            correct = (pred == y).double()
            rows.append({
                "dataset": r.dataset, "seed": r.seed, "method": name,
                "acc": 100 * float(correct.mean()), "mean_conf": 100 * float(conf.mean()),
                "gap": 100 * float(conf.mean() - correct.mean()),
                "gap_agree": 100 * float((conf - correct)[~dis].mean()),
                "gap_disagree": 100 * float((conf - correct)[dis].mean()) if dis.any() else np.nan,
                "gate_open": 100 * float(dis.double().mean()),
                "acc_agree": 100 * float(correct[~dis].mean()),
                "acc_disagree": 100 * float(correct[dis].mean()) if dis.any() else np.nan,
                "fixed_vs_base": 100 * float(((pred == y) & (base_pred != y)).double().mean()),
                "broken_vs_base": 100 * float(((pred != y) & (base_pred == y)).double().mean()),
            })
    return pd.DataFrame(rows)


def cmd_mechanism(trace_root, datasets, seeds, split, features, out_name):
    runs = load_runs(trace_root, datasets, seeds, split, "tda_eq_clean", features)
    frame = with_mean(mechanism_rows(runs), ["method"])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.round(3).to_csv(OUT / out_name, index=False)
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace-root", default="replay_results/t14")
    ap.add_argument("--features", default="features")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fit-on", choices=["val", "dev"], default="val",
                    help="data the TDA + T reference temperature is fitted on (never test)")
    ap.add_argument("--protocols-csv", default="results/protocols.csv")
    ap.add_argument("--mechanism-only", choices=["test", "dev"], default=None,
                    help="only write the mechanism table for the test splits or the dev sets")
    args = ap.parse_args(argv)
    if args.mechanism_only:
        ds = DATASETS if args.mechanism_only == "test" else DEV_DATASETS
        frame = cmd_mechanism(args.trace_root, ds, args.seeds, "test", args.features,
                              f"mechanism_{args.mechanism_only}.csv")
        cols = ["dataset", "method", "acc", "gap", "gap_agree", "gap_disagree", "gate_open",
                "acc_agree", "acc_disagree", "fixed_vs_base", "broken_vs_base"]
        print(frame[cols].round(2).to_string(index=False))
        return

    test = load_runs(args.trace_root, DATASETS, args.seeds, "test", "tda_eq_clean", args.features)
    fit = (load_runs(args.trace_root, DATASETS, args.seeds, "val", "tda_eq_clean", args.features)
           if args.fit_on == "val" else
           load_runs(args.trace_root, DEV_DATASETS, args.seeds, "test", "tda_eq_clean", args.features))
    T = fit_temperature(fit)
    curve = with_mean(curve_rows(test), ["alpha"])
    points = with_mean(point_rows(test, T), ["method"])
    recs = pd.read_csv(args.protocols_csv)
    recs = recs[(recs.protocol == "P2") & recs.method.isin(RECORD_METHODS)]
    records = recs.pivot_table(index=["dataset", "method"], columns="metric", values="mean").reset_index()
    records = records.rename(columns={"accuracy": "acc"}).assign(method=lambda d: d.method.map(RECORD_METHODS))
    all_points = pd.concat([points[["dataset", "method", "acc", "ece"]], records[["dataset", "method", "acc", "ece"]]])
    match = pd.concat([matched(curve, all_points, m) for m in all_points.method.unique()
                       if m != "TDA"], ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    curve.round(4).to_csv(OUT / "alpha_curve_test.csv", index=False)
    points.round(4).to_csv(OUT / "points_test.csv", index=False)
    match.round(4).to_csv(OUT / "matched_test.csv", index=False)
    figure(curve, points, records, FIG)
    print(f"TDA + T reference: T = {T:.3f} fitted on {args.fit_on}")
    print(points[points.dataset == "mean"][["method", "acc", "ece", "nll", "gate_open"]].round(3).to_string(index=False))
    print(match[match.dataset == "mean"].round(3).to_string(index=False))
    print(f"wrote {OUT} and {FIG}")


if __name__ == "__main__":
    main()
