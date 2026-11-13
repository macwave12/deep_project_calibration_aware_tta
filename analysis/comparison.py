"""T16 comparison table on the five test sets, P2, seeds 0/1/2; TDA is the reference of every paired delta.

Rows: CLIP; TDA; the agreement gate; read-before-write TDA; TDA + one global temperature fitted on
the val splits (labels used for the fit only, val only); TDA + ZAEC and gate + ZAEC (post-hoc,
label-free); DMN-weak and ECALP (their own replays). The trace rows are exact re-evaluations of
`--trace full` replays (`analysis/adaptive_alpha.py`); DMN-weak and ECALP come from their run records.

ZAEC (Jiang et al., arXiv 2608.05945): per sample, if the adapted prediction has lower entropy than
zero-shot CLIP's, divide its logits by the smallest T >= 1 that restores CLIP's entropy (bisection);
otherwise leave it. Ranking-preserving, so accuracy is unchanged.

    python analysis/comparison.py   -> results/analysis/t16/comparison{,_paired,_per_dataset}.csv
    python analysis/comparison.py --generalization   -> results/analysis/t16/zaec_generalization*.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.adaptive_alpha import disagree_mask, load_runs, metrics, trace_path  # noqa: E402
from analysis.seed_stats import MEAN_ROW, METRICS, method_stats, paired_stats, seed_dir, with_dataset_mean  # noqa: E402
from analysis.t14_results import fit_temperature  # noqa: E402
from utils.feature_cache import DATASETS, DEV_DATASETS, FRESH_DATASETS  # noqa: E402

ALPHA = 2.0
OUT = REPO / "results" / "analysis" / "t16"
ORDER = ("CLIP", "TDA", "agreement gate", "read-first TDA", "TDA + T (val)", "TDA + ZAEC", "gate + ZAEC",
         "DMN-weak", "ECALP")
RECORD_METHODS = {"dmn_weak": "DMN-weak", "ecalp": "ECALP"}


def entropy(logits: torch.Tensor) -> torch.Tensor:
    lp = logits.log_softmax(-1)
    return -(lp.exp() * lp).sum(-1)


def zaec(adapted: torch.Tensor, zero_shot: torch.Tensor, iters: int = 60) -> torch.Tensor:
    """Per-sample temperature-scaled `adapted` logits whose entropy is at least zero-shot's.

    Entropy of softmax(o / T) is non-decreasing in T and tends to log C, so the smallest T >= 1
    with H(o / T) >= H(zero_shot) exists whenever H(zero_shot) < log C; bisection finds it.
    """
    target = entropy(zero_shot)
    need = entropy(adapted) < target
    lo = torch.ones(adapted.shape[0], dtype=adapted.dtype)
    hi = torch.ones_like(lo)
    for _ in range(64):  # grow hi until it brackets the target
        short = need & (entropy(adapted / hi[:, None]) < target)
        if not short.any():
            break
        hi = torch.where(short, hi * 2, hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        low = entropy(adapted / mid[:, None]) < target
        lo = torch.where(need & low, mid, lo)
        hi = torch.where(need & ~low, mid, hi)
    t = torch.where(need, hi, torch.ones_like(hi))
    return adapted / t[:, None]


def trace_logits(run, read_first=None, temperature=1.0) -> dict:
    tda = run.base + ALPHA * run.cache
    gate = run.base + (ALPHA * disagree_mask(run).double())[:, None] * run.cache
    out = {"CLIP": run.clip, "TDA": tda, "agreement gate": gate, "TDA + T (val)": tda / temperature,
           "TDA + ZAEC": zaec(tda, run.clip), "gate + ZAEC": zaec(gate, run.clip)}
    if read_first is not None:
        out["read-first TDA"] = read_first.base + ALPHA * read_first.cache
    return out


def trace_rows(runs, read_first, temperature) -> pd.DataFrame:
    rf = {(r.dataset, r.seed): r for r in read_first}
    rows = []
    for r in runs:
        for method, lg in trace_logits(r, rf.get((r.dataset, r.seed)), temperature).items():
            m = metrics(lg.softmax(-1), r.labels)
            rows.append({"dataset": r.dataset, "method": method, "seed": r.seed,
                         "accuracy": m["acc"], "ece": m["ece"]})
    return pd.DataFrame(rows)


def record_rows(raw_dir, datasets, seeds) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for ds in datasets:
            for key, name in RECORD_METHODS.items():
                path = seed_dir(Path(raw_dir), seed) / f"{ds}_{key}.json"
                if not path.exists():
                    continue
                rec = json.loads(path.read_text(encoding="utf-8"))
                if rec["seed"] != seed or rec["dataset"] != ds:
                    raise ValueError(f"{path}: holds {rec['dataset']} seed {rec['seed']}")
                rows.append({"dataset": ds, "method": name, "seed": seed,
                             "accuracy": 100 * rec["accuracy"], "ece": 100 * rec["ece"]})
    return pd.DataFrame(rows, columns=["dataset", "method", "seed", *METRICS])


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Wide table: mean and std over seeds per (dataset, method); methods in ORDER."""
    stats = method_stats(frame).pivot_table(index=["dataset", "method"], columns="metric",
                                            values=["mean", "std", "n"])
    stats.columns = [f"{m}_{s}" for s, m in stats.columns]
    stats = stats.reset_index()
    stats["order"] = stats["method"].map({m: i for i, m in enumerate(ORDER)})
    return stats.sort_values(["dataset", "order"]).drop(columns="order")[
        ["dataset", "method", "accuracy_n", "accuracy_mean", "accuracy_std", "ece_mean", "ece_std"]]


GEN_METHODS = ("TDA", "agreement gate", "TDA + ZAEC", "gate + ZAEC")


def generalization_rows(runs, tier: str) -> pd.DataFrame:
    """ZAEC vs the gate on sets never used in the design; no temperature, no read-first."""
    rows = []
    for r in runs:
        lg = trace_logits(r)
        for method in GEN_METHODS:
            m = metrics(lg[method].softmax(-1), r.labels)
            rows.append({"tier": tier, "dataset": r.dataset, "method": method, "seed": r.seed,
                         "accuracy": m["acc"], "ece": m["ece"]})
    return pd.DataFrame(rows)


def run_generalization(a) -> None:
    """Dev traces (t14) and fresh traces (t16_fresh); fresh sets without full traces are skipped."""
    tiers = {"dev": (a.trace_root, DEV_DATASETS), "fresh": (a.fresh_root, FRESH_DATASETS)}
    frames, skipped = [], []
    for tier, (root, datasets) in tiers.items():
        for ds in datasets:
            if not all(trace_path(root, ds, s, "test", "tda_eq_clean").exists() for s in a.seeds):
                skipped.append(ds)
                continue
            frames.append(generalization_rows(load_runs(root, [ds], a.seeds, "test", "tda_eq_clean"), tier))
    frame = pd.concat(frames, ignore_index=True)
    means = [with_dataset_mean(g.drop(columns="tier")).query("dataset == @MEAN_ROW").assign(tier=t)
             for t, g in frame.groupby("tier")]
    means.append(with_dataset_mean(frame.drop(columns="tier")).query("dataset == @MEAN_ROW").assign(tier="all"))
    frame = pd.concat([frame, *means], ignore_index=True)
    out = Path(a.out_dir)
    frame.round(4).to_csv(out / "zaec_generalization_per_seed.csv", index=False)
    paired = pd.concat([paired_stats(g.drop(columns="tier"), ["TDA + ZAEC", "TDA"]).assign(tier=t)
                        for t, g in frame.groupby("tier")], ignore_index=True)
    paired.round(4).to_csv(out / "zaec_generalization_paired.csv", index=False)
    print(f"skipped (no full trace): {', '.join(skipped) or 'none'}")
    ece = paired[(paired.metric == "ece") & (paired.reference == "TDA + ZAEC")]
    print(ece[["tier", "dataset", "method", "mean_diff", "wins", "losses"]].round(2).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace-root", default="replay_results/t14")
    ap.add_argument("--fresh-root", default="replay_results/t16_fresh")
    ap.add_argument("--baselines", default="replay_results/t16_baselines")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--generalization", action="store_true",
                    help="ZAEC vs gate on the dev and fresh sets -> zaec_generalization*.csv")
    a = ap.parse_args(argv)
    if a.generalization:
        run_generalization(a)
        return
    temperature = fit_temperature(load_runs(a.trace_root, DATASETS, a.seeds, "val", "tda_eq_clean"), ALPHA)
    runs = load_runs(a.trace_root, DATASETS, a.seeds, "test", "tda_eq_clean")
    rf = load_runs(a.trace_root, DATASETS, a.seeds, "test", "tda_read_first_clean")
    frame = pd.concat([trace_rows(runs, rf, temperature), record_rows(a.baselines, DATASETS, a.seeds)],
                      ignore_index=True)
    # The mean row needs every dataset for a (method, seed); incomplete baselines get none.
    frame = with_dataset_mean(frame)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(out / "comparison_per_seed.csv", index=False)
    summary(frame).round(3).to_csv(out / "comparison.csv", index=False)
    paired_stats(frame, ["TDA"]).round(4).to_csv(out / "comparison_paired.csv", index=False)
    print(f"T (val) = {temperature:.3f}; wrote comparison*.csv to {out}")
    tab = summary(frame)
    print(tab[tab.dataset == "mean"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
