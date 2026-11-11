"""Appendix: P1 reproduces the published TDA numbers, and P1 vs P2 differ little.

Reads the published values from `results/analysis/literature_values.csv`, our GPU P1 run
(`results/summary.csv`, seed 0) and the replayed protocol table (`results/protocols.csv`,
seeds 0/1/2). Writes `results/analysis/p1_appendix_reproduction.csv` (accuracy, CLIP and
TDA) and `results/analysis/p1_appendix_p1_vs_p2.csv` (every method, P1 vs P2, acc and ECE).

    python analysis/p1_appendix.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATASETS = ["dtd", "flower102", "pets", "eurosat", "aircraft"]


def reproduction(literature: pd.DataFrame, summary: pd.DataFrame, protocols: pd.DataFrame) -> pd.DataFrame:
    lit = literature.set_index("key")["value"]
    acc = protocols[(protocols.metric == "accuracy")].set_index(["protocol", "dataset", "method"])
    rows = []
    for method in ("clipzs", "tda"):
        for d in DATASETS + ["mean"]:
            if d == "mean":
                bench = sum(float(lit[f"bench_vitb16_{method}_{x}_acc"]) for x in DATASETS) / len(DATASETS)
                paper = sum(float(lit[f"tda_paper_vitb16_{method}_{x}_acc"]) for x in DATASETS) / len(DATASETS)
                gpu = float(summary[summary.method == method]["accuracy"].mean())
            else:
                bench = float(lit[f"bench_vitb16_{method}_{d}_acc"])
                paper = float(lit[f"tda_paper_vitb16_{method}_{d}_acc"])
                gpu = float(summary[(summary.method == method) & (summary.dataset == d)]["accuracy"].iloc[0])
            p1 = acc.loc[("P1", d, method)]
            p2 = acc.loc[("P2", d, method)]
            rows.append({"method": method, "dataset": d, "benchmark_table14": bench,
                         "ours_gpu_p1_seed0": gpu, "delta_gpu_vs_benchmark": gpu - bench,
                         "ours_replay_p1_mean": p1["mean"], "ours_replay_p1_std": p1["std"],
                         "ours_replay_p2_mean": p2["mean"], "tda_paper_table3_per_dataset_templates": paper})
    return pd.DataFrame(rows)


def p1_vs_p2(protocols: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    wide = protocols.pivot_table(index=["dataset", "method", "metric"], columns="protocol",
                                 values=["mean", "std"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    diff = paired[paired.comparison == "P2_minus_P1"][["dataset", "method", "metric", "mean_diff", "wins", "n"]]
    out = wide.merge(diff, on=["dataset", "method", "metric"], how="left")
    out = out.rename(columns={"mean_diff": "p2_minus_p1", "wins": "seeds_p2_better"})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--literature", default="results/analysis/literature_values.csv")
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--protocols", default="results/protocols.csv")
    ap.add_argument("--paired", default="results/analysis/protocols_paired.csv")
    ap.add_argument("--out-dir", default="results/analysis")
    a = ap.parse_args(argv)
    lit, summ = pd.read_csv(a.literature), pd.read_csv(a.summary)
    prot, paired = pd.read_csv(a.protocols), pd.read_csv(a.paired)
    out = Path(a.out_dir)
    rep = reproduction(lit, summ, prot)
    cmp_ = p1_vs_p2(prot, paired)
    rep.round(3).to_csv(out / "p1_appendix_reproduction.csv", index=False)
    cmp_.round(3).to_csv(out / "p1_appendix_p1_vs_p2.csv", index=False)
    print(rep.round(2).to_string(index=False))
    m = cmp_[cmp_.dataset == "mean"]
    print(m.round(2).to_string(index=False))
    return rep, cmp_


if __name__ == "__main__":
    main()
