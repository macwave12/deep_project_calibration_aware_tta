"""Seed statistics for replayed runs (seed policy, DECISIONS 2026-11-10).

For every (dataset, method) over the replayed seeds: mean, std (ddof=1) and MAE, the
mean absolute deviation from the seed mean. For every (method, reference) pair:
the paired-by-seed difference `method - reference` (mean, std, MAE = mean |d|) and
wins / ties / losses, where a win is higher accuracy or lower ECE. A `mean` dataset
row averages the datasets per seed first, over seeds that have every dataset.

Record layout as utils/replay_runner.py writes it: seed 0 in `<raw>/*.json`, seed k
in `<raw>/seed<k>/*.json`.

    python analysis/seed_stats.py                                   # headline, seeds 0 1 2
    python analysis/seed_stats.py --raw-dir replay_results/raw/ablation --name ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = {"accuracy": +1, "ece": -1}  # sign: +1 higher is better, -1 lower is better
MEAN_ROW = "mean"


def seed_dir(raw_dir: Path, seed: int) -> Path:
    return raw_dir if seed == 0 else raw_dir / f"seed{seed}"


def load_seed_records(raw_dir, seeds) -> pd.DataFrame:
    """One row per (dataset, method, seed), metrics in percent, unrounded."""
    raw_dir = Path(raw_dir)
    rows = []
    for seed in seeds:
        files = sorted(seed_dir(raw_dir, seed).glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no records for seed {seed} in {seed_dir(raw_dir, seed)}")
        for path in files:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["seed"] != seed:
                raise ValueError(f"{path} holds seed {record['seed']}, expected {seed}")
            rows.append({
                "dataset": record["dataset"], "method": record["method"], "seed": seed,
                "accuracy": record["accuracy"] * 100, "ece": record["ece"] * 100,
            })
    frame = pd.DataFrame(rows)
    dupes = frame.duplicated(["dataset", "method", "seed"])
    if dupes.any():
        raise ValueError(f"duplicate records: {frame[dupes].to_dict('records')}")
    return frame


def with_dataset_mean(frame: pd.DataFrame) -> pd.DataFrame:
    """Append `mean` rows: per (method, seed), the unweighted mean over all datasets."""
    n_datasets = frame["dataset"].nunique()
    grouped = frame.groupby(["method", "seed"])
    complete = grouped.filter(lambda g: g["dataset"].nunique() == n_datasets)
    if complete.empty:
        return frame
    means = complete.groupby(["method", "seed"], as_index=False)[list(METRICS)].mean()
    means.insert(0, "dataset", MEAN_ROW)
    return pd.concat([frame, means], ignore_index=True)


def method_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=False):
        for metric in METRICS:
            values = group[metric]
            rows.append({
                "dataset": dataset, "method": method, "metric": metric,
                "n": len(values), "seeds": " ".join(map(str, sorted(group["seed"]))),
                "mean": values.mean(), "std": values.std(ddof=1),
                "mae": (values - values.mean()).abs().mean(),
            })
    return pd.DataFrame(rows)


CI_MIN_SEEDS = 10  # DECISIONS 2026-11-10: no intervals with 3 seeds; 90% at wrap-up


def ci_halfwidth(diff: pd.Series, level: float = 0.90) -> float:
    """Half-width of the t-interval of the mean paired difference; NaN below CI_MIN_SEEDS."""
    n = len(diff)
    if n < CI_MIN_SEEDS:
        return float("nan")
    from scipy.stats import t as student_t
    return float(student_t.ppf(0.5 + level / 2, n - 1) * diff.std(ddof=1) / n ** 0.5)


def paired_stats(frame: pd.DataFrame, references) -> pd.DataFrame:
    rows = []
    for dataset, by_dataset in frame.groupby("dataset", sort=False):
        wide = {metric: by_dataset.pivot(index="seed", columns="method", values=metric)
                for metric in METRICS}
        methods = list(dict.fromkeys(by_dataset["method"]))
        for reference in references:
            if reference not in methods:
                continue
            for method in methods:
                if method == reference:
                    continue
                for metric, sign in METRICS.items():
                    pair = wide[metric][[method, reference]].dropna()
                    diff = pair[method] - pair[reference]
                    better = sign * diff
                    half = ci_halfwidth(diff)
                    rows.append({
                        "dataset": dataset, "method": method, "reference": reference,
                        "metric": metric, "n": len(diff),
                        "seeds": " ".join(map(str, diff.index)),
                        "mean_diff": diff.mean(), "std_diff": diff.std(ddof=1),
                        "mae_diff": diff.abs().mean(),
                        "ci90_lo": diff.mean() - half, "ci90_hi": diff.mean() + half,
                        "wins": int((better > 0).sum()), "ties": int((better == 0).sum()),
                        "losses": int((better < 0).sum()),
                    })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="replay_results/raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--reference", nargs="+", default=["tda", "clipzs"],
                        help="methods every other method is paired against")
    parser.add_argument("--name", default="headline", help="output file tag")
    parser.add_argument("--out-dir", default="results/analysis")
    args = parser.parse_args(argv)

    frame = with_dataset_mean(load_seed_records(args.raw_dir, args.seeds))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "methods": method_stats(frame),
        "paired": paired_stats(frame, args.reference),
    }
    paths = []
    for kind, table in outputs.items():
        path = out_dir / f"seed_stats_{args.name}_{kind}.csv"
        table.round(4).to_csv(path, index=False)
        paths.append(path)
        print(f"wrote {path} ({len(table)} rows)")
    return paths


if __name__ == "__main__":
    main()
