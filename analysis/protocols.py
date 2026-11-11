"""T07: protocols P1 (published per-dataset alpha/beta) and P2 (clean, alpha=2 beta=5).

Reads replayed records (seed layout as analysis/seed_stats.py); a `_clean` suffix on
the record's method marks P2. clipzs has no alpha/beta, so its P1 records stand for
both protocols. Writes

- `results/protocols.csv`: per protocol x dataset x method x metric, mean/std/MAE over seeds;
- `results/analysis/protocols_paired.csv`: paired-by-seed differences, `comparison` =
  `within_P1` / `within_P2` (method - reference) or `P2_minus_P1` (same method);

and prints the headline-protocol rule (DECISIONS 2026-11-07): P2 is the headline iff
P2-TDA beats zero-shot CLIP accuracy, read on the dataset mean over seeds.

    python analysis/protocols.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.seed_stats import (  # noqa: E402
    load_seed_records, method_stats, paired_stats, with_dataset_mean,
)

METHODS = ["clipzs", "tda", "cal_tda_agree", "cal_tda_gate", "cal_tda_temp", "cal_tda_gate_temp",
           "cal_tda_margin", "cal_tda_fusion"]
PROTOCOL_DEPENDENT = [m for m in METHODS if m != "clipzs"]
CLEAN_SUFFIX = "_clean"


def split_protocols(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `protocol` (P1/P2), strip the suffix, keep T07's methods, copy clipzs to P2."""
    frame = frame.copy()
    clean = frame["method"].str.endswith(CLEAN_SUFFIX)
    frame["protocol"] = clean.map({True: "P2", False: "P1"})
    frame.loc[clean, "method"] = frame.loc[clean, "method"].str[: -len(CLEAN_SUFFIX)]
    frame = frame[frame["method"].isin(METHODS)]
    clip_p2 = frame[(frame["method"] == "clipzs") & (frame["protocol"] == "P1")].assign(protocol="P2")
    return pd.concat([frame, clip_p2], ignore_index=True)


def protocol_tables(frame: pd.DataFrame, references=("tda", "clipzs")):
    stats, paired = [], []
    by_protocol = {}
    for protocol in ["P1", "P2"]:
        part = with_dataset_mean(frame[frame["protocol"] == protocol].drop(columns="protocol"))
        by_protocol[protocol] = part
        stats.append(method_stats(part).assign(protocol=protocol))
        paired.append(paired_stats(part, list(references)).assign(comparison=f"within_{protocol}"))
    for method in PROTOCOL_DEPENDENT:
        both = pd.concat([by_protocol[p][by_protocol[p]["method"] == method].assign(method=p)
                          for p in ["P1", "P2"]], ignore_index=True)
        if both["method"].nunique() < 2:
            continue
        diff = paired_stats(both, ["P1"]).assign(method=method, reference=method,
                                                 comparison="P2_minus_P1")
        paired.append(diff)
    stats = pd.concat(stats, ignore_index=True)
    stats = stats[["protocol"] + [c for c in stats.columns if c != "protocol"]]
    paired = pd.concat(paired, ignore_index=True)
    paired = paired[["comparison"] + [c for c in paired.columns if c != "comparison"]]
    return stats, paired


def headline_rule(paired: pd.DataFrame) -> dict:
    row = paired.query(
        "comparison == 'within_P2' and dataset == 'mean' and method == 'tda' "
        "and reference == 'clipzs' and metric == 'accuracy'"
    ).iloc[0]
    return {"p2_tda_minus_clip_acc": row["mean_diff"], "wins": int(row["wins"]), "n": int(row["n"]),
            "headline": "P2" if row["mean_diff"] > 0 else "P1"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="replay_results/raw")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out-csv", default="results/protocols.csv")
    parser.add_argument("--paired-csv", default="results/analysis/protocols_paired.csv")
    args = parser.parse_args(argv)

    frame = split_protocols(load_seed_records(args.raw_dir, args.seeds))
    stats, paired = protocol_tables(frame)
    for table, path in [(stats, args.out_csv), (paired, args.paired_csv)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(path, index=False)
        print(f"wrote {path} ({len(table)} rows)")
    rule = headline_rule(paired)
    print(f"P2-TDA minus CLIP, dataset-mean accuracy: {rule['p2_tda_minus_clip_acc']:+.2f} "
          f"({rule['wins']}/{rule['n']} seeds) -> headline {rule['headline']}")
    return rule


if __name__ == "__main__":
    main()
