"""T14 dev-set confirmation: CLIP, TDA and the agreement gate on the four development sets.

Nothing is fitted on these sets (DECISIONS 2026-11-10: method frozen, parameter-free); they
are an independent check on datasets the method was never designed on. Records come from
`scripts/replay_dev.sh` (`replay_results/t14/`); TDA is `tda_equivalent` (== upstream TDA).

    python analysis/t14_dev.py
Outputs: results/analysis/t14/dev_protocols.csv, results/analysis/t14/dev_paired.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.protocols import protocol_tables, split_protocols  # noqa: E402
from analysis.seed_stats import load_seed_records  # noqa: E402
from utils.feature_cache import DEV_DATASETS  # noqa: E402


def dev_frame(raw_dir, seeds, datasets=DEV_DATASETS):
    frame = load_seed_records(raw_dir, seeds)
    frame = frame[frame.dataset.isin(datasets)].copy()
    frame["method"] = frame["method"].str.replace(r"^tda_eq", "tda", regex=True)
    return split_protocols(frame)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default="replay_results/t14")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--datasets", nargs="+", default=list(DEV_DATASETS))
    ap.add_argument("--out", default="results/analysis/t14")
    a = ap.parse_args(argv)
    stats, paired = protocol_tables(dev_frame(a.raw_dir, a.seeds, a.datasets))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats.round(4).to_csv(out / "dev_protocols.csv", index=False)
    paired.round(4).to_csv(out / "dev_paired.csv", index=False)
    s = stats.assign(v=stats["mean"].round(2).astype(str) + " ± " + stats["std"].round(2).astype(str))
    print(s.pivot_table(index=["protocol", "dataset", "method"], columns="metric", values="v",
                        aggfunc="first").to_string())
    q = paired[(paired.method == "cal_tda_agree") & (paired.reference == "tda")]
    print(q[["comparison", "dataset", "metric", "mean_diff", "wins", "ties", "losses"]].round(2).to_string(index=False))
    return stats, paired


if __name__ == "__main__":
    main()
