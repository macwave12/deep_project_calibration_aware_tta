"""T02: replay single-view methods on CPU from the T01 feature cache.

One run (same flags as scripts/run_one.*, plus split/protocol):

    python scripts/replay.py --dataset dtd --method cal_tda --config configs/method/cal_tda_gate.yaml \
        --run-name cal_tda_gate [--protocol clean] [--seed 1] [--split val]

A whole suite, mirroring scripts/run_all.ps1 / run_ablation*.ps1 record names:

    python scripts/replay.py --suite headline        # -> replay_results/raw/
    python scripts/replay.py --suite ablation        # -> replay_results/raw/ablation/
    python scripts/replay.py --suite ablation_gate   # -> replay_results/raw/ablation_gate/

then compare with the GPU records:  python analysis/compare_replay.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from online_method.tda import PROTOCOLS  # noqa: E402
from online_tta import PROTOCOL_DEPENDENT_ALGORITHMS  # noqa: E402
from utils.feature_cache import ALL_DATASETS, DATASETS  # noqa: E402
from utils.feature_replay import REPLAYABLE_ALGORITHMS  # noqa: E402

# (algorithm, config, run_name), as scripts/run_all.ps1 runs them.
HEADLINE = [
    ("clipzs", None, None),
    ("tda", "configs/method/tda.yaml", None),
    ("cal_tda", "configs/method/cal_tda_gate_temp.yaml", "cal_tda_gate_temp"),
    ("cal_tda", "configs/method/cal_tda_gate.yaml", "cal_tda_gate"),
    ("cal_tda", "configs/method/cal_tda_temp.yaml", "cal_tda_temp"),
]
ABLATION = ["none", "margin", "fusion", "temp", "margin_fusion", "margin_temp", "fusion_temp", "all"]
ABLATION_GATE = ["none", "gate", "temp", "gate_temp"]


def suite_runs(suite: str):
    """[(algorithm, config, run_name, out_subdir)] for a named suite."""
    if suite == "headline":
        return [(a, c, n, "") for a, c, n in HEADLINE]
    if suite == "ablation":
        return [("cal_tda", f"configs/method/ablation/{v}.yaml", v, "ablation") for v in ABLATION]
    if suite == "ablation_gate":
        return [("cal_tda", f"configs/method/ablation_gate/{v}.yaml", v, "ablation_gate") for v in ABLATION_GATE]
    raise ValueError(suite)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", default="features", help="T01 output folder (default: features/)")
    p.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=ALL_DATASETS)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--seed", nargs="+", type=int, default=[0])
    p.add_argument("--method", choices=REPLAYABLE_ALGORITHMS, help="single run: the --algorithm")
    p.add_argument("--config", default=None, help="single run: method YAML (as online_tta --config)")
    p.add_argument("--run-name", default=None, help="single run: record name (default: method)")
    p.add_argument("--suite", choices=["headline", "ablation", "ablation_gate"])
    p.add_argument("--protocol", default="published", choices=PROTOCOLS)
    p.add_argument("--out-dir", default="replay_results/raw")
    p.add_argument("--autocast", default="emulate", choices=["emulate", "off"],
                   help="emulate CUDA autocast's casts (default, matches the GPU runs) or run plain fp32")
    p.add_argument("--limit", type=int, default=None, help="debug: only the first N stream positions")
    p.add_argument("--trace", default="off", choices=["off", "top", "full"],
                   help="cal_tda only: write the label-free trace (top-k fields; full adds whole z/n/c vectors)")
    args = p.parse_args(argv)
    if (args.method is None) == (args.suite is None):
        p.error("give exactly one of --method or --suite")
    return args


def main(argv=None):
    args = parse_args(argv)
    from utils.replay_runner import replay_one

    if args.suite:
        runs = suite_runs(args.suite)
        if args.protocol == "clean":
            # clipzs has no alpha/beta: its P2 record would duplicate its P1 one.
            runs = [(a, c, f"{n or a}_clean", d) for a, c, n, d in runs
                    if a in PROTOCOL_DEPENDENT_ALGORITHMS]
    else:
        runs = [(args.method, args.config, args.run_name, "")]

    started = time.time()
    paths = []
    for dataset in args.dataset:
        for seed in args.seed:
            for algorithm, config, run_name, subdir in runs:
                paths.append(replay_one(
                    args.features, dataset, algorithm=algorithm, split=args.split, seed=seed,
                    config=config, protocol=args.protocol, run_name=run_name,
                    out_dir=Path(args.out_dir) / subdir, autocast=args.autocast, limit=args.limit,
                    trace=args.trace,
                ))
    print(f"{len(paths)} replay run(s) in {time.time() - started:.1f}s")
    return paths


if __name__ == "__main__":
    main()
