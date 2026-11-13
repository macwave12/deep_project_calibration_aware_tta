"""Confidence switch: keep TDA's echo on easy streams, gate it elsewhere (exact trace re-evaluation, P2, seeds 0/1/2).

The gate removes the positive cache where it agrees with CLIP, because there it only adds confidence (Lemma 2). Adding
confidence is right where CLIP is underconfident, which for zero-shot CLIP tends to happen on easy streams. The switch
keeps TDA's logits while the running mean of CLIP's max-softmax confidence over the samples seen so far (current one
included) exceeds a threshold, and uses the gate otherwise. Online and label-free. The threshold is chosen on the design
tiers (val splits of the five originals + the four development sets: three confident sets, six not); test and fresh
are held out.

    python analysis/switch.py   -> results/analysis/t16/switch{,_per_seed,_sweep}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.adaptive_alpha import disagree_mask, load_runs, metrics, trace_path  # noqa: E402
from analysis.comparison import zaec  # noqa: E402
from utils.feature_cache import DATASETS, DEV_DATASETS, FRESH_DATASETS  # noqa: E402

ALPHA = 2.0
OUT = REPO / "results" / "analysis" / "t16"
GRID = tuple(np.round(np.arange(0.60, 0.951, 0.01), 2))
METHODS = ("TDA", "agreement gate", "TDA + ZAEC", "confidence switch")
DESIGN = ("val", "dev")


def running_confidence(clip_logits: torch.Tensor) -> torch.Tensor:
    """Mean max-softmax confidence over samples 0..t, for every t (past and current only)."""
    conf = clip_logits.softmax(-1).max(-1).values
    return conf.cumsum(0) / torch.arange(1, len(conf) + 1, dtype=conf.dtype)


def switch_logits(run, threshold: float, alpha: float = ALPHA) -> torch.Tensor:
    tda = run.base + alpha * run.cache
    gate = torch.where(disagree_mask(run)[:, None], tda, run.base)
    keep = running_confidence(run.clip) > threshold
    return torch.where(keep[:, None], tda, gate)


def method_logits(run, threshold: float) -> dict:
    tda = run.base + ALPHA * run.cache
    return {"TDA": tda, "agreement gate": switch_logits(run, float("inf")),
            "TDA + ZAEC": zaec(tda, run.clip), "confidence switch": switch_logits(run, threshold)}


def sweep_rows(runs, tier: str, grid=GRID) -> pd.DataFrame:
    rows = []
    for r in runs:
        for t in grid:
            m = metrics(switch_logits(r, t).softmax(-1), r.labels)
            rows.append({"tier": tier, "dataset": r.dataset, "seed": r.seed, "threshold": t,
                         "accuracy": m["acc"], "ece": m["ece"]})
    return pd.DataFrame(rows)


def mean_over_datasets(frame: pd.DataFrame, by) -> pd.DataFrame:
    """Unweighted mean over datasets per seed, then mean over seeds."""
    return (frame.groupby([*by, "seed"])[["accuracy", "ece"]].mean()
            .groupby(by).mean().reset_index())


PLATEAU = 0.1  # ECE points


def plateau(val_sweep: pd.DataFrame, width: float = PLATEAU) -> pd.Series:
    """Thresholds whose design ECE (mean over datasets, then seeds) is within `width` of the minimum."""
    m = mean_over_datasets(val_sweep, ["threshold"])
    return m[m.ece <= m.ece.min() + width].threshold


def select_threshold(val_sweep: pd.DataFrame, width: float = PLATEAU) -> float:
    """Centre of the design plateau, not the argmin.

    Design ECE is flat over a wide range (no design dataset's confidence falls inside it) and its minimum sits next to
    a cliff where a dataset stops switching on, so the centre is the robust choice.
    """
    return round(float(plateau(val_sweep, width).mean()), 2)


def method_rows(runs, tier: str, threshold: float) -> pd.DataFrame:
    rows = []
    for r in runs:
        rc = running_confidence(r.clip)
        share = 100 * (rc > threshold).double().mean().item()
        clip_acc = 100 * (r.clip.argmax(-1) == r.labels).double().mean().item()  # label-based diagnostic only
        for name, lg in method_logits(r, threshold).items():
            m = metrics(lg.softmax(-1), r.labels)
            rows.append({"tier": tier, "dataset": r.dataset, "seed": r.seed, "method": name,
                         "accuracy": m["acc"], "ece": m["ece"], "switch_on_pct": share,
                         "clip_conf": 100 * rc[-1].item(), "clip_gap": clip_acc - 100 * rc[-1].item()})
    return pd.DataFrame(rows)


def tiers(a) -> dict:
    return {"val": (a.trace_root, DATASETS, "val"), "test": (a.trace_root, DATASETS, "test"),
            "dev": (a.trace_root, DEV_DATASETS, "test"), "fresh": (a.fresh_root, FRESH_DATASETS, "test")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace-root", default="replay_results/t14")
    ap.add_argument("--fresh-root", default="replay_results/t16_fresh")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)
    runs, skipped = {}, []
    for tier, (root, datasets, split) in tiers(a).items():
        runs[tier] = []
        for ds in datasets:
            if all(trace_path(root, ds, s, split, "tda_eq_clean").exists() for s in a.seeds):
                runs[tier] += load_runs(root, [ds], a.seeds, split, "tda_eq_clean")
            else:
                skipped.append(ds)
    sweep = pd.concat([sweep_rows(rs, t) for t, rs in runs.items()], ignore_index=True)
    design = sweep[sweep.tier.isin(DESIGN)]
    threshold, near = select_threshold(design), plateau(design)
    frame = pd.concat([method_rows(rs, t, threshold) for t, rs in runs.items()], ignore_index=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(out / "switch_per_seed.csv", index=False)
    mean_over_datasets(sweep, ["tier", "threshold"]).round(3).to_csv(out / "switch_sweep.csv", index=False)
    per_ds = (frame.groupby(["tier", "dataset", "method"])[["accuracy", "ece", "switch_on_pct", "clip_conf", "clip_gap"]]
              .mean().reset_index())
    means = mean_over_datasets(frame, ["tier", "method"]).assign(dataset="mean")
    table = pd.concat([per_ds, means], ignore_index=True).assign(
        threshold=threshold, plateau_lo=near.min(), plateau_hi=near.max())
    table.round(3).to_csv(out / "switch.csv", index=False)
    print(f"threshold (design: val + dev) = {threshold}; skipped (no full trace): {', '.join(skipped) or 'none'}")
    print(means.pivot(index="method", columns="tier", values="ece")[["val", "test", "dev", "fresh"]].round(2))
    print(means.pivot(index="method", columns="tier", values="accuracy")[["val", "test", "dev", "fresh"]].round(2))


if __name__ == "__main__":
    main()
