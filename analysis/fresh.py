"""Held-out generalization (T16, T17 tiers) and "where it helps".

Held-out sets: the T16 fresh sets (`scripts/replay_fresh.sh` -> `replay_results/t16_fresh`) and the T17 shifted sets
(`replay_results/t17_shift`); ImageNet and ImageNetV2 are left out (`EXCLUDED`: TDA's hyperparameter-search
distribution). Each set is tagged near / far / other by `held_tier` (what the images are, fixed before results).
P2, seeds 0/1/2. Per dataset: CLIP, TDA and the agreement gate from their own online replays, and the paired-by-seed
delta vs TDA. A dataset enters only when TDA and the gate both have every requested seed.

"Where it helps" (DECISIONS 2026-11-11): dECE (gate - TDA, paired run records) per design-test, near and far dataset
against label-free CLIP-confidence signals on the first stream samples (from full traces) and against CLIP's own
confidence - accuracy gap (label-based: an explanation, never a method input). Spearman rho over all and per tier.

    python analysis/fresh.py  -> results/analysis/t16/fresh*.csv, where_it_helps*.csv,
                                 analysis/figures/story_where_it_helps_all.png
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.adaptive_alpha import trace_path  # noqa: E402
from analysis.seed_stats import load_seed_records, paired_stats, seed_dir  # noqa: E402
from analysis.story import FIG, plot_where_it_helps, signal_row, spearman_rows  # noqa: E402
from utils.feature_cache import DATASETS, EVAL_DEV, EXCLUDED, NEAR, FAR, held_tier, replay_root  # noqa: E402

HEAD_ONLY = ("imagenet_sketch",)  # only a 500-sample trace run (its full trace is too large); enough for head signals

OUT = REPO / "results" / "analysis" / "t16"
METHODS = {"clipzs": "CLIP", "tda_eq_clean": "TDA", "cal_tda_agree_clean": "agreement gate"}
SIGNALS = ["share", "over06", "conf", "clip_gap"]


def fresh_records(roots, seeds) -> pd.DataFrame:
    """Run records (percent) of CLIP, TDA and the gate, one row per (dataset, method, seed in `seeds`)."""
    frames = []
    for root in roots:
        present = [s for s in seeds if seed_dir(Path(root), s).is_dir()]
        frame = load_seed_records(root, present)
        frames.append(frame[frame["method"].isin(METHODS)])
    frame = pd.concat(frames, ignore_index=True)
    frame["method"] = frame["method"].map(METHODS)
    return frame


def complete(frame: pd.DataFrame, seeds) -> pd.DataFrame:
    """Keep datasets where TDA and the gate both have every seed and CLIP has a record; warn about the rest."""
    keep = []
    for ds, g in frame.groupby("dataset", sort=False):
        ok = (all(set(g[g.method == m].seed) >= set(seeds) for m in ("TDA", "agreement gate"))
              and (g.method == "CLIP").any())
        if ok:
            keep.append(ds)
        else:
            warnings.warn(f"{ds}: incomplete records, left out")
    return frame[frame.dataset.isin(keep)]


def fresh_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: CLIP (deterministic; seed mean), TDA and gate mean/std, paired delta gate - TDA."""
    paired = paired_stats(frame[frame.method != "CLIP"], ["TDA"])
    rows = []
    for ds, g in frame.groupby("dataset", sort=False):
        row = {"dataset": ds}
        for m, key in (("CLIP", "clip"), ("TDA", "tda"), ("agreement gate", "gate")):
            sel = g[g.method == m]
            row[f"{key}_n"] = len(sel)
            for metric in ("accuracy", "ece"):
                row[f"{key}_{metric}"] = sel[metric].mean()
                if key != "clip":
                    row[f"{key}_{metric}_std"] = sel[metric].std(ddof=1)
        for metric in ("accuracy", "ece"):
            p = paired[(paired.dataset == ds) & (paired.metric == metric)].iloc[0]
            row[f"d_{metric}"] = p["mean_diff"]
            row[f"{metric}_wins"] = f"{p['wins']}/{p['n']}"
        rows.append(row)
    table = pd.DataFrame(rows)
    mean = table.drop(columns=["dataset", "accuracy_wins", "ece_wins"]).mean(numeric_only=True).to_frame().T
    mean.insert(0, "dataset", "mean")
    return pd.concat([table, mean], ignore_index=True)


def record_dece(root, dataset, seeds, tda="tda_eq_clean", gate="cal_tda_agree_clean") -> float:
    """Mean over seeds of ECE(gate) - ECE(TDA) from the online run records, in points."""
    diffs = []
    for seed in seeds:
        ece = {}
        for name in (tda, gate):
            rec = json.loads((seed_dir(Path(root), seed) / f"{dataset}_{name}.json").read_text(encoding="utf-8"))
            if rec["seed"] != seed:
                raise ValueError(f"{dataset}_{name}: holds seed {rec['seed']}, expected {seed}")
            ece[name] = 100 * rec["ece"]
        diffs.append(ece[gate] - ece[tda])
    return sum(diffs) / len(diffs)


def head_signals(root, dataset, seeds, name, split="test", head=500) -> dict:
    """Signals from the first `head` steps of a full trace, mean over seeds."""
    rows = []
    for seed in seeds:
        st = torch.load(trace_path(root, dataset, seed, split, name), weights_only=True)["steps"]
        rows.append(signal_row(dataset, st["clip_logits"], st["base_logits"], head))
    return pd.DataFrame(rows).groupby("dataset").mean().iloc[0].to_dict()


def record_gaps(root, dataset, seeds, tda="tda_eq_clean", gate="cal_tda_agree_clean") -> dict:
    """Signed confidence - accuracy (pts) of CLIP, TDA and the gate: TDA and gate averaged over seeds, CLIP from seed 0
    (its output does not depend on the stream order). Positive: overconfident. Labels: a diagnostic only."""
    out = {"clip_gap": clip_gap(seed_dir(Path(root), 0) / f"{dataset}_clipzs.json")}
    for key, name in (("tda_gap", tda), ("gate_gap", gate)):
        vals = [clip_gap(seed_dir(Path(root), s) / f"{dataset}_{name}.json") for s in seeds]
        out[key] = sum(vals) / len(vals)
    return out


def clip_gap(record_path) -> float:
    """CLIP's mean confidence - accuracy (pts) over the whole stream, from its record's reliability bins.

    Uses labels: a diagnostic of why the gate helps or not, never an input to the method (RULES 2).
    """
    rel = json.loads(Path(record_path).read_text(encoding="utf-8"))["reliability"]
    count = torch.tensor(rel["bin_count"], dtype=torch.float64)
    conf = torch.tensor(rel["bin_conf"], dtype=torch.float64)
    acc = torch.tensor(rel["bin_acc"], dtype=torch.float64)
    return 100 * float(((conf - acc) * count).sum() / count.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--t14", default="replay_results/t14")
    ap.add_argument("--fresh", default="replay_results/t16_fresh")
    ap.add_argument("--shift", default="replay_results/t17_shift")
    ap.add_argument("--test-records", default="replay_results/raw")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = complete(fresh_records([a.fresh, a.shift], a.seeds), a.seeds)
    records = records[~records.dataset.isin(EXCLUDED)]
    records.round(4).to_csv(out / "fresh_per_seed.csv", index=False)
    tiers = records.dataset.map(held_tier)
    table = pd.concat([fresh_table(records[tiers == t]).assign(tier=t) for t in ("near", "far", "other")
                       if (tiers == t).any()], ignore_index=True)
    table.round(3).to_csv(out / "fresh.csv", index=False)

    groups = {**{d: "test" for d in DATASETS}, **{d: "near" for d in NEAR}, **{d: "far" for d in FAR}}
    sig, dece = {}, {}
    for ds, g in groups.items():
        held = replay_root(ds, a.t14, a.fresh, a.shift)
        if held != a.t14:
            dece[ds] = float(table[table.dataset == ds].iloc[0]["d_ece"])
            root = held
        else:
            root = a.test_records if g == "test" else a.t14
            dece[ds] = record_dece(root, ds, a.seeds, "tda_clean" if g == "test" else "tda_eq_clean")
        sig[ds] = {"clip_gap": clip_gap(Path(root) / f"{ds}_clipzs.json")}
        name = "tda_eq_clean_head500" if ds in HEAD_ONLY else "tda_eq_clean"
        sig[ds].update(head_signals(held, ds, a.seeds, name))
    gaps = []
    held_out = [d for d in table.dataset if d != "mean"] + [d for d in EVAL_DEV if d not in EXCLUDED]
    for ds, g in {**groups, **{d: held_tier(d) for d in dict.fromkeys(held_out) if d not in groups}}.items():
        root = a.test_records if g == "test" else replay_root(ds, a.t14, a.fresh, a.shift)
        gaps.append({"dataset": ds, "tier": g,
                     **record_gaps(root, ds, a.seeds, "tda_clean" if g == "test" else "tda_eq_clean")})
    pd.DataFrame(gaps).round(3).to_csv(out / "gaps.csv", index=False)
    sig = pd.DataFrame(sig).T
    helps = sig.assign(group=pd.Series(groups), d_ece=pd.Series(dece))
    helps.index.name = "dataset"
    helps.round(3).to_csv(out / "where_it_helps.csv")
    rho = pd.concat([spearman_rows(sig, dece, SIGNALS).assign(datasets="all")]
                    + [spearman_rows(sig, {k: v for k, v in dece.items() if groups[k] == t}, SIGNALS).assign(datasets=t)
                       for t in ("test", "near", "far")])
    rho.round(3).to_csv(out / "where_it_helps_rho.csv", index=False)
    plot_where_it_helps(sig, dece, groups, FIG / "story_where_it_helps_all.png")
    print(table.round(2).to_string(index=False))
    print(rho.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
