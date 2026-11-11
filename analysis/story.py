"""The paper's story figures and tables (T16), from `tda_eq_clean` full traces.

The diagnosis (self-vote, unsure vs confident CLIP, agreement vs disagreement) is drawn for the
val splits, where it was found, and repeated on the test and dev splits. The test splits also
get the cache ablation (the TDA paper's Fig. 3 with ECE) and reliability diagrams. Every
dataset gets the "where it helps" relationship. Everything is P2 (alpha = 2, beta = 5), seeds
0/1/2, `replay_results/t14/`. All variants are exact re-evaluations of one TDA trace (the caches
do not depend on alpha or on each other); the one exception is the self-vote removal, which
subtracts 1 (the self affinity) at CLIP's argmax and is approximate under fp16 emulation.

    python analysis/story.py   -> analysis/figures/story_*.png, results/analysis/story/*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.adaptive_alpha import load_runs  # noqa: E402
from utils.calibration import compute_ece, reliability_curve  # noqa: E402
from utils.feature_cache import DATASETS, DEV_DATASETS  # noqa: E402

OUT = REPO / "results" / "analysis" / "story"
RELIABILITY = ("clip", "tda", "gate")
VARIANTS = {  # name -> label, as shown in the figures
    "clip": "CLIP", "base": "CLIP + negative cache", "pos": "CLIP + positive cache",
    "tda": "TDA (both caches)", "noself": "TDA without self-vote", "gate": "agreement gate",
}

ALPHA = 2.0
FIG = REPO / "analysis" / "figures"
NAMES = {"dtd": "DTD", "flower102": "Flowers", "pets": "Pets", "eurosat": "EuroSAT", "aircraft": "Aircraft",
         "imagenetv2": "ImageNetV2", "caltech101": "Caltech101", "ucf101": "UCF101", "food101": "Food101"}
C_CLIP, C_BASE, C_TDA, C_GATE = "#4c78a8", "#8d99ae", "#e45756", "#2a9d8f"


def derive(r) -> dict:
    """Per-sample arrays (label-free signals + correctness for analysis)."""
    y = r.labels
    p0 = r.clip.softmax(-1)
    pred0 = p0.argmax(-1)
    has = r.cache.clamp_min(0).sum(-1) > 0
    agree = has & (r.cache.argmax(-1) == pred0)
    o_tda = r.base + ALPHA * r.cache
    c_ns = r.cache.clone()
    sv = r.self_vote
    c_ns[sv, pred0[sv]] -= 1.0
    o_ns = r.base + ALPHA * c_ns.clamp_min(0)
    o_gate = r.base + (ALPHA * (has & ~agree).double())[:, None] * r.cache
    top2 = r.clip.topk(2, -1).values
    n = r.clip - r.base
    safe = (top2[:, 0] - top2[:, 1]) > ALPHA * r.cache.max(-1).values + (n.max(-1).values - n.min(-1).values)
    out = {"y": y, "pred0": pred0, "agree": agree, "self": sv, "safe": safe,
           "clip_conf": p0.max(-1).values}
    o_pos = r.clip + ALPHA * r.cache
    for k, o in (("clip", r.clip), ("base", r.base), ("pos", o_pos), ("tda", o_tda), ("noself", o_ns),
                 ("gate", o_gate)):
        p = o.softmax(-1)
        out[f"{k}_pred"], out[f"{k}_conf"] = p.argmax(-1), p.max(-1).values
        out[f"{k}_ece"] = 100 * compute_ece(p.numpy(), y.numpy())
        out[f"{k}_acc"] = 100 * float((p.argmax(-1) == y).double().mean())
        if k in RELIABILITY:
            out[f"{k}_rel"] = reliability_curve(p.numpy(), y.numpy(), n_bins=10)
    return out


def per_dataset(runs) -> dict:
    by = {}
    for r in runs:
        by.setdefault(r.dataset, []).append(derive(r))
    return by


def mean_over_seeds(ds, f):
    return float(np.mean([f(d) for d in ds]))


def gap(d, k, mask):
    return 100 * float((d[f"{k}_conf"][mask] - (d[f"{k}_pred"][mask] == d["y"][mask]).double()).mean())


def acc(d, k, mask):
    return 100 * float((d[f"{k}_pred"][mask] == d["y"][mask]).double().mean())


# --- Step 1: self-vote -----------------------------------------------------------------------
def fig_step1(val, path, split="val"):
    names = list(val)
    x = np.arange(len(names))
    share = [mean_over_seeds(val[n], lambda d: 100 * float(d["self"].double().mean())) for n in names]
    pa_s = [mean_over_seeds(val[n], lambda d: 100 * float(d["agree"][d["self"]].double().mean())) for n in names]
    pa_n = [mean_over_seeds(val[n], lambda d: 100 * float(d["agree"][~d["self"]].double().mean())) for n in names]
    ece = {k: [mean_over_seeds(val[n], lambda d, k=k: d[f"{k}_ece"]) for n in names] for k in ("clip", "tda", "noself")}
    accs = {k: [mean_over_seeds(val[n], lambda d, k=k: d[f"{k}_acc"]) for n in names] for k in ("tda", "noself")}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].bar(x, share, color=C_TDA)
    ax[0].set_title("a. Share of samples that vote for themselves")
    ax[0].set_ylabel("% of stream")
    w = 0.38
    ax[1].bar(x - w / 2, pa_s, w, color=C_TDA, label="with self-vote")
    ax[1].bar(x + w / 2, pa_n, w, color=C_BASE, label="without")
    ax[1].set_title("b. P(cache argmax = CLIP argmax)")
    ax[1].legend(fontsize=8)
    w = 0.27
    ax[2].bar(x - w, ece["clip"], w, color=C_CLIP, label="CLIP")
    ax[2].bar(x, ece["tda"], w, color=C_TDA, label="TDA")
    ax[2].bar(x + w, ece["noself"], w, color="#f4a261", label="TDA without self-vote")
    for i in range(len(names)):
        d_acc = accs["noself"][i] - accs["tda"][i]
        ax[2].text(x[i] + w, ece["noself"][i] + 0.3, f"{d_acc:+.1f}", ha="center", fontsize=7)
    ax[2].set_title("c. ECE (label: accuracy change without self-vote)")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x, [NAMES[n] for n in names], fontsize=8)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"The self-vote ({split}, P2): frequent, almost always agrees with CLIP, and costs calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- Step 2: confident CLIP cannot be changed (the echo figure, P2) --------------------------
def fig_step2(val, path, split="val", bins=10):
    names = list(val)
    fig, ax = plt.subplots(3, len(names), figsize=(3.4 * len(names), 8.4), sharex="col")
    for j, n in enumerate(names):
        rows = []
        for d in val[n]:
            q = pd.qcut(d["clip_conf"].numpy(), bins, labels=False, duplicates="drop")
            for b in np.unique(q):
                m = torch.as_tensor(q == b)
                c_ok = d["base_pred"][m] == d["y"][m]
                t_ok = d["tda_pred"][m] == d["y"][m]
                rows.append({"bin": b, "conf": float(d["clip_conf"][m].mean()),
                             "net_fixed": 100 * float((t_ok & ~c_ok).double().mean() - (~t_ok & c_ok).double().mean()),
                             "gap_clip": gap(d, "clip", m), "gap_tda": gap(d, "tda", m),
                             "safe": 100 * float(d["safe"][m].double().mean())})
        t = pd.DataFrame(rows).groupby("bin").mean()
        ax[0, j].bar(t["conf"], t["net_fixed"], width=0.03, color=C_GATE)
        ax[0, j].axhline(0, color="black", lw=0.6)
        ax[1, j].plot(t["conf"], t["gap_clip"], "o-", color=C_CLIP, label="CLIP")
        ax[1, j].plot(t["conf"], t["gap_tda"], "s-", color=C_TDA, label="TDA")
        ax[1, j].axhline(0, color="black", lw=0.6)
        ax[2, j].plot(t["conf"], t["safe"], "o-", color="#6d597a")
        ax[2, j].set_ylim(-3, 103)
        ax[2, j].set_xlabel("CLIP confidence (bin mean)")
        ax[0, j].set_title(NAMES[n])
        for i in range(3):
            ax[i, j].grid(alpha=0.3)
    ax[0, 0].set_ylabel("net decisions fixed\nby the cache (% of bin)")
    ax[1, 0].set_ylabel("confidence - accuracy\n(pts)")
    ax[1, 0].legend(fontsize=8)
    ax[2, 0].set_ylabel("no-flip region\n(% of bin, Lemma 2)")
    fig.suptitle("The cache fixes decisions only when CLIP is unsure, but inflates confidence in most bins "
                 f"({split}, P2, 10 equal-count bins)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- Step 3: agreement vs disagreement -------------------------------------------------------
def fig_step3(val, path, split="val"):
    names = list(val)
    x = np.arange(len(names))
    fig, ax = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True)
    for col, (label, sel) in enumerate((("cache agrees with CLIP", lambda d: d["agree"]),
                                        ("cache disagrees", lambda d: ~d["agree"]))):
        share = [mean_over_seeds(val[n], lambda d: 100 * float(sel(d).double().mean())) for n in names]
        w = 0.38
        for k, color, off in (("base", C_BASE, -w / 2), ("tda", C_TDA, w / 2)):
            a = [mean_over_seeds(val[n], lambda d, k=k: acc(d, k, sel(d))) for n in names]
            g = [mean_over_seeds(val[n], lambda d, k=k: gap(d, k, sel(d))) for n in names]
            lab = "CLIP + negative cache" if k == "base" else "TDA"
            ax[0, col].bar(x + off, a, w, color=color, label=lab)
            ax[1, col].bar(x + off, g, w, color=color, label=lab)
        ax[0, col].set_title(f"{label}  (share: " + ", ".join(f"{s:.0f}%" for s in share) + ")", fontsize=9)
        ax[1, col].axhline(0, color="black", lw=0.6)
        ax[1, col].set_xticks(x, [NAMES[n] for n in names], fontsize=8)
        for i in range(2):
            ax[i, col].grid(axis="y", alpha=0.3)
    ax[0, 0].set_ylabel("accuracy (%)")
    ax[1, 0].set_ylabel("confidence - accuracy (pts)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Where the cache agrees it adds no accuracy, only confidence; "
                 f"all of TDA's gain is where it disagrees ({split}, P2)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- Step 5: where it helps most (exploratory relationship; DECISIONS 2026-11-11) -------------
SHOT_CAPACITY = 3


def worst_case_safe(r) -> torch.Tensor:
    """No flip possible whatever the cache holds: CLIP gap > alpha * S + range(n) (cache-independent)."""
    top2 = r.clip.topk(2, -1).values
    n = r.clip - r.base
    return (top2[:, 0] - top2[:, 1]) > ALPHA * SHOT_CAPACITY + (n.max(-1).values - n.min(-1).values)


CONFIDENCE_SIGNALS = (
    ("share", "worst-case no-flip share (%)"),
    ("over06", "share with CLIP confidence > 0.6 (%)"),
    ("conf", "CLIP mean confidence (%)"),
)


def confidence_signals(runs, head=500) -> pd.DataFrame:
    """Label-free CLIP-confidence signals on the first `head` stream samples, mean over seeds."""
    rows = []
    for r in runs:
        conf = r.clip.softmax(-1).max(-1).values[:head]
        rows.append({"dataset": r.dataset,
                     "share": 100 * float(worst_case_safe(r)[:head].double().mean()),
                     "over06": 100 * float((conf > 0.6).double().mean()),
                     "conf": 100 * float(conf.mean())})
    return pd.DataFrame(rows).groupby("dataset").mean()


def fig_step5(runs, data_by_group, path):
    from scipy.stats import spearmanr
    sig = confidence_signals(runs)
    dece = {n: mean_over_seeds(ds, lambda d: d["gate_ece"] - d["tda_ece"])
            for data in data_by_group.values() for n, ds in data.items()}
    names = list(dece)
    fig, axes = plt.subplots(1, len(CONFIDENCE_SIGNALS), figsize=(16, 4.6), sharey=True)
    for ax, (key, label) in zip(axes, CONFIDENCE_SIGNALS):
        for (group, data), marker in zip(data_by_group.items(), ("o", "s", "D")):
            for n in data:
                x, y = sig.loc[n, key], dece[n]
                ax.plot(x, y, marker, ms=9, color=C_GATE if y < 0 else C_TDA, mec="black")
                ax.annotate(NAMES.get(n, n), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        rho = spearmanr([sig.loc[n, key] for n in names], [dece[n] for n in names]).correlation
        ax.set_title(f"Spearman rho = {rho:+.2f}  (n = {len(names)})", fontsize=10)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel(label + ", first 500 samples")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ECE(gate) - ECE(TDA)  (pts; < 0: gate better)")
    groups = ", ".join(f"{g} {m}" for g, m in zip(data_by_group, ("o", "[]", "<>")))
    fig.suptitle(f"Where the gate helps most: where CLIP is not confident (exploratory; {groups}; P2, 3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- Cache ablation (TDA paper Fig. 3, with ECE) --------------------------------------------
ABLATION = ("clip", "base", "pos", "tda", "gate")


def summary_table(data) -> pd.DataFrame:
    """Per dataset (mean over seeds) and the unweighted dataset mean: accuracy and ECE of every
    variant, the self-vote and agreement shares, and the agreeing / disagreeing split."""
    rows = []
    for n, ds in data.items():
        row = {"dataset": n}
        for k in VARIANTS:
            row[f"{k}_acc"] = mean_over_seeds(ds, lambda d, k=k: d[f"{k}_acc"])
            row[f"{k}_ece"] = mean_over_seeds(ds, lambda d, k=k: d[f"{k}_ece"])
        row["self_vote_share"] = mean_over_seeds(ds, lambda d: 100 * float(d["self"].double().mean()))
        row["agree_share"] = mean_over_seeds(ds, lambda d: 100 * float(d["agree"].double().mean()))
        row["p_agree_given_self"] = mean_over_seeds(ds, lambda d: 100 * float(d["agree"][d["self"]].double().mean()))
        row["p_agree_given_no_self"] = mean_over_seeds(ds, lambda d: 100 * float(d["agree"][~d["self"]].double().mean()))
        for part, sel in (("agree", lambda d: d["agree"]), ("disagree", lambda d: ~d["agree"])):
            for k in ("base", "tda"):
                row[f"{k}_acc_{part}"] = mean_over_seeds(ds, lambda d, k=k: acc(d, k, sel(d)))
                row[f"{k}_gap_{part}"] = mean_over_seeds(ds, lambda d, k=k: gap(d, k, sel(d)))
        rows.append(row)
    frame = pd.DataFrame(rows)
    mean = frame.drop(columns="dataset").mean().to_frame().T
    mean.insert(0, "dataset", "mean")
    return pd.concat([frame, mean], ignore_index=True)


def fig_cache_ablation(table, path, split="test"):
    names = list(table["dataset"])
    x = np.arange(len(names))
    colors = {"clip": C_CLIP, "base": C_BASE, "pos": "#f4a261", "tda": C_TDA, "gate": C_GATE}
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.9))
    w = 0.16
    for i, k in enumerate(ABLATION):
        for j, metric in enumerate(("acc", "ece")):
            ax[j].bar(x + (i - 2) * w, table[f"{k}_{metric}"], w, color=colors[k], label=VARIANTS[k])
    ax[0].set_title("accuracy (%)")
    ax[1].set_title("ECE (%, lower is better)")
    for a in ax:
        a.set_xticks(x, [NAMES.get(n, n) for n in names], fontsize=8)
        a.grid(axis="y", alpha=0.3)
    ax[0].set_ylim(max(0, float(table[[f"{k}_acc" for k in ABLATION]].min().min()) - 10), None)
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Which cache does what ({split}, P2): the positive cache brings the gain and all the "
                 "overconfidence; the negative cache helps both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_reliability(data, path, split="test"):
    """Reliability diagrams (10 bins, seeds pooled) of CLIP, TDA and the gate, one panel per dataset."""
    names = list(data)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.5))
    for ax, n in zip(np.atleast_1d(axes), names):
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
        for k, color in (("clip", C_CLIP), ("tda", C_TDA), ("gate", C_GATE)):
            conf = np.zeros(10); acc_ = np.zeros(10); cnt = np.zeros(10)
            for d in data[n]:
                rel = d[f"{k}_rel"]
                c = np.asarray(rel["bin_count"], dtype=float)
                conf += np.asarray(rel["bin_conf"]) * c
                acc_ += np.asarray(rel["bin_acc"]) * c
                cnt += c
            m = cnt > 0
            ax.plot(conf[m] / cnt[m], acc_[m] / cnt[m], "o-", color=color, ms=4, label=VARIANTS[k])
        ax.set_title(NAMES.get(n, n))
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f"Reliability ({split}, P2, seeds pooled): below the diagonal = overconfident")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace-root", default="replay_results/t14")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {
        "val": load_runs(a.trace_root, DATASETS, a.seeds, "val", "tda_eq_clean"),
        "test": load_runs(a.trace_root, DATASETS, a.seeds, "test", "tda_eq_clean"),
        "dev": load_runs(a.trace_root, DEV_DATASETS, a.seeds, "test", "tda_eq_clean"),
    }
    data = {g: per_dataset(runs) for g, runs in groups.items()}
    for split, d in data.items():
        table = summary_table(d)
        table.round(3).to_csv(OUT / f"summary_{split}.csv", index=False)
        fig_step1(d, FIG / f"story_selfvote_{split}.png", split)
        fig_step2(d, FIG / f"story_echo_{split}.png", split)
        fig_step3(d, FIG / f"story_agreement_{split}.png", split)
        fig_cache_ablation(table, FIG / f"story_cache_ablation_{split}.png", split)
    fig_reliability(data["test"], FIG / "story_reliability_test.png", "test")
    signals = confidence_signals(groups["test"] + groups["dev"])
    signals.round(3).to_csv(OUT / "confidence_signals.csv")
    fig_step5(groups["test"] + groups["dev"], {"test": data["test"], "dev": data["dev"]},
              FIG / "story_where_it_helps.png")
    print("wrote", *sorted(p.name for p in FIG.glob("story_*.png")), "and", OUT)


if __name__ == "__main__":
    main()
