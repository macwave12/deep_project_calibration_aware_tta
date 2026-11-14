"""Compact figures for the course report and the deck, drawn only from tracked CSVs.

Val (diagnosis and design) and test (results) are kept apart: every figure says its split.
  fig_selfvote   (val)  results/analysis/story/summary_val.csv
  fig_agreement  (val)  results/analysis/story/summary_val.csv
  fig_where      (design test, near, far)  results/analysis/t16/where_it_helps.csv, where_it_helps_rho.csv

Text sizes come from the active rcParams: `PAPER_RC` for the report, `SLIDE_RC` for the deck (which renders its
own copies at slide size), so one drawing function serves both.

    python analysis/paper_figures.py  -> analysis/figures/paper_*.png (copied to report/figures/)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.story import C_BASE, C_CLIP, C_GATE, C_TDA, NAMES  # noqa: E402

FIG = REPO / "analysis" / "figures"
REPORT_FIG = REPO / "report" / "figures"
STORY = REPO / "results" / "analysis" / "story"
T16 = REPO / "results" / "analysis" / "t16"
C_READ_FIRST = "#f4a261"
# tiers by marker and grey level, so that colour keeps meaning a method (CLIP blue, TDA red, gate teal)
TIER_STYLE = {"test": ("#525252", "o", "design (test)"), "near": ("#bdbdbd", "s", "near"),
              "far": ("white", "^", "far")}
PAPER_RC = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
            "xtick.labelsize": 6.5, "ytick.labelsize": 6.5}
SLIDE_RC = {"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 14,
            "xtick.labelsize": 14, "ytick.labelsize": 14}


def _datasets(table):
    t = table[table.dataset != "mean"]
    return t, [NAMES.get(n, n) for n in t.dataset]


def _legend_above(ax, ncol):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, fontsize="small", frameon=False,
              handlelength=1.0, columnspacing=0.8)


def fig_selfvote(summary: pd.DataFrame, path, figsize=(3.6, 2.0), letters=True):
    t, labels = _datasets(summary)
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    w = 0.38
    ax[0].bar(x - w / 2, t.p_agree_given_self, w, color=C_TDA, label="with a self-vote")
    ax[0].bar(x + w / 2, t.p_agree_given_no_self, w, color=C_BASE, label="without")
    ax[0].set_title(("(a) " if letters else "") + "cache agrees with CLIP", pad=16 if letters else 30)
    _legend_above(ax[0], 2)
    w = 0.27
    for i, (k, c, lab) in enumerate((("clip", C_CLIP, "CLIP"), ("tda", C_TDA, "TDA"),
                                     ("noself", C_READ_FIRST, "read-first TDA"))):
        ax[1].bar(x + (i - 1) * w, t[f"{k}_ece"], w, color=c, label=lab)
    ax[1].set_ylabel("ECE (%)")
    ax[1].set_title(("(b) " if letters else "") + "ECE", pad=16 if letters else 30)
    _legend_above(ax[1], 3)
    ax[0].set_ylabel("% of samples")
    for a in ax:
        a.set_xticks(x, labels, rotation=40, ha="right")
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_agreement(summary: pd.DataFrame, path, figsize=(3.6, 2.2), letters=True):
    """Change from adding TDA's positive cache to b, on agreeing (gate closed) and disagreeing (gate open) samples."""
    t, labels = _datasets(summary)
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    w = 0.38
    share = {"agree": t.agree_share.mean(), "disagree": 100 - t.agree_share.mean()}
    for j, (metric, title) in enumerate((("acc", "accuracy change (pts)"), ("gap", "conf. $-$ acc. change (pts)"))):
        for i, (part, c, lab) in enumerate((("agree", C_TDA, "agreeing, gate closed"),
                                            ("disagree", C_GATE, "disagreeing, gate open"))):
            d = t[f"tda_{metric}_{part}"] - t[f"base_{metric}_{part}"]
            ax[j].bar(x + (i - 0.5) * w, d, w, color=c, label=f"{lab} ({share[part]:.0f}%)")
        ax[j].axhline(0, color="black", lw=0.6)
        ax[j].set_title((f"({'cd'[j]}) " if letters else "") + title)
        ax[j].set_xticks(x, labels, rotation=40, ha="right")
        ax[j].grid(axis="y", alpha=0.3)
    handles, names = ax[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize="small", handlelength=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# the sets the discussion names, with label offsets (points) that keep the crowd near the origin readable
WHERE_NAMED = {"pets": (-8, 6), "caltech101": (-14, -9), "food101": (-28, -9), "cars": (4, -6), "imagenet_r": (7, -3),
               "gtsrb": (4, 2)}
WHERE_LABEL = {"clip_gap": "CLIP confidence - accuracy (pts; uses labels)",
               "conf": "CLIP mean confidence, first 500 samples (%)"}


def fig_where(helps: pd.DataFrame, rho: pd.DataFrame, path, signal="clip_gap", figsize=(3.6, 2.6), offset_scale=1.0,
              label_size="x-small"):
    fig, ax = plt.subplots(figsize=figsize)
    size = 28 * offset_scale ** 2
    for g in [t for t in TIER_STYLE if t in set(helps.group)]:
        sub = helps[helps.group == g]
        color, marker, label = TIER_STYLE[g]
        ax.scatter(sub[signal], sub.d_ece, marker=marker, s=size, color=color, edgecolors="black", linewidths=0.6,
                   label=label)
    for _, r in helps[helps.dataset.isin(WHERE_NAMED)].iterrows():
        dx, dy = WHERE_NAMED[r.dataset]
        ax.annotate(NAMES.get(r.dataset, r.dataset), (r[signal], r.d_ece), textcoords="offset points",
                    xytext=(dx * offset_scale, dy * offset_scale), fontsize=label_size)
    r_all = rho[(rho.signal == signal) & (rho.datasets == "all")].iloc[0]
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(WHERE_LABEL[signal])
    if signal == "clip_gap":
        ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("ECE(gate) - ECE(TDA) (pts)\nbelow 0: the gate is better")
    ax.set_title(f"Spearman rho = {r_all.rho:+.2f} over {int(r_all.n)} datasets")
    ax.legend(loc="upper right", fontsize="small")
    ax.margins(y=0.14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    val = pd.read_csv(STORY / "summary_val.csv")
    out = {
        "paper_selfvote_val.png": lambda p: fig_selfvote(val, p),
        "paper_agreement_val.png": lambda p: fig_agreement(val, p),
        "paper_where_all.png": lambda p: fig_where(pd.read_csv(T16 / "where_it_helps.csv"),
                                                   pd.read_csv(T16 / "where_it_helps_rho.csv"), p),
    }
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        for name, draw in out.items():
            draw(FIG / name)
            shutil.copy(FIG / name, REPORT_FIG / name)
    print("wrote", *out)


if __name__ == "__main__":
    main()
