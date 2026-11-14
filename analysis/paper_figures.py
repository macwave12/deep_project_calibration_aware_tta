"""Compact figures for the course report and deck v3, drawn only from tracked CSVs.

Val (diagnosis and design) and test (results) are kept apart: every figure says its split.
  fig_selfvote   (val)  results/analysis/story/summary_val.csv
  fig_agreement  (val)  results/analysis/story/summary_val.csv
  fig_plane      (test) results/analysis/t16/comparison.csv
  fig_where      (all)  results/analysis/t16/where_it_helps.csv, where_it_helps_rho.csv
  fig_ablation   (test) results/analysis/story/summary_test.csv

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
C_OTHER = "#9c755f"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})


def _datasets(table):
    t = table[table.dataset != "mean"]
    return t, [NAMES.get(n, n) for n in t.dataset]


def fig_selfvote(summary: pd.DataFrame, path):
    t, labels = _datasets(summary)
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=(3.6, 2.0))
    w = 0.38
    ax[0].bar(x - w / 2, t.p_agree_given_self, w, color=C_TDA, label="with a self-vote")
    ax[0].bar(x + w / 2, t.p_agree_given_no_self, w, color=C_BASE, label="without")
    ax[0].set_ylabel("%")
    ax[0].set_title("(a) cache agrees with CLIP")
    ax[0].legend(loc="lower left")
    w = 0.27
    for i, (k, c, lab) in enumerate((("clip", C_CLIP, "CLIP"), ("tda", C_TDA, "TDA"),
                                     ("noself", "#f4a261", "TDA, self-vote removed"))):
        ax[1].bar(x + (i - 1) * w, t[f"{k}_ece"], w, color=c, label=lab)
    ax[1].set_ylabel("ECE (%)")
    ax[1].set_title("(b) ECE")
    ax[1].legend()
    for a in ax:
        a.set_xticks(x, labels, rotation=40, ha="right")
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_agreement(summary: pd.DataFrame, path):
    t, labels = _datasets(summary)
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=(3.6, 2.0))
    w = 0.38
    share = {"agree": t.agree_share.mean(), "disagree": 100 - t.agree_share.mean()}
    for j, (metric, title) in enumerate((("acc", "(c) accuracy change (pts)"),
                                         ("gap", "(d) conf. - acc. change (pts)"))):
        for i, (part, c, lab) in enumerate((("agree", C_TDA, "cache agrees"), ("disagree", C_GATE, "cache disagrees"))):
            d = t[f"tda_{metric}_{part}"] - t[f"base_{metric}_{part}"]
            ax[j].bar(x + (i - 0.5) * w, d, w, color=c, label=f"{lab} ({share[part]:.0f}%)")
        ax[j].axhline(0, color="black", lw=0.6)
        ax[j].set_title(title)
        ax[j].set_xticks(x, labels, rotation=40, ha="right")
        ax[j].grid(axis="y", alpha=0.3)
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


PLANE_STYLE = {
    "CLIP": (C_CLIP, "o"), "TDA": (C_TDA, "o"), "agreement gate": (C_GATE, "*"),
    "read-first TDA": (C_TDA, "v"), "TDA + T (val)": (C_TDA, "s"), "TDA + ZAEC": (C_TDA, "D"),
    "gate + ZAEC": (C_GATE, "D"), "DMN-weak": (C_OTHER, "^"), "ECALP": (C_OTHER, "P"),
}


def fig_plane(comparison: pd.DataFrame, path):
    m = comparison[comparison.dataset == "mean"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for _, r in m.iterrows():
        color, marker = PLANE_STYLE.get(r.method, ("gray", "o"))
        ax.errorbar(r.ece_mean, r.accuracy_mean, xerr=r.ece_std, yerr=r.accuracy_std, fmt=marker, color=color,
                    ms=10 if marker == "*" else 5, mec="black", mew=0.5, capsize=2, lw=0.8, label=r.method)
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)
    ax.set_xlabel("ECE (%), lower is better")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("5 test sets, P2, mean of 3 seeds (± std)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


WHERE_LABEL = {"clip_gap": "CLIP confidence - accuracy (pts; uses labels)",
               "conf": "CLIP mean confidence, first 500 samples (%)"}


def fig_where(helps: pd.DataFrame, rho: pd.DataFrame, path, signal="clip_gap"):
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    markers = {"test": "o", "near": "s", "far": "^"}
    for g, sub in helps.groupby("group"):
        ax.scatter(sub[signal], sub.d_ece, marker=markers.get(g, "o"), s=28,
                   c=[C_GATE if v < 0 else C_TDA for v in sub.d_ece], edgecolors="black", linewidths=0.5)
        ax.scatter([], [], marker=markers.get(g, "o"), s=28, c="white", edgecolors="black", linewidths=0.5, label=g)
    for _, r in helps.iterrows():
        if abs(r.d_ece) < 10 and abs(r[signal]) < 10:  # the crowd near the origin stays unlabelled
            continue
        ax.annotate(NAMES.get(r.dataset, r.dataset), (r[signal], r.d_ece), textcoords="offset points",
                    xytext=(3, 2), fontsize=5)
    r_all = rho[(rho.signal == signal) & (rho.datasets == "all")].iloc[0]
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(WHERE_LABEL[signal])
    if signal == "clip_gap":
        ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("ECE(gate) - ECE(TDA) (pts)")
    ax.set_title(f"Spearman rho = {r_all.rho:+.2f} over {int(r_all.n)} datasets")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


ABLATION = (("clip", C_CLIP, "CLIP"), ("base", C_BASE, "+ negative cache"), ("gpos", "#f4a261", "+ gated positive cache"),
            ("gate", C_GATE, "both (= the gate)"), ("tda", C_TDA, "TDA (ungated)"))


def fig_ablation(summary: pd.DataFrame, path, split="test"):
    m = summary[summary.dataset == "mean"].iloc[0]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for k, c, lab in ABLATION:
        ax.scatter(m[f"{k}_ece"], m[f"{k}_acc"], s=40, color=c, edgecolors="black", linewidths=0.5, label=lab)
    ax.set_xlabel("ECE (%)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Cache ablation ({split}, P2, mean of 5)")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    val = pd.read_csv(STORY / "summary_val.csv")
    test = pd.read_csv(STORY / "summary_test.csv")
    out = {
        "paper_selfvote_val.png": lambda p: fig_selfvote(val, p),
        "paper_agreement_val.png": lambda p: fig_agreement(val, p),
        "paper_plane_test.png": lambda p: fig_plane(pd.read_csv(T16 / "comparison.csv"), p),
        "paper_where_all.png": lambda p: fig_where(pd.read_csv(T16 / "where_it_helps.csv"),
                                                   pd.read_csv(T16 / "where_it_helps_rho.csv"), p),
        "paper_ablation_test.png": lambda p: fig_ablation(test, p),
    }
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    for name, draw in out.items():
        draw(FIG / name)
        shutil.copy(FIG / name, REPORT_FIG / name)
    print("wrote", *out)


if __name__ == "__main__":
    main()
