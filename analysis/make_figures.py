"""Generate every figure the report uses. No manual plotting, ever.

Headline claim, stated visually: our point should sit as far right as TDA
(same accuracy) but as far down as zero-shot CLIP (honest confidence).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, not windows
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_STYLE = {
    "clipzs": {"label": "CLIP zero-shot", "marker": "o", "color": "#4C72B0"},
    "tda": {"label": "TDA (CVPR'24)", "marker": "s", "color": "#DD8452"},
    "cal_tda": {"label": "Ours, all 3 contributions (loses on ECE)", "marker": "*", "color": "#55A868"},
}
# The ablation's best-performing variant (LOO temperature scaling alone, margin
# admission and probabilistic fusion both off) -- the configuration the README
# actually recommends, not the three-contribution bundle above. Plotted
# separately so the headline figure shows the result this project stands
# behind, not just the one it argues against.
TEMP_VARIANT_STYLE = {
    "label": "Ours, temperature only (recommended)", "marker": "D", "color": "#8172B2",
}

# Connector colours for the per-dataset lines in the headline scatter. Marker
# colour already encodes the method, so one flat grey for every connector left
# DTD's and EuroSAT's point clouds interleaved with no way to tell which point
# belonged to which line. These hues are deliberately chosen away from the four
# method colours above (blue / orange / green / purple) so a thin connector is
# never mistaken for a marker's colour.
DATASET_LINE_COLORS = [
    "#C44E52",  # red
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#4D4D4D",  # dark grey
    "#17BECF",  # cyan
]


def plot_accuracy_vs_ece(
    summary: pd.DataFrame, out_path: Path, temp_variant: pd.DataFrame | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for method, style in METHOD_STYLE.items():
        subset = summary[summary["method"] == method]
        if subset.empty:
            continue
        ax.scatter(
            subset["accuracy"], subset["ece"],
            marker=style["marker"], color=style["color"], s=110,
            label=style["label"], edgecolors="white", linewidths=0.8, zorder=3,
        )
    if temp_variant is not None and not temp_variant.empty:
        style = TEMP_VARIANT_STYLE
        ax.scatter(
            temp_variant["accuracy"], temp_variant["ece"],
            marker=style["marker"], color=style["color"], s=110,
            label=style["label"], edgecolors="white", linewidths=0.8, zorder=3,
        )
    # The grey polyline joins every point belonging to one dataset, so the eye can
    # follow a single dataset across methods. The temp-only variant is one of those
    # points -- it was previously scattered but left off the line, which made the
    # recommended configuration look like it belonged to no dataset at all.
    #
    # Points are joined in increasing accuracy, not in method order. Method order
    # makes the path double back on itself (TDA is the most accurate on most
    # datasets, so a clipzs -> tda -> all -> temp path runs right, left, then right
    # again) and the resulting crossings are unreadable once five datasets overlap.
    # The line carries no ordering meaning of its own -- it only says "same
    # dataset" -- so sorting by x costs nothing and guarantees a path that never
    # crosses itself.
    temp_by_dataset = (
        temp_variant.set_index("dataset") if temp_variant is not None and not temp_variant.empty
        else None
    )
    for index, (dataset, group) in enumerate(summary.groupby("dataset")):
        ordered = group.set_index("method").reindex(METHOD_STYLE).dropna(subset=["accuracy"])
        points = list(zip(ordered["accuracy"], ordered["ece"]))
        if temp_by_dataset is not None and dataset in temp_by_dataset.index:
            row = temp_by_dataset.loc[dataset]
            points.append((row["accuracy"], row["ece"]))
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        color = DATASET_LINE_COLORS[index % len(DATASET_LINE_COLORS)]
        ax.plot([p[0] for p in points], [p[1] for p in points],
                color=color, alpha=0.75, linewidth=1.4, zorder=1)
        # The dataset name is drawn in its line's colour, at the line's right-hand
        # end. That is what ties name to line: with five lines and no second
        # legend, a grey label floating next to a grey line is exactly the
        # ambiguity this colouring exists to remove.
        # 10pt clears the marker: s=110 draws a glyph roughly 12pt across, so a
        # smaller offset puts the first letter on top of the point it labels.
        ax.annotate(
            dataset, points[-1], textcoords="offset points", xytext=(10, 6),
            fontsize=8, color=color, fontweight="bold",
        )

    ax.set_xlabel("Top-1 accuracy (%)  -  higher is better")
    ax.set_ylabel("ECE, 20 bins (%)  -  lower is better")
    ax.set_title("Accuracy vs. calibration error")
    # Pets sits at 88.8% accuracy, hard against the right spine; without extra
    # margin its label is clipped by the axes edge.
    ax.margins(x=0.10)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_reliability(
    raw_dir: Path, dataset: str, out_path: Path, ablation_raw_dir: Path | None = None
) -> None:
    """One reliability curve per headline method, plus the ablation's `temp` variant if
    `ablation_raw_dir` has a record for this dataset.

    A record that is silently missing must not produce a figure that looks like a complete
    one with fewer curves -- on a fresh clone without `results/raw/` committed, this used to
    happen for every method on every dataset with no indication anything was wrong. Now it
    warns loudly (one `UserWarning` per missing curve) so a diagram with a curve quietly
    missing is at least visible in the run's console output, not only in the PNG.
    """
    # 3.4in, not 5in. These panels render into about 150pt (0.32\textwidth) in the
    # report, so what governs on-page legibility is the ratio of font size to
    # figure size, not the font size alone. Shrinking the canvas raises every text
    # element's effective size on the page without enlarging the float -- five
    # square panels simply do not fit down the page at any larger width (three rows
    # of them overflow \textheight, which pushes the caption off the paper).
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, label="perfect calibration")

    for method, style in METHOD_STYLE.items():
        record_path = raw_dir / f"{dataset}_{method}.json"
        if not record_path.exists():
            warnings.warn(
                f"plot_reliability({dataset}): no run record at {record_path} -- the "
                f"'{style['label']}' curve will be MISSING from {out_path.name}, not just "
                f"absent from this dataset by design. Check results/raw/ is populated "
                f"(committed, or regenerated by scripts/run_all).",
                stacklevel=2,
            )
            continue
        curve = json.loads(record_path.read_text(encoding="utf-8"))["reliability"]
        points = [
            (c, a) for c, a, n in zip(curve["bin_conf"], curve["bin_acc"], curve["bin_count"]) if n > 0
        ]
        if not points:
            continue
        ax.plot(
            [p[0] for p in points], [p[1] for p in points],
            marker=style["marker"], color=style["color"], label=style["label"], linewidth=1.5,
        )

    if ablation_raw_dir is not None:
        temp_path = Path(ablation_raw_dir) / f"{dataset}_temp.json"
        if temp_path.exists():
            curve = json.loads(temp_path.read_text(encoding="utf-8"))["reliability"]
            points = [
                (c, a) for c, a, n in zip(curve["bin_conf"], curve["bin_acc"], curve["bin_count"]) if n > 0
            ]
            if points:
                style = TEMP_VARIANT_STYLE
                ax.plot(
                    [p[0] for p in points], [p[1] for p in points],
                    marker=style["marker"], color=style["color"], label=style["label"], linewidth=1.5,
                )
        else:
            warnings.warn(
                f"plot_reliability({dataset}): no ablation record at {temp_path} -- the "
                f"recommended temp-only curve will be MISSING from {out_path.name}.",
                stacklevel=2,
            )

    # These panels are placed five-to-a-figure in the report, each scaled to about
    # 0.48\textwidth -- roughly 0.6x. Matplotlib's defaults (10pt) land near 6pt on
    # the page and the legend was illegible in print, so every text element here is
    # sized for the *rendered* size, not for the PNG viewed alone.
    ax.set_xlabel("confidence", fontsize=11)
    ax.set_ylabel("accuracy", fontsize=11)
    ax.set_title(f"Reliability - {dataset}", fontsize=12)
    ax.tick_params(labelsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    # No per-panel legend: five copies of a four-entry legend cost more space than
    # the panels themselves, and each copy sat on top of the curves it described.
    # make_all_figures writes one shared legend image into the grid's empty slot.
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_reliability_legend(out_path: Path) -> None:
    """The shared legend for the reliability grid, as its own image.

    The grid holds five datasets in a three-wide layout, so slot six is empty;
    this fills it. One legend for five panels is both smaller and more readable
    than five in-panel copies, and nothing overlaps a curve.
    """
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    ax.axis("off")
    handles = [
        ax.plot([], [], linestyle="--", color="black", alpha=0.5,
                label="perfect calibration")[0],
    ]
    for style in list(METHOD_STYLE.values()) + [TEMP_VARIANT_STYLE]:
        handles.append(
            ax.plot([], [], marker=style["marker"], color=style["color"],
                    linewidth=1.5, label=style["label"])[0]
        )
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all_figures(
    summary_csv, raw_dir, figures_dir, ablation_csv=None, ablation_raw_dir=None
) -> None:
    summary = pd.read_csv(summary_csv)
    raw_dir = Path(raw_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    temp_variant = None
    if ablation_csv is not None:
        ablation_csv = Path(ablation_csv)
        if ablation_csv.exists():
            ablation = pd.read_csv(ablation_csv)
            temp_variant = ablation[ablation["variant"] == "temp"][["dataset", "accuracy", "ece"]]

    ablation_raw_dir = Path(ablation_raw_dir) if ablation_raw_dir is not None else None

    plot_accuracy_vs_ece(summary, figures_dir / "accuracy_vs_ece.png", temp_variant)
    plot_reliability_legend(figures_dir / "reliability_legend.png")
    for dataset in summary["dataset"].unique():
        plot_reliability(
            raw_dir, dataset, figures_dir / f"reliability_{dataset}.png",
            ablation_raw_dir=ablation_raw_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default="results/summary.csv")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--figures-dir", default="analysis/figures")
    parser.add_argument(
        "--ablation-csv", default="results/ablation.csv",
        help="adds the ablation's temp-only variant to the headline scatter, if present",
    )
    parser.add_argument(
        "--ablation-raw-dir", default="results/raw/ablation",
        help="adds the ablation's temp-only reliability curve to each per-dataset diagram",
    )
    args = parser.parse_args()

    make_all_figures(
        args.summary_csv, args.raw_dir, args.figures_dir, args.ablation_csv, args.ablation_raw_dir
    )
    print(f"wrote figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
