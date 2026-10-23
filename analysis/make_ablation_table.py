"""Ablation: which of the three contributions actually moved the numbers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:  # running as a script puts analysis/ itself on sys.path
    from latex_table import wrap_table
except ImportError:  # imported as analysis.make_ablation_table (tests)
    from analysis.latex_table import wrap_table

VARIANT_ORDER = [
    "none", "margin", "fusion", "temp",
    "margin_fusion", "margin_temp", "fusion_temp", "all",
]
# Each label leads with the variant's config name, because that is what the report's
# prose calls it -- Section 5 refers to "the `none` row" and to `temp`, `margin_fusion`
# and `all` by name, and none of those strings appeared in this table before.
VARIANT_LABEL = {
    "none": "none: TDA-equivalent (all flags off)",
    "margin": "margin: + margin admission",
    "fusion": "fusion: + probabilistic fusion",
    "temp": "temp: + LOO temperature",
    "margin_fusion": "margin_fusion: + margin + fusion",
    "margin_temp": "margin_temp: + margin + temperature",
    "fusion_temp": "fusion_temp: + fusion + temperature",
    "all": "all: all three (ours)",
}


# The second ablation keeps only the two contributions that survived the first
# one -- gated probabilistic fusion and LOO temperature -- with margin admission
# off throughout. Same shape of study, half the flags, four variants.
GATE_VARIANT_ORDER = ["none", "gate", "temp", "gate_temp"]
GATE_VARIANT_LABEL = {
    "none": "none: TDA-equivalent (both off)",
    "gate": "gate: + gated fusion",
    "temp": "temp: + LOO temperature",
    "gate_temp": "gate_temp: + gated fusion + temperature",
}

DEFAULT_CAPTION = "Ablation: mean accuracy and ECE (\\%) across the five datasets."

STUDIES = {
    "three": dict(
        raw_dir="results/raw/ablation",
        out_csv="results/ablation.csv",
        out_tex="report/tables/ablation.tex",
        variant_order=VARIANT_ORDER,
        variant_label=VARIANT_LABEL,
        caption=DEFAULT_CAPTION,
        label="tab:ablation",
    ),
    "gate": dict(
        raw_dir="results/raw/ablation_gate",
        out_csv="results/ablation_gate.csv",
        out_tex="report/tables/ablation_gate.tex",
        variant_order=GATE_VARIANT_ORDER,
        variant_label=GATE_VARIANT_LABEL,
        caption=("Second ablation, margin admission off: mean accuracy and ECE (\\%) "
                 "across the five datasets for gated fusion and LOO temperature."),
        label="tab:ablation-gate",
    ),
}


def build_ablation(raw_dir, out_csv, out_tex, *, variant_order=VARIANT_ORDER,
                   variant_label=VARIANT_LABEL, caption=DEFAULT_CAPTION,
                   label="tab:ablation") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no ablation records found in {raw_dir}")

    rows = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        dataset, variant = path.stem.split("_", 1)
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "accuracy": round(record["accuracy"] * 100, 2),
                "ece": round(record["ece"] * 100, 2),
            }
        )

    frame = pd.DataFrame(rows)
    frame["_v"] = frame["variant"].apply(
        lambda v: variant_order.index(v) if v in variant_order else len(variant_order)
    )
    frame = frame.sort_values(["_v", "dataset"]).drop(columns="_v")

    means = (
        frame.groupby("variant")[["accuracy", "ece"]].mean().round(2)
        .reindex([v for v in variant_order if v in set(frame["variant"])])
    )
    means.index = [variant_label.get(v, v) for v in means.index]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    Path(out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(out_tex).write_text(
        wrap_table(
            means.rename(columns={"accuracy": "Accuracy", "ece": "ECE"}).to_latex(
                float_format="%.2f",
                # See analysis/aggregate.py: unescaped underscores break the LaTeX
                # build, and three variant labels now carry one (margin_fusion,
                # margin_temp, fusion_temp), so this is load-bearing, not defensive.
                escape=True,
            ),
            caption=caption,
            label=label,
        ),
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=sorted(STUDIES), default="three",
        help="which ablation to build: 'three' (the original 8-variant study) or "
             "'gate' (gated fusion x temperature, margin off); sets the defaults below",
    )
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-tex", default=None)
    args = parser.parse_args()

    study = STUDIES[args.study]
    frame = build_ablation(
        args.raw_dir or study["raw_dir"],
        args.out_csv or study["out_csv"],
        args.out_tex or study["out_tex"],
        variant_order=study["variant_order"],
        variant_label=study["variant_label"],
        caption=study["caption"],
        label=study["label"],
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
