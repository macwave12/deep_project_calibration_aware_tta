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


def build_ablation(raw_dir, out_csv, out_tex) -> pd.DataFrame:
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
        lambda v: VARIANT_ORDER.index(v) if v in VARIANT_ORDER else len(VARIANT_ORDER)
    )
    frame = frame.sort_values(["_v", "dataset"]).drop(columns="_v")

    means = (
        frame.groupby("variant")[["accuracy", "ece"]].mean().round(2)
        .reindex([v for v in VARIANT_ORDER if v in set(frame["variant"])])
    )
    means.index = [VARIANT_LABEL.get(v, v) for v in means.index]

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
            caption="Ablation: mean accuracy and ECE (\\%) across the five datasets.",
            label="tab:ablation",
        ),
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="results/raw/ablation")
    parser.add_argument("--out-csv", default="results/ablation.csv")
    parser.add_argument("--out-tex", default="report/tables/ablation.tex")
    args = parser.parse_args()

    frame = build_ablation(args.raw_dir, args.out_csv, args.out_tex)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
