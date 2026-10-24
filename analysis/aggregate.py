"""Turn per-run JSON records into the single table the report quotes.

This is the only path from raw runs to reported numbers. If a number appears in
the report that did not come through here, it is not reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:  # running as a script puts analysis/ itself on sys.path
    from latex_table import wrap_table
except ImportError:  # imported as analysis.aggregate (tests, pytest pythonpath=.)
    from analysis.latex_table import wrap_table

# Presentation order: baseline, prior work, then ours -- the two-contribution
# combination first, then each surviving contribution alone. The recommended
# method is `cal_tda_gate`; the three-contribution bundle is no longer a
# headline method (its numbers survive as the first ablation's `all` rows).
METHOD_ORDER = ["clipzs", "tda", "cal_tda_gate_temp", "cal_tda_gate", "cal_tda_temp"]
DATASET_ORDER = ["dtd", "flower102", "pets", "eurosat", "aircraft"]
# Display names for the report table only. The CSV keeps the raw keys, which are
# what the run records and the CLI flags use.
DATASET_LABEL = {
    "dtd": "DTD", "flower102": "Flowers102", "pets": "Pets",
    "eurosat": "EuroSAT", "aircraft": "Aircraft",
}
# Column headers for the report table. The PDF never names a code identifier,
# so the method keys (`cal_tda_gate`, `clipzs`, ...) stay in the CSV and the
# table shows these labels instead; the caption spells them out.
METHOD_LABEL = {
    "clipzs": "CLIP",
    "tda": "TDA",
    "cal_tda_gate_temp": "Gate+Temp",
    "cal_tda_gate": "Gate",
    "cal_tda_temp": "Temp",
}
MAIN_TABLE_CAPTION = (
    "Top-1 accuracy and ECE (\\%, 20 bins) on the five test splits. CLIP is zero-shot "
    "CLIP; Gate, Temp and Gate+Temp are our method with gated fusion, LOO temperature "
    "scaling, or both (margin admission off in all three). Mean is the unweighted mean "
    "over the five datasets."
)


def summary_table_tex(frame: pd.DataFrame) -> str:
    """The headline table (`tab:main`) as a LaTeX float, from the aggregated frame."""
    pivot = frame.pivot(index="dataset", columns="method", values=["accuracy", "ece"])
    # Present columns baseline -> prior work -> ours, not pandas' alphabetical
    # order, which would put our method first and read as if it were the
    # reference point.
    present = [m for m in METHOD_ORDER if m in pivot.columns.get_level_values(1)]
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product([["accuracy", "ece"], present])
    )
    # Rows in the report's dataset order (the per-contribution tables use the
    # same one), then the mean the prose quotes, so it can be checked on the page.
    datasets = [d for d in DATASET_ORDER if d in pivot.index]
    datasets += sorted(d for d in pivot.index if d not in DATASET_ORDER)
    pivot = pivot.reindex(index=datasets)
    pivot.loc["Mean"] = pivot.mean(axis=0).round(2)
    # Display names, applied only after the reorder above -- renaming first would
    # leave the reindex looking for "accuracy" in a frame that now says "Accuracy"
    # and silently produce a table of NaNs. The CSV keeps the raw keys.
    pivot = pivot.rename(index=DATASET_LABEL)
    pivot = pivot.rename(columns={"accuracy": "Accuracy", "ece": "ECE"}, level=0)
    pivot = pivot.rename(columns=METHOD_LABEL, level=1)
    pivot.index.name = None
    pivot.columns.names = [None, None]
    tabular = pivot.to_latex(
        float_format="%.2f",
        # Labels and dataset names may carry characters LaTeX treats specially;
        # pandas >= 2.0 defaults escape=False, so keep escaping explicit.
        escape=True,
    )
    # Eleven columns at the body size run ~37pt past the text width; shrink
    # the float, not the document.
    return wrap_table(tabular, caption=MAIN_TABLE_CAPTION, label="tab:main",
                      preamble=r"\small\setlength{\tabcolsep}{4pt}")


COLUMNS = [
    "dataset", "method", "accuracy", "ece", "n_samples", "n_classes", "backbone", "seed",
    "admission_rate", "temp_mean", "temp_frac_at_grid_boundary",
]


def _diagnostic_fields(hyperparams: dict) -> dict:
    """Cache-admission rate and temperature-search summary, if the record has them.

    `clipzs` and `tda` records carry no `hyperparams` fields for these at all;
    `cal_tda` records with a contribution disabled omit just that
    contribution's key (`as_record_dict` in `CalTdaConfig` removes inert
    knobs, and `temperature_summary`/`admission_summary` are only added to the
    record when they have something to report). Either way a missing value
    must become an absent/NaN cell here, never a resurrected zero -- a 0.0
    would read as "the gate admitted nothing", which is a different claim
    than "this run never measured admission at all".
    """
    admission = hyperparams.get("admission") if isinstance(hyperparams, dict) else None
    temperature = hyperparams.get("temperature") if isinstance(hyperparams, dict) else None

    admission_rate = None
    if isinstance(admission, dict) and "admission_rate" in admission:
        admission_rate = round(admission["admission_rate"], 4)

    temp_mean = None
    temp_frac_at_grid_boundary = None
    if isinstance(temperature, dict):
        if "mean" in temperature:
            temp_mean = round(temperature["mean"], 4)
        if "frac_at_grid_boundary" in temperature:
            temp_frac_at_grid_boundary = round(temperature["frac_at_grid_boundary"], 4)

    return {
        "admission_rate": admission_rate,
        "temp_mean": temp_mean,
        "temp_frac_at_grid_boundary": temp_frac_at_grid_boundary,
    }


def aggregate(raw_dir, out_csv) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no result records found in {raw_dir}")

    rows = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "dataset": record["dataset"],
            "method": record["method"],
            "accuracy": round(record["accuracy"] * 100, 2),
            "ece": round(record["ece"] * 100, 2),
            "n_samples": record["n_samples"],
            "n_classes": record["n_classes"],
            "backbone": record["backbone"],
            "seed": record["seed"],
        }
        row.update(_diagnostic_fields(record.get("hyperparams", {}) or {}))
        rows.append(row)

    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["_d"] = frame["dataset"].apply(
        lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else len(DATASET_ORDER)
    )
    frame["_m"] = frame["method"].apply(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)
    )
    frame = frame.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"]).reset_index(drop=True)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--out-csv", default="results/summary.csv")
    args = parser.parse_args()

    frame = aggregate(args.raw_dir, args.out_csv)
    print(frame.to_string(index=False))
    print(f"\nwrote {args.out_csv}")

    tex_path = Path(args.out_csv).with_name("summary_table.tex")
    tex_path.write_text(summary_table_tex(frame), encoding="utf-8")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
