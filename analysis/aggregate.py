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

# Presentation order: baseline, prior work, ours.
METHOD_ORDER = ["clipzs", "tda", "cal_tda"]
DATASET_ORDER = ["dtd", "flower102", "pets", "eurosat", "aircraft"]
# Display names for the report table only. The CSV keeps the raw keys, which are
# what the run records and the CLI flags use.
DATASET_LABEL = {
    "dtd": "DTD", "flower102": "Flowers102", "pets": "Pets",
    "eurosat": "EuroSAT", "aircraft": "Aircraft",
}
# Method names as they appear in the emitted LaTeX *after* escaping (cal_tda -> cal\_tda).
METHOD_CELLS = frozenset({"clipzs", "tda", r"cal\_tda"})


def _texttt_method_headers(tabular: str, names=METHOD_CELLS) -> str:
    """Wrap the method sub-header cells in \\texttt{}, matching how the prose sets them.

    This runs *after* `to_latex`, not before. `escape=True` is mandatory on that
    call (an unescaped underscore is a subscript operator and breaks the build),
    and it would turn any backslash injected beforehand into \\textbackslash.
    """
    out = []
    for line in tabular.split("\n"):
        stripped = line.rstrip()
        ends_row = stripped.endswith(r"\\")
        core = stripped[:-2] if ends_row else stripped
        cells = core.split("&")
        if sum(1 for c in cells if c.strip() in names) >= len(names):
            core = "&".join(
                f" \\texttt{{{c.strip()}}} " if c.strip() in names else c for c in cells
            )
            line = core + (r"\\" if ends_row else "")
        out.append(line)
    return "\n".join(out)
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
    pivot = frame.pivot(index="dataset", columns="method", values=["accuracy", "ece"])
    # Present columns baseline -> prior work -> ours, not pandas' alphabetical
    # cal_tda, clipzs, tda, which puts our method first and reads as if it were
    # the reference point.
    present = [m for m in METHOD_ORDER if m in pivot.columns.get_level_values(1)]
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product([["accuracy", "ece"], present])
    )
    # Display names, applied only after the reorder above -- renaming first would
    # leave the reindex looking for "accuracy" in a frame that now says "Accuracy"
    # and silently produce a table of NaNs. The CSV keeps the raw keys.
    #
    # The report's prose capitalises dataset names and writes ECE as an initialism;
    # a table whose header reads "ece" and whose rows read "flower102" -- a string
    # that appears nowhere else in the document -- looks like a CSV dump pasted in
    # rather than one of the report's own tables.
    pivot = pivot.rename(index=DATASET_LABEL)
    pivot = pivot.rename(columns={"accuracy": "Accuracy", "ece": "ECE"}, level=0)
    tex_path.write_text(
        wrap_table(
            _texttt_method_headers(pivot.to_latex(
                float_format="%.2f",
                # Method names contain underscores (cal_tda, clipzs). pandas >= 2.0
                # defaults escape=False, which emits a bare `_` -- a subscript
                # operator in LaTeX text mode, so the document fails to compile with
                # "Missing $ inserted". Escaping is required, not cosmetic.
                escape=True,
            )),
            caption="Top-1 accuracy and ECE (\\%, 20 bins) on the five test splits.",
            label="tab:main",
        ),
        encoding="utf-8",
    )
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
