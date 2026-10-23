"""Per-contribution tables for the results section: each change against the
TDA-equivalent `none` baseline it was run beside, one dataset per row.

Four tables, two per ablation study:
  margin_vs_none, fusion_vs_none  -- from the first ablation (results/ablation.csv)
  gate_vs_none, temp_vs_none      -- from the second (results/ablation_gate.csv)

The margin table also carries the cache admission rate read from the raw
`margin` records, since that is the number that explains the per-dataset
spread. Generated, never hand-typed: the report `\\input`s these files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:  # running as a script puts analysis/ itself on sys.path
    from aggregate import DATASET_LABEL, DATASET_ORDER
    from latex_table import wrap_table
except ImportError:  # imported as analysis.make_contribution_tables (tests)
    from analysis.aggregate import DATASET_LABEL, DATASET_ORDER
    from analysis.latex_table import wrap_table

BASELINE = "none"

# (key, study, variant, label, caption)
TABLES = [
    ("margin", "first", "margin", "tab:margin",
     "Margin admission against the TDA-equivalent baseline (\\texttt{none}, all flags off, "
     "run in the same sweep): accuracy and ECE (\\%) per dataset, and the fraction of test "
     "samples the margin gate admitted to the cache."),
    ("fusion", "first", "fusion", "tab:fusion",
     "Probabilistic fusion as proposed (ungated, $w = 0.5$) against the TDA-equivalent "
     "baseline (\\texttt{none}): accuracy and ECE (\\%) per dataset."),
    ("gate", "second", "gate", "tab:gate",
     "Gated probabilistic fusion against the TDA-equivalent baseline (\\texttt{none}): "
     "accuracy and ECE (\\%) per dataset."),
    ("temp", "second", "temp", "tab:temp",
     "LOO temperature scaling against the TDA-equivalent baseline (\\texttt{none}): "
     "accuracy and ECE (\\%) per dataset. Accuracy is identical by construction."),
]


def _admission_rates(ablation_raw_dir: Path) -> dict[str, float]:
    rates = {}
    for path in sorted(Path(ablation_raw_dir).glob("*_margin.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        admission = (record.get("hyperparams") or {}).get("admission") or {}
        if "admission_rate" in admission:
            rates[record["dataset"]] = 100.0 * float(admission["admission_rate"])
    return rates


def _comparison_frame(study: pd.DataFrame, variant: str,
                      admission: dict[str, float] | None) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        base = study[(study["dataset"] == dataset) & (study["variant"] == BASELINE)]
        other = study[(study["dataset"] == dataset) & (study["variant"] == variant)]
        if base.empty or other.empty:
            continue
        row = {
            "Dataset": DATASET_LABEL.get(dataset, dataset),
            f"{BASELINE} acc.": float(base.iloc[0]["accuracy"]),
            f"{variant} acc.": float(other.iloc[0]["accuracy"]),
            f"{BASELINE} ECE": float(base.iloc[0]["ece"]),
            f"{variant} ECE": float(other.iloc[0]["ece"]),
        }
        if admission is not None:
            row["Admission rate"] = admission.get(dataset, float("nan"))
        rows.append(row)
    if not rows:
        raise ValueError(f"no rows comparing {BASELINE!r} with {variant!r}")
    frame = pd.DataFrame(rows)
    mean = frame.drop(columns="Dataset").mean(numeric_only=True)
    mean["Dataset"] = "Mean"
    return pd.concat([frame, pd.DataFrame([mean])[frame.columns]], ignore_index=True)


def build_contribution_tables(ablation_csv, ablation_gate_csv, ablation_raw_dir, out_dir) -> dict[str, Path]:
    studies = {
        "first": pd.read_csv(ablation_csv),
        "second": pd.read_csv(ablation_gate_csv),
    }
    admission = _admission_rates(Path(ablation_raw_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for key, study, variant, label, caption in TABLES:
        frame = _comparison_frame(studies[study], variant, admission if key == "margin" else None)
        tabular = frame.to_latex(
            index=False, float_format="%.2f",
            # Variant names carry underscores in the header (none, gate, temp are
            # safe today, but `escape=True` is what keeps a future `gate_temp`
            # column from breaking the build with "Missing $ inserted").
            escape=True,
            column_format="l" + "r" * (len(frame.columns) - 1),
        )
        path = out_dir / f"{variant}_vs_{BASELINE}.tex"
        path.write_text(wrap_table(tabular, caption=caption, label=label), encoding="utf-8")
        written[key] = path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-csv", default="results/ablation.csv")
    parser.add_argument("--ablation-gate-csv", default="results/ablation_gate.csv")
    parser.add_argument("--ablation-raw-dir", default="results/raw/ablation")
    parser.add_argument("--out-dir", default="report/tables")
    args = parser.parse_args()

    for key, path in build_contribution_tables(
        args.ablation_csv, args.ablation_gate_csv, args.ablation_raw_dir, args.out_dir
    ).items():
        print(f"wrote {path}  ({key})")


if __name__ == "__main__":
    main()
