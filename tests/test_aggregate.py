import json
import re
from pathlib import Path

import pandas as pd
import pytest

from analysis.aggregate import aggregate


def _write_record(directory, dataset, method, accuracy, ece, hyperparams=None):
    record = {
        "dataset": dataset, "method": method, "backbone": "RN50", "seed": 0,
        "n_samples": 1692, "n_classes": 47, "n_bins": 20,
        "accuracy": accuracy, "ece": ece,
        "reliability": {"bin_count": [0] * 20, "bin_acc": [0.0] * 20, "bin_conf": [0.0] * 20},
        "hyperparams": {"alpha": 2.0} if hyperparams is None else hyperparams,
        "timestamp": "2026-08-16T00:00:00",
    }
    (directory / f"{dataset}_{method}.json").write_text(json.dumps(record))


def test_aggregate_writes_one_row_per_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_record(raw, "dtd", "tda", 0.4312, 0.0921)
    _write_record(raw, "dtd", "clipzs", 0.4021, 0.0570)

    out = tmp_path / "summary.csv"
    frame = aggregate(raw, out)

    assert len(frame) == 2
    assert out.exists()


def test_metrics_are_stored_as_rounded_percentages(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_record(raw, "dtd", "tda", 0.431234, 0.092149)

    frame = aggregate(raw, tmp_path / "summary.csv")
    row = frame.iloc[0]

    assert row["accuracy"] == pytest.approx(43.12)
    assert row["ece"] == pytest.approx(9.21)


def test_rows_are_sorted_by_dataset_then_method_order(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for method in ["cal_tda_temp", "cal_tda_gate", "clipzs", "cal_tda_gate_temp", "tda"]:
        _write_record(raw, "pets", method, 0.5, 0.05)
        _write_record(raw, "dtd", method, 0.4, 0.06)

    frame = aggregate(raw, tmp_path / "summary.csv")

    assert list(frame["dataset"])[:5] == ["dtd"] * 5
    assert list(frame["method"])[:5] == [
        "clipzs", "tda", "cal_tda_gate_temp", "cal_tda_gate", "cal_tda_temp"
    ]


HEADLINE = {
    "clipzs": (0.40, 0.05), "tda": (0.43, 0.09), "cal_tda_gate_temp": (0.42, 0.06),
    "cal_tda_gate": (0.42, 0.04), "cal_tda_temp": (0.43, 0.07),
}


def _headline_frame(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for dataset, shift in [("pets", 0.10), ("dtd", 0.0)]:
        for method, (acc, ece) in HEADLINE.items():
            _write_record(raw, dataset, method, acc + shift, ece)
    return aggregate(raw, tmp_path / "summary.csv")


def test_summary_table_shows_display_names_not_code_keys(tmp_path):
    # The report must not name code identifiers; the column headers are the
    # one place the method keys used to leak into the PDF.
    from analysis.aggregate import summary_table_tex

    tex = summary_table_tex(_headline_frame(tmp_path))

    for leaked in ("cal_tda", r"cal\_tda", "clipzs", "\\texttt"):
        assert leaked not in tex
    for shown in ("CLIP", "TDA", "Gate+Temp", "Gate", "Temp"):
        assert shown in tex
    assert r"\label{tab:main}" in tex


def test_summary_table_is_set_small_with_tight_columns_so_eleven_columns_fit(tmp_path):
    # Eleven columns at the body size overflow the text width by ~37pt
    # (pdflatex "Overfull \\hbox" on tab:main); the float must shrink itself.
    from analysis.aggregate import summary_table_tex

    tex = summary_table_tex(_headline_frame(tmp_path))

    body_start = tex.index(r"\begin{tabular}")
    assert r"\small" in tex[:body_start]
    assert r"\setlength{\tabcolsep}{4pt}" in tex[:body_start]


def test_summary_table_orders_datasets_as_the_report_does_and_ends_with_a_mean_row(tmp_path):
    from analysis.aggregate import summary_table_tex

    tex = summary_table_tex(_headline_frame(tmp_path))

    assert tex.index("DTD") < tex.index("Pets") < tex.index("Mean")
    # clipzs accuracy: dtd 40.00, pets 50.00 -> mean 45.00 in the first numeric column
    assert re.search(r"^Mean & 45\.00 &", tex, re.M)


def test_no_bundle_records_in_headline_dir():
    # The three-contribution bundle is no longer a headline method; a leftover
    # record would silently add a sixth row to summary.csv and both figures.
    assert not list(Path("results/raw").glob("*_cal_tda.json"))


def test_empty_results_directory_is_an_explicit_error(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(FileNotFoundError):
        aggregate(raw, tmp_path / "summary.csv")


def test_admission_and_temperature_columns_are_populated_for_cal_tda(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_record(
        raw, "dtd", "cal_tda", 0.45, 0.19,
        hyperparams={
            "use_margin_admission": True,
            "admission": {"n_offered": 1692, "n_admitted": 1268, "admission_rate": 0.7494089834515366},
            "temperature": {"mean": 0.7361997635933807, "frac_at_grid_boundary": 0.7594562647754137},
        },
    )

    frame = aggregate(raw, tmp_path / "summary.csv")
    row = frame.iloc[0]

    assert row["admission_rate"] == pytest.approx(0.7494)
    assert row["temp_mean"] == pytest.approx(0.7362)
    assert row["temp_frac_at_grid_boundary"] == pytest.approx(0.7595)


def test_missing_admission_and_temperature_fields_are_empty_not_zero(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # clipzs/tda records never carry these fields at all.
    _write_record(raw, "dtd", "clipzs", 0.44, 0.08, hyperparams={})
    _write_record(raw, "dtd", "tda", 0.47, 0.17, hyperparams={"alpha": 2.0})

    frame = aggregate(raw, tmp_path / "summary.csv")

    for _, row in frame.iterrows():
        assert pd.isna(row["admission_rate"])
        assert pd.isna(row["temp_mean"])
        assert pd.isna(row["temp_frac_at_grid_boundary"])


def test_cal_tda_with_a_disabled_contribution_only_omits_that_contributions_columns(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # use_loo_temperature is off -> no "temperature" key in hyperparams, but
    # admission is unconditional (rate 1.0) and must still be recorded, not
    # dropped alongside the disabled contribution.
    _write_record(
        raw, "dtd", "cal_tda", 0.47, 0.17,
        hyperparams={
            "use_margin_admission": False,
            "admission": {"n_offered": 1692, "n_admitted": 1692, "admission_rate": 1.0},
            "inert": ["margin_threshold"],
        },
    )

    frame = aggregate(raw, tmp_path / "summary.csv")
    row = frame.iloc[0]

    assert row["admission_rate"] == pytest.approx(1.0)
    assert pd.isna(row["temp_mean"])
    assert pd.isna(row["temp_frac_at_grid_boundary"])


def test_ablation_subdirectory_is_not_aggregated(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_record(raw, "dtd", "tda", 0.47, 0.17)

    ablation = raw / "ablation"
    ablation.mkdir()
    _write_record(ablation, "dtd", "margin_fusion", 0.46, 0.18)

    frame = aggregate(raw, tmp_path / "summary.csv")

    assert len(frame) == 1
    assert list(frame["method"]) == ["tda"]


def test_verification_subdirectory_is_not_aggregated(tmp_path):
    # The flags-off equivalence control (cal_tda with tda_equivalent.yaml) is
    # a verification artifact, not a headline result: it must not appear as a
    # `cal_tda_tdaequiv` row in the canonical summary.
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_record(raw, "dtd", "cal_tda", 0.46, 0.19)

    verification = raw / "verification"
    verification.mkdir()
    _write_record(verification, "dtd", "cal_tda_tdaequiv", 0.47, 0.17)

    frame = aggregate(raw, tmp_path / "summary.csv")

    assert len(frame) == 1
    assert list(frame["method"]) == ["cal_tda"]
