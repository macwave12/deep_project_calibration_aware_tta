import json

import pandas as pd
import pytest

from analysis.make_contribution_tables import build_contribution_tables


@pytest.fixture
def fake_study(tmp_path):
    """Two datasets for both ablation studies, plus margin records carrying
    admission rates, so every table has rows to fill and a mean to take."""
    first = pd.DataFrame([
        {"dataset": d, "variant": v, "accuracy": a, "ece": e}
        for d, rows in {
            "dtd": [("none", 47.04, 17.41), ("margin", 44.86, 20.11), ("fusion", 45.80, 17.69)],
            "pets": [("none", 88.74, 2.48), ("margin", 88.42, 2.53), ("fusion", 88.77, 43.78)],
        }.items()
        for v, a, e in rows
    ])
    second = pd.DataFrame([
        {"dataset": d, "variant": v, "accuracy": a, "ece": e}
        for d, rows in {
            "dtd": [("none", 47.04, 17.41), ("gate", 45.69, 6.24), ("temp", 47.04, 4.68)],
            "pets": [("none", 88.74, 2.48), ("gate", 88.77, 5.68), ("temp", 88.74, 15.54)],
        }.items()
        for v, a, e in rows
    ])
    ablation_csv = tmp_path / "ablation.csv"
    gate_csv = tmp_path / "ablation_gate.csv"
    first.to_csv(ablation_csv, index=False)
    second.to_csv(gate_csv, index=False)

    raw = tmp_path / "raw_ablation"
    raw.mkdir()
    for dataset, rate in [("dtd", 0.53), ("pets", 0.82)]:
        (raw / f"{dataset}_margin.json").write_text(json.dumps({
            "dataset": dataset, "method": "margin",
            "hyperparams": {"admission": {"admission_rate": rate}},
        }))
    return ablation_csv, gate_csv, raw


def test_writes_four_tables_with_expected_labels(fake_study, tmp_path):
    paths = build_contribution_tables(*fake_study, tmp_path / "tables")
    assert set(paths) == {"margin", "fusion", "gate", "temp"}
    assert r"\label{tab:margin}" in paths["margin"].read_text(encoding="utf-8")
    assert r"\label{tab:gate}" in paths["gate"].read_text(encoding="utf-8")


def test_margin_table_carries_admission_rate_as_percent(fake_study, tmp_path):
    text = build_contribution_tables(*fake_study, tmp_path / "tables")["margin"].read_text(encoding="utf-8")
    assert "53.00" in text and "DTD" in text


def test_tables_end_with_a_mean_row(fake_study, tmp_path):
    text = build_contribution_tables(*fake_study, tmp_path / "tables")["gate"].read_text(encoding="utf-8")
    body = text.split(r"\bottomrule")[0].rstrip().splitlines()
    assert body[-1].startswith("Mean")


def test_variant_names_are_escaped_for_latex(fake_study, tmp_path):
    text = build_contribution_tables(*fake_study, tmp_path / "tables")["temp"].read_text(encoding="utf-8")
    assert "_" not in text.replace(r"\_", "")


def test_mean_row_averages_the_dataset_rows(fake_study, tmp_path):
    # gate accuracy: (45.69 + 88.77) / 2 = 67.23
    text = build_contribution_tables(*fake_study, tmp_path / "tables")["gate"].read_text(encoding="utf-8")
    mean_line = [l for l in text.splitlines() if l.startswith("Mean")][0]
    assert "67.23" in mean_line
