import json

import pandas as pd
import pytest

from analysis.make_figures import make_all_figures


@pytest.fixture
def fake_results(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for dataset in ["dtd", "pets"]:
        for method, accuracy, ece in [
            ("clipzs", 40.2, 5.7), ("tda", 43.1, 9.2), ("cal_tda", 43.0, 6.1)
        ]:
            rows.append({
                "dataset": dataset, "method": method, "accuracy": accuracy, "ece": ece,
                "n_samples": 1692, "n_classes": 47, "backbone": "RN50", "seed": 0,
            })
            (raw / f"{dataset}_{method}.json").write_text(json.dumps({
                "dataset": dataset, "method": method,
                "reliability": {
                    "bin_lower": [i / 20 for i in range(20)],
                    "bin_upper": [(i + 1) / 20 for i in range(20)],
                    "bin_conf": [(i + 0.5) / 20 for i in range(20)],
                    "bin_acc": [(i + 0.5) / 20 * 0.8 for i in range(20)],
                    "bin_count": [10] * 20,
                },
            }))
    summary = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    return summary, raw, tmp_path / "figures"


def test_creates_the_headline_scatter(fake_results):
    summary, raw, figures = fake_results
    make_all_figures(summary, raw, figures)
    scatter = figures / "accuracy_vs_ece.png"
    assert scatter.exists() and scatter.stat().st_size > 0


def test_creates_one_reliability_diagram_per_dataset(fake_results):
    summary, raw, figures = fake_results
    make_all_figures(summary, raw, figures)
    assert (figures / "reliability_dtd.png").exists()
    assert (figures / "reliability_pets.png").exists()


def test_reruns_are_idempotent(fake_results):
    summary, raw, figures = fake_results
    make_all_figures(summary, raw, figures)
    make_all_figures(summary, raw, figures)
    # scatter + shared reliability legend + one reliability plot per dataset.
    # The point of this test is that a second run overwrites rather than
    # accumulating, so the count must stay exact -- not >=.
    assert len(list(figures.glob("*.png"))) == 4
    assert (figures / "reliability_legend.png").exists()
