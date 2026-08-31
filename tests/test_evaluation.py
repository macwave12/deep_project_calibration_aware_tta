import json

import numpy as np
import pytest

from utils.evaluation import score_and_save


def _probs_and_labels():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    labels = np.array([0, 1, 1, 0])
    return probs, labels


def test_writes_file_named_by_dataset_and_method(tmp_path):
    probs, labels = _probs_and_labels()
    path = score_and_save(
        probs, labels,
        dataset="dtd", method="cal_tda", backbone="RN50", seed=0,
        hyperparams={"alpha": 2.0}, out_dir=tmp_path, timestamp="2026-08-16T00:00:00",
    )
    assert path.name == "dtd_cal_tda.json"
    assert path.exists()


def test_record_contains_every_field_the_report_needs(tmp_path):
    probs, labels = _probs_and_labels()
    path = score_and_save(
        probs, labels,
        dataset="pets", method="tda", backbone="ViT-B/16", seed=1,
        hyperparams={"alpha": 2.0, "beta": 5.0}, out_dir=tmp_path,
        timestamp="2026-08-16T00:00:00",
    )
    record = json.loads(path.read_text())

    assert record["dataset"] == "pets"
    assert record["method"] == "tda"
    assert record["backbone"] == "ViT-B/16"
    assert record["seed"] == 1
    assert record["n_samples"] == 4
    assert record["n_bins"] == 20
    assert record["hyperparams"] == {"alpha": 2.0, "beta": 5.0}
    assert record["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= record["ece"] <= 1.0
    assert len(record["reliability"]["bin_count"]) == 20


def test_rejects_mismatched_probs_and_labels(tmp_path):
    probs = np.array([[0.9, 0.1], [0.5, 0.5]])
    labels = np.array([0])
    with pytest.raises(ValueError):
        score_and_save(
            probs, labels, dataset="dtd", method="tda", backbone="RN50", seed=0,
            hyperparams={}, out_dir=tmp_path, timestamp="2026-08-16T00:00:00",
        )


def test_rejects_probability_rows_that_do_not_sum_to_one(tmp_path):
    probs = np.array([[0.9, 0.9], [0.1, 0.1]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError):
        score_and_save(
            probs, labels, dataset="dtd", method="tda", backbone="RN50", seed=0,
            hyperparams={}, out_dir=tmp_path, timestamp="2026-08-16T00:00:00",
        )
