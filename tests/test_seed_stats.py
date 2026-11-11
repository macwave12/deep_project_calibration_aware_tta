import json

import pandas as pd
import pytest

from analysis.seed_stats import (
    load_seed_records, main, method_stats, paired_stats, with_dataset_mean,
)


def _write(raw, dataset, method, seed, accuracy, ece):
    directory = raw if seed == 0 else raw / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    record = {"dataset": dataset, "method": method, "seed": seed,
              "accuracy": accuracy, "ece": ece}
    (directory / f"{dataset}_{method}.json").write_text(json.dumps(record))


def _frame(rows):
    return pd.DataFrame(rows, columns=["dataset", "method", "seed", "accuracy", "ece"])


def test_loads_seed_subfolders_as_percent(tmp_path):
    _write(tmp_path, "dtd", "tda", 0, 0.40, 0.10)
    _write(tmp_path, "dtd", "tda", 2, 0.42, 0.12)

    frame = load_seed_records(tmp_path, [0, 2])

    assert sorted(frame["seed"]) == [0, 2]
    assert frame.set_index("seed").loc[2, "accuracy"] == pytest.approx(42.0)


def test_rejects_record_in_wrong_seed_folder(tmp_path):
    _write(tmp_path, "dtd", "tda", 0, 0.40, 0.10)
    (tmp_path / "seed1").mkdir()
    (tmp_path / "seed1" / "dtd_tda.json").write_text(
        json.dumps({"dataset": "dtd", "method": "tda", "seed": 0, "accuracy": 0.4, "ece": 0.1}))

    with pytest.raises(ValueError, match="expected 1"):
        load_seed_records(tmp_path, [0, 1])


def test_missing_seed_folder_is_an_error(tmp_path):
    _write(tmp_path, "dtd", "tda", 0, 0.40, 0.10)

    with pytest.raises(FileNotFoundError):
        load_seed_records(tmp_path, [0, 1])


def test_method_stats_mean_std_mae():
    frame = _frame([("dtd", "tda", s, a, 10.0) for s, a in [(0, 40.0), (1, 42.0), (2, 47.0)]])

    row = method_stats(frame).query("metric == 'accuracy'").iloc[0]

    assert row["n"] == 3 and row["seeds"] == "0 1 2"
    assert row["mean"] == pytest.approx(43.0)
    assert row["std"] == pytest.approx(pd.Series([40.0, 42.0, 47.0]).std(ddof=1))
    assert row["mae"] == pytest.approx((3 + 1 + 4) / 3)


def test_paired_differences_and_wins_respect_metric_direction():
    frame = _frame([
        ("dtd", "tda", 0, 40.0, 10.0), ("dtd", "gate", 0, 41.0, 8.0),
        ("dtd", "tda", 1, 42.0, 10.0), ("dtd", "gate", 1, 41.0, 10.0),
        ("dtd", "tda", 2, 44.0, 10.0), ("dtd", "gate", 2, 47.0, 12.0),
    ])

    paired = paired_stats(frame, ["tda"]).set_index("metric")

    acc = paired.loc["accuracy"]
    assert acc["mean_diff"] == pytest.approx((1 - 1 + 3) / 3)
    assert acc["mae_diff"] == pytest.approx((1 + 1 + 3) / 3)
    assert (acc["wins"], acc["ties"], acc["losses"]) == (2, 0, 1)
    ece = paired.loc["ece"]  # lower ECE wins
    assert (ece["wins"], ece["ties"], ece["losses"]) == (1, 1, 1)


def test_pairing_uses_only_shared_seeds():
    frame = _frame([
        ("dtd", "tda", 0, 40.0, 10.0), ("dtd", "tda", 1, 40.0, 10.0),
        ("dtd", "gate", 1, 45.0, 10.0),
    ])

    acc = paired_stats(frame, ["tda"]).query("metric == 'accuracy'").iloc[0]

    assert acc["n"] == 1 and acc["seeds"] == "1"
    assert acc["mean_diff"] == pytest.approx(5.0)


def test_dataset_mean_row_only_over_seeds_with_every_dataset():
    frame = _frame([
        ("dtd", "tda", 0, 40.0, 10.0), ("pets", "tda", 0, 80.0, 20.0),
        ("dtd", "tda", 1, 50.0, 10.0),
    ])

    mean = with_dataset_mean(frame).query("dataset == 'mean'")

    assert list(mean["seed"]) == [0]
    assert mean.iloc[0]["accuracy"] == pytest.approx(60.0)


def test_main_writes_both_csvs(tmp_path):
    raw = tmp_path / "raw"
    for seed in (0, 1):
        _write(raw, "dtd", "tda", seed, 0.40 + seed / 100, 0.10)
        _write(raw, "dtd", "gate", seed, 0.41 + seed / 100, 0.09)

    paths = main(["--raw-dir", str(raw), "--seeds", "0", "1", "--reference", "tda",
                  "--out-dir", str(tmp_path / "out"), "--name", "t"])

    methods, paired = (pd.read_csv(p) for p in paths)
    assert set(methods["dataset"]) == {"dtd", "mean"}
    gate = paired.query("dataset == 'dtd' and metric == 'accuracy'").iloc[0]
    assert gate["method"] == "gate" and gate["wins"] == 2


def test_ci90_only_with_enough_seeds():
    from analysis.seed_stats import ci_halfwidth
    from scipy.stats import t as student_t
    few = pd.Series([1.0, 2.0, 3.0])
    assert ci_halfwidth(few) != ci_halfwidth(few)  # NaN below 10 seeds
    many = pd.Series([float(i) for i in range(10)])
    expected = student_t.ppf(0.95, 9) * many.std(ddof=1) / 10 ** 0.5
    assert ci_halfwidth(many) == pytest.approx(expected)
