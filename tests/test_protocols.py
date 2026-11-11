import json

import pandas as pd
import pytest

from analysis.protocols import headline_rule, main, protocol_tables, split_protocols


def _frame(rows):
    return pd.DataFrame(rows, columns=["dataset", "method", "seed", "accuracy", "ece"])


def _write(raw, dataset, method, seed, accuracy, ece):
    directory = raw if seed == 0 else raw / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{dataset}_{method}.json").write_text(json.dumps(
        {"dataset": dataset, "method": method, "seed": seed, "accuracy": accuracy, "ece": ece}))


def test_split_marks_clean_and_copies_clip_to_p2():
    frame = split_protocols(_frame([
        ("dtd", "clipzs", 0, 40.0, 5.0), ("dtd", "tda", 0, 45.0, 9.0),
        ("dtd", "tda_clean", 0, 43.0, 8.0), ("dtd", "unrelated", 0, 1.0, 1.0),
    ]))

    assert set(frame["method"]) == {"clipzs", "tda"}
    assert sorted(frame.query("method == 'clipzs'")["protocol"]) == ["P1", "P2"]
    assert frame.query("method == 'tda' and protocol == 'P2'")["accuracy"].item() == 43.0


def test_p2_minus_p1_is_paired_by_seed():
    frame = split_protocols(_frame([
        ("dtd", "tda", 0, 45.0, 9.0), ("dtd", "tda_clean", 0, 43.0, 8.0),
        ("dtd", "tda", 1, 46.0, 9.0), ("dtd", "tda_clean", 1, 47.0, 8.0),
    ]))

    _, paired = protocol_tables(frame, references=())
    acc = paired.query("comparison == 'P2_minus_P1' and dataset == 'dtd' "
                       "and metric == 'accuracy'").iloc[0]

    assert acc["method"] == "tda" and acc["n"] == 2
    assert acc["mean_diff"] == pytest.approx((-2 + 1) / 2)
    assert (acc["wins"], acc["losses"]) == (1, 1)


@pytest.mark.parametrize("p2_tda, expected", [(42.0, "P2"), (39.0, "P1")])
def test_headline_rule_reads_p2_tda_against_clip(p2_tda, expected):
    frame = split_protocols(_frame([
        ("dtd", "clipzs", 0, 40.0, 5.0), ("dtd", "tda", 0, 45.0, 9.0),
        ("dtd", "tda_clean", 0, p2_tda, 8.0),
    ]))

    _, paired = protocol_tables(frame)

    assert headline_rule(paired)["headline"] == expected


def test_main_writes_tables(tmp_path):
    raw = tmp_path / "raw"
    for seed in (0, 1):
        _write(raw, "dtd", "clipzs", seed, 0.40, 0.05)
        _write(raw, "dtd", "tda", seed, 0.45, 0.09)
        _write(raw, "dtd", "tda_clean", seed, 0.44, 0.08)

    rule = main(["--raw-dir", str(raw), "--seeds", "0", "1",
                 "--out-csv", str(tmp_path / "p.csv"), "--paired-csv", str(tmp_path / "pp.csv")])

    stats = pd.read_csv(tmp_path / "p.csv")
    assert set(stats["protocol"]) == {"P1", "P2"}
    assert rule["headline"] == "P2" and rule["wins"] == 2
