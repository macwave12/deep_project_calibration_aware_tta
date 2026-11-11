import pandas as pd
import pytest

from analysis.report_assets import REPO, f, generalization, macro, prot, tab_earlier, tab_main, tab_p1


def test_format_signs_and_latex_minus():
    assert f(1.234) == "1.23"
    assert f(-1.234) == r"\ensuremath{-}1.23"
    assert f(0.5, 1, sign=True) == "+0.5"
    assert f(-0.5, 1, sign=True) == r"\ensuremath{-}0.5"


def test_macro_rejects_non_letter_names():
    assert macro("mainTdaAcc", "56.75") == "\\newcommand{\\mainTdaAcc}{56.75}"
    with pytest.raises(ValueError):
        macro("main2", "1")


def _tracked():
    an = REPO / "results" / "analysis"
    return {"prot": pd.read_csv(REPO / "results" / "protocols_10seeds.csv"),
            "paired": pd.read_csv(an / "protocols_paired_10seeds.csv"),
            "p1": pd.read_csv(an / "p1_appendix_p1_vs_p2.csv")}


def test_main_table_reads_the_ten_seed_records():
    d = _tracked()
    text = tab_main(d)
    assert f(prot(d, "mean", "cal_tda_agree", "ece")) in text
    assert text.count("\\\\") == 7  # header + 3 methods x 2 metrics
    assert "\\pm" in text


def test_earlier_and_p1_tables_have_one_row_per_method():
    d = _tracked()
    assert tab_earlier(d).count("\\\\") == 1 + 6
    assert tab_p1(d).count("\\\\") == 1 + 3


def test_generalization_joins_dev_and_fresh_and_computes_deltas():
    rows = []
    for m, a, e in (("clipzs", 60.0, 3.0), ("tda", 62.0, 8.0), ("cal_tda_agree", 61.5, 3.5)):
        for metric, v in (("accuracy", a), ("ece", e)):
            rows.append({"protocol": "P2", "dataset": "ucf101", "method": m, "metric": metric, "mean": v})
            rows.append({"protocol": "P1", "dataset": "ucf101", "method": m, "metric": metric, "mean": 0.0})
    fresh = pd.DataFrame([{"dataset": "cars", "clip_accuracy": 65, "clip_ece": 4, "tda_accuracy": 67, "tda_ece": 7,
                           "gate_accuracy": 67.2, "gate_ece": 3}, {"dataset": "mean"}])
    g = generalization({"dev": pd.DataFrame(rows), "fresh": fresh})
    assert list(g.tier) == ["dev", "fresh"]
    assert g.d_ece.tolist() == pytest.approx([-4.5, -4.0])
    assert g.d_accuracy.tolist() == pytest.approx([-0.5, 0.2])
