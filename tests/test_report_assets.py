import pandas as pd
import pytest

from analysis.report_assets import (REPO, f, generalization, macro, prot, switch_numbers, tab_earlier, tab_main,
                                    tab_p1, tab_switch, tab_switch_design, no_info_numbers,
                                    tab_ablation_groups, rule_numbers, gap_numbers)


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


def _switch_frame():
    rows = []
    for tier in ("val", "dev", "test", "near", "far"):
        for method, e in (("TDA", 10.0), ("TDA + ZAEC", 6.0), ("agreement gate", 5.5), ("confidence switch", 5.0)):
            rows.append({"tier": tier, "dataset": "mean", "method": method, "accuracy": 56.0, "ece": e,
                         "switch_on_pct": None, "clip_conf": None, "threshold": 0.78, "plateau_lo": 0.7,
                         "plateau_hi": 0.84})
    for tier, ds, pct, conf in (("val", "pets", 99.0, 84.0), ("test", "pets", 99.0, 84.5),
                                ("near", "cifar10", 98.0, 85.0), ("near", "cifar100", 0.0, 56.0)):
        rows.append({"tier": tier, "dataset": ds, "method": "confidence switch", "accuracy": 1.0, "ece": 1.0,
                     "switch_on_pct": pct, "clip_conf": conf, "clip_gap": conf - 80, "threshold": 0.78, "plateau_lo": 0.7,
                     "plateau_hi": 0.84})
    return pd.DataFrame(rows)


def _sweep():
    return pd.DataFrame({"tier": ["near"] * 4 + ["val"], "threshold": [0.6, 0.65, 0.7, 0.8, 0.6],
                         "accuracy": 50.0, "ece": [5.4, 5.2, 5.25, 5.5, 1.0]})


def test_switch_numbers_and_table_read_the_tier_means():
    n = switch_numbers(_switch_frame(), _sweep())
    assert n["swThresh"] == "0.78"
    assert n["swSwTestEce"] == "5.00" and n["swGateNearEce"] == "5.50"
    assert n["swDDevEce"] == r"\ensuremath{-}0.50"
    assert n["swOnN"] == "2" and n["swHeldN"] == "3" and n["swNearOnN"] == "1"
    assert n["swMinOnConf"] == "0.84" and n["swMaxOffConf"] == "0.56" and n["swPlateauHi"] == "0.84"
    text = tab_switch({"switch": _switch_frame()})
    assert text.count("\\\\") == 5 and "5.00 (56.0)" in text
    assert n["swNearBestEce"] == "5.20" and n["swNearBestLo"] == "0.65" and n["swNearBestHi"] == "0.70"
    design = tab_switch_design({"switch": _switch_frame()})
    assert design.count("\\\\") == 1 + 1 and "\\ensuremath{-}4.0 & yes" in design  # header + val pets only


def test_generalization_tags_near_far_other_drops_excluded_and_computes_deltas():
    rows = []
    for m, a, e in (("clipzs", 60.0, 3.0), ("tda", 62.0, 8.0), ("cal_tda_agree", 61.5, 3.5)):
        for metric, v in (("accuracy", a), ("ece", e)):
            rows.append({"protocol": "P2", "dataset": "caltech101", "method": m, "metric": metric, "mean": v})
            rows.append({"protocol": "P1", "dataset": "caltech101", "method": m, "metric": metric, "mean": 0.0})
    fresh = pd.DataFrame([{"dataset": "gtsrb", "tier": "fresh", "clip_accuracy": 65, "clip_ece": 4, "tda_accuracy": 67, "tda_ece": 7,
                           "gate_accuracy": 67.2, "gate_ece": 3},
                          {"dataset": "imagenet", "tier": "fresh", "clip_accuracy": 66, "clip_ece": 2, "tda_accuracy": 68,
                           "tda_ece": 4, "gate_accuracy": 68, "gate_ece": 2}, {"dataset": "mean"}])
    g = generalization({"dev": pd.DataFrame(rows), "fresh": fresh})
    assert list(g.tier) == ["near", "far"]
    assert g.d_ece.tolist() == pytest.approx([-4.5, -4.0])
    assert g.d_accuracy.tolist() == pytest.approx([-0.5, 0.2])


def _ni_frames():
    tiers = ("val", "test", "near", "far")
    abl = pd.DataFrame([{"tier": t, "agree_share": 60.0, "disagree_share": 40.0,
                         **{f"{k}_{m}": 50.0 + i for i, k in enumerate(("clip", "base", "apos", "gate", "tda", "gpos"))
                            for m in ("acc", "ece")},
                         "base_acc_agree": 80.0, "tda_acc_agree": 80.1, "base_gap_agree": -2.0, "tda_gap_agree": 10.0,
                         "base_acc_disagree": 40.0, "tda_acc_disagree": 45.0, "base_gap_disagree": 8.0,
                         "tda_gap_disagree": 9.0} for t in tiers])
    aucs = pd.DataFrame([{"measure": m, "tier": t, "auc": a} for t in tiers
                         for m, a in (("rho", 0.48 if t in ("near", "far") else 0.4), ("combined", 0.5))])
    filt = pd.DataFrame([{"tier": t, "threshold": 0.45, "filter_gain": -0.002, "oracle_block": 2.0, "oracle_pick": 5.0}
                         for t in tiers])
    return abl, aucs, filt


def test_no_info_numbers_and_ablation_table():
    abl, aucs, filt = _ni_frames()
    n = no_info_numbers(abl, aucs, filt)
    assert n["niFilterFar"] == "+0.00" and n["niAgreeTdaGapVal"] == "+10.0"
    assert n["niHeldAucLo"] == "0.48" and n["niAposMaxDAcc"] == "1.00" and n["niAgreeMaxDAcc"] == "0.10"
    text = tab_ablation_groups({"ni_abl": abl})
    assert text.count("\\\\") == 1 + 6 + 6 and r"80.0 $\to$ 80.1" in text


def test_rule_numbers_split_val_from_held_out_tiers():
    rules = pd.DataFrame({"measure": ["a", "a", "b", "b"], "tier": ["val", "far", "val", "near"],
                          "gain": [-0.1, -0.5, -0.2, 0.24]})
    filt = pd.DataFrame({"tier": ["val", "near", "far"], "oracle_block": [9.0, 1.42, 2.37]})
    n = rule_numbers(rules, filt)
    assert n["niRuleBestVal"] == r"\ensuremath{-}0.10" and n["niRuleBestHeld"] == "+0.24"
    assert n["niOracleBlockLo"] == "1.4" and n["niOracleBlockHi"] == "2.4"


def test_gap_numbers_count_directions_and_split_ece_losses_by_clip_sign():
    gaps = pd.DataFrame({"dataset": ["pets", "gtsrb", "dtd", "cifar10"], "tier": ["test", "far", "test", "other"],
                         "clip_gap": [-3.0, 0.2, 8.0, -5.0], "tda_gap": [-1.0, 7.0, 17.0, 1.0],
                         "gate_gap": [-4.0, 0.0, 7.0, -5.0]})
    helps = pd.DataFrame({"dataset": ["pets", "gtsrb", "dtd", "cifar10"], "d_ece": [2.0, 0.8, -9.0, 3.0]})
    n = gap_numbers(gaps, helps)
    assert n["gapN"] == "3" and n["gapTdaUpN"] == "3" and n["gapGateDownN"] == "3" and n["gapTdaOverN"] == "2"
    assert n["gapTestTda"] == "+8.0" and n["gapFarGate"] == "+0.0"
    assert n["gapLoseUnderNames"] == "Pets" and n["gapLoseOtherNames"] == "GTSRB"
