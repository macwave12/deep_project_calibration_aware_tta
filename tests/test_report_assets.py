import pandas as pd
import pytest

from analysis.report_assets import (REPO, f, far_spread_numbers, generalization, macro, prot, sweep_numbers, tab_main,
                                    tab_p1, no_info_numbers, tab_ablation_groups, rule_numbers, gap_numbers,
                                    corr_numbers)


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
    # bold marks the best method per column (6 columns x 2 metrics), not our method
    assert text.count("\\textbf") == 12
    pets_ece = f(prot(d, "pets", "tda", "ece"))
    assert "\\textbf{" + pets_ece in text and pets_ece < f(prot(d, "pets", "cal_tda_agree", "ece"))


def test_p1_table_has_one_row_per_method():
    assert tab_p1(_tracked()).count("\\\\") == 2 + 3  # two header rows


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
                         **{f"{k}_{m}": 50.0 + i for i, k in enumerate(("clip", "base", "apos", "gate", "rand", "tda"))
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
    assert text.count("\\\\") == 1 + 6 + 1 and "40\\%" in text  # header, six variants, the gate's share


def test_rule_numbers_split_val_from_held_out_tiers():
    rules = pd.DataFrame({"measure": ["a", "a", "b", "b"], "tier": ["val", "far", "val", "near"],
                          "gain": [-0.1, -0.5, -0.2, 0.24]})
    filt = pd.DataFrame({"tier": ["val", "near", "far"], "oracle_block": [9.0, 1.42, 2.37]})
    n = rule_numbers(rules, filt)
    assert n["niRuleBestVal"] == r"\ensuremath{-}0.10" and n["niRuleBestNear"] == "+0.24"
    assert n["niRuleBestFar"] == r"\ensuremath{-}0.50"  # held-out tiers are reported one by one
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
    assert n["gapLoseN"] == "2" and n["gapLoseUnderN"] == "1" and n["gapLoseGateMax"] == "+0.0"


def test_corr_numbers_use_held_out_tiers_and_the_three_discussed_scores():
    corrs = pd.DataFrame([{"measure": m, "tier": t, "rho": r, "rho_min": r - 0.1, "rho_max": r + 0.3}
                          for m, r in (("clip_conf", -0.1), ("cache_conf", 0.05), ("one_minus_js", 0.2), ("rho", 0.9))
                          for t in ("val", "near", "far")])
    deciles = pd.DataFrame([{"measure": m, "tier": t, "decile": 0, "right_pct": p}
                            for m, p in (("clip_conf", 55.0), ("rho", 10.0)) for t in ("val", "near", "far")]
                           + [{"measure": "cache_conf", "tier": "far", "decile": 1, "right_pct": 52.4},
                              {"measure": "cache_conf", "tier": "val", "decile": 1, "right_pct": 30.0}])
    n = corr_numbers(corrs, deciles)
    assert n["niCorrClipNear"] == r"\ensuremath{-}0.10" and n["niCorrJsFar"] == "+0.20"
    assert n["niCorrMaxAbs"] == "0.20" and n["niCorrDsMaxAbs"] == "0.50"  # the "rho" score is not discussed
    assert n["niDecileMinNear"] == "55" and n["niDecileMinFar"] == "52"
    assert n["niDecileAllMinNear"] == "10"  # every score counts here


def test_sweep_numbers_interpolate_where_tda_reaches_the_gates_accuracy():
    matched = pd.DataFrame({"dataset": ["mean"], "method": ["agreement gate (ours)"], "acc": [56.5]})
    curve = pd.DataFrame({"dataset": "mean", "alpha": [0.0, 1.0, 2.0, 3.0], "acc": [54.0, 56.0, 57.0, 56.9],
                          "ece": [4.0, 8.0, 12.0, 14.0]})
    n = sweep_numbers(matched, curve)
    assert n["sweepZeroAcc"] == "54.0" and n["sweepZeroEce"] == "4.0" and n["sweepMatchAlpha"] == "1.5"
    assert n["matchedTdaEce"] == "10.0"
    with pytest.raises(ValueError):
        sweep_numbers(matched.assign(acc=60.0), curve)


def test_far_spread_names_the_three_largest_reductions_and_the_range_of_the_rest():
    g = pd.DataFrame({"dataset": ["mnist", "pcam", "svhn", "gtsrb", "imagenet_r", "caltech101"],
                      "tier": ["far", "far", "far", "far", "far", "near"], "d_ece": [-7.2, -9.7, -4.0, 0.8, 2.1, -20.0]})
    n = far_spread_numbers(g)
    assert n["farBigNames"] == "PCam, MNIST and SVHN" and n["farBigLo"] == "4.0" and n["farBigHi"] == "9.7"
    assert n["farRestLo"] == "+0.8" and n["farRestHi"] == "+2.1" and n["farRestN"] == "2"


def test_comparison_mean_is_taken_over_the_five_datasets_not_the_rounded_mean_row():
    from analysis.report_assets import comp
    rows = [{"dataset": ds, "method": "CLIP", "ece_mean": v}
            for ds, v in zip(("a", "b", "c", "d", "e"), (5.458, 8.238, 7.314, 2.905, 4.358))]
    d = {"comp": pd.DataFrame(rows + [{"dataset": "mean", "method": "CLIP", "ece_mean": 5.655}])}
    assert f(comp(d, "CLIP", "ece")) == "5.65"  # the CSV's mean row would print 5.66
    assert comp(d, "CLIP", "ece", "b") == 8.238


def test_read_first_and_agree_range_use_the_actual_runs_and_every_dataset():
    from analysis.report_assets import agree_range_numbers, read_first
    rf = pd.DataFrame({"split": ["val", "val", "val", "test"],
                       "method": ["TDA read first (no self-vote)", "TDA read first (no self-vote)", "TDA (write first)",
                                  "TDA read first (no self-vote)"],
                       "dataset": ["dtd", "pets", "dtd", "dtd"], "ece": [12.0, 4.0, 19.0, 1.0], "acc": [46.0, 86.0, 45.0, 1.0]})
    d = {"read_first": rf}
    assert read_first(d, "val", "ece") == 8.0 and read_first(d, "val", "acc", "pets") == 86.0
    seed = pd.DataFrame({"tier": ["far", "far", "near", "near"], "dataset": ["svhn", "svhn", "caltech101", "caltech101"],
                         "seed": [0, 1, 0, 1], "agree_share": [20.0, 28.0, 92.0, 94.0]})
    n = agree_range_numbers(seed)
    assert n["niAgreeDsLo"] == "24" and n["niAgreeDsHi"] == "93"
