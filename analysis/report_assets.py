"""Every number and table of the course report (T16), generated from tracked CSVs.

Writes `report/generated/numbers.tex` (one `\\newcommand` per number the prose uses) and
`report/generated/tab_*.tex` (the tables). The prose never types a result by hand (RULES 9).

Sources (all tracked):
  results/protocols_10seeds.csv, results/analysis/protocols_paired_10seeds.csv   main table (test, 10 seeds)
  results/analysis/t16/comparison{,_paired}.csv                                comparison (test, 3 seeds)
  results/analysis/t14/dev_protocols.csv, results/analysis/t16/fresh.csv       generalization
  results/analysis/t16/where_it_helps{,_rho}.csv                               where it helps
  results/analysis/story/summary_{val,test}.csv                                diagnosis (val), ablation (test)
  results/analysis/t14/matched_test.csv, alpha_curve_test.csv                  TDA's alpha sweep vs the gate
  results/analysis/p1_appendix_p1_vs_p2.csv                                    appendix P1

    python analysis/report_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.story import CONF_CUT, NAMES, SIGNAL_HEAD  # noqa: E402
from utils.feature_cache import EXCLUDED, held_tier  # noqa: E402

OUT = REPO / "report" / "generated"
RES = REPO / "results"
AN = RES / "analysis"
METHOD = {"clipzs": "CLIP", "tda": "TDA", "cal_tda_agree": "agreement gate"}


def f(x, d=2, sign=False) -> str:
    x = 0.0 if round(x, d) == 0 else x  # no "-0.0"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", r"\ensuremath{-}") if s.startswith("-") else s


def macro(name: str, value: str) -> str:
    if not name.isalpha():
        raise ValueError(f"macro names must be letters only: {name}")
    return f"\\newcommand{{\\{name}}}{{{value}}}"


# --- loaders ----------------------------------------------------------------------------------
def load(root: Path = REPO) -> dict:
    an = Path(root) / "results" / "analysis"
    return {
        "prot": pd.read_csv(Path(root) / "results" / "protocols_10seeds.csv"),
        "paired": pd.read_csv(an / "protocols_paired_10seeds.csv"),
        "comp": pd.read_csv(an / "t16" / "comparison.csv"),
        "comp_paired": pd.read_csv(an / "t16" / "comparison_paired.csv"),
        "dev": pd.read_csv(an / "t14" / "dev_protocols.csv"),
        "fresh": pd.read_csv(an / "t16" / "fresh.csv"),
        "helps": pd.read_csv(an / "t16" / "where_it_helps.csv"),
        "gaps": pd.read_csv(an / "t16" / "gaps.csv"),
        "rho": pd.read_csv(an / "t16" / "where_it_helps_rho.csv"),
        "val": pd.read_csv(an / "story" / "summary_val.csv"),
        "test": pd.read_csv(an / "story" / "summary_test.csv"),
        "matched": pd.read_csv(an / "t14" / "matched_test.csv"),
        "alpha": pd.read_csv(an / "t14" / "alpha_curve_test.csv"),
        "p1": pd.read_csv(an / "p1_appendix_p1_vs_p2.csv"),
        "repro": pd.read_csv(an / "p1_appendix_reproduction.csv"),
        "ni_abl": pd.read_csv(an / "t16" / "no_info_ablation.csv"),
        "ni_abl_seed": pd.read_csv(an / "t16" / "no_info_ablation_per_seed.csv"),
        "read_first": pd.read_csv(an / "t16" / "read_first_val_test.csv"),
        "ni_auc": pd.read_csv(an / "t16" / "no_info_auc.csv"),
        "ni_filter": pd.read_csv(an / "t16" / "no_info_filter.csv"),
        "ni_rates": pd.read_csv(an / "t16" / "no_info_rates.csv"),
        "ni_rules": pd.read_csv(an / "t16" / "no_info_rules.csv"),
        "ni_corr": pd.read_csv(an / "t16" / "no_info_corr.csv"),
        "ni_deciles": pd.read_csv(an / "t16" / "no_info_deciles.csv"),
    }


READ_FIRST = "TDA read first (no self-vote)"


def read_first(d, split: str, metric: str, dataset="mean") -> float:
    """Actual read-before-write TDA runs (3 seeds). The mean is over the five design sets."""
    t = d["read_first"]
    t = t[(t.split == split) & (t.method == READ_FIRST)]
    return float(t[metric].mean() if dataset == "mean" else t[t.dataset == dataset][metric].item())


def p1_mean(d, method: str, metric: str, col: str) -> float:
    """Protocol table (direct replay, 3 seeds), mean over the five design sets' rows (not the rounded mean row)."""
    p = d["p1"]
    rows = p[(p.dataset != "mean") & (p.method == method) & (p.metric == metric)]
    if len(rows) != 5:
        raise KeyError((method, metric))
    return float(rows[col].mean())


def prot(d, dataset, method, metric, col="mean", protocol="P2"):
    p = d["prot"]
    row = p[(p.protocol == protocol) & (p.dataset == dataset) & (p.method == method) & (p.metric == metric)]
    if len(row) != 1:
        raise KeyError((protocol, dataset, method, metric))
    return float(row.iloc[0][col])


def paired(d, method, metric, reference="tda", dataset="mean"):
    p = d["paired"]
    row = p[(p.comparison == "within_P2") & (p.dataset == dataset) & (p.method == method)
            & (p.reference == reference) & (p.metric == metric)]
    if len(row) != 1:
        raise KeyError((method, reference, metric, dataset))
    return row.iloc[0]


def comp(d, method, metric, dataset="mean"):
    """Comparison-table value. The mean is taken over the per-dataset rows, as in every other table, so that the
    CSV's three-decimal mean row is not rounded twice (5.6546 -> 5.655 -> "5.66")."""
    c = d["comp"]
    if dataset == "mean":
        rows = c[(c.dataset != "mean") & (c.method == method)]
        if len(rows) != 5:
            raise KeyError((method, dataset))
        return float(rows[f"{metric}_mean"].mean())
    row = c[(c.dataset == dataset) & (c.method == method)]
    if len(row) != 1:
        raise KeyError((method, dataset))
    return float(row.iloc[0][f"{metric}_mean"])


def comp_paired(d, method, metric, dataset="mean"):
    c = d["comp_paired"]
    row = c[(c.dataset == dataset) & (c.method == method) & (c.metric == metric)]
    if len(row) != 1:
        raise KeyError((method, metric, dataset))
    return row.iloc[0]


def generalization(d) -> pd.DataFrame:
    """One row per held-out dataset (EXCLUDED left out), tagged near / far / other by `held_tier`: CLIP, TDA, gate
    accuracy / ECE (P2, 3 seeds) and deltas vs TDA."""
    rows = []
    dev = d["dev"]
    for ds in dict.fromkeys(dev.dataset[(dev.dataset != "mean") & ~dev.dataset.isin(EXCLUDED)]):
        row = {"dataset": ds, "tier": held_tier(ds)}
        for m, k in (("clipzs", "clip"), ("tda", "tda"), ("cal_tda_agree", "gate")):
            for metric in ("accuracy", "ece"):
                sel = dev[(dev.protocol == "P2") & (dev.dataset == ds) & (dev.method == m) & (dev.metric == metric)]
                row[f"{k}_{metric}"] = float(sel.iloc[0]["mean"])
        rows.append(row)
    fresh = d["fresh"][(d["fresh"].dataset != "mean") & ~d["fresh"].dataset.isin(EXCLUDED)]
    for _, r in fresh.iterrows():
        rows.append({"dataset": r.dataset, "tier": held_tier(r.dataset),
                     **{f"{k}_{m}": r[f"{k}_{m}"] for k in ("clip", "tda", "gate") for m in ("accuracy", "ece")}})
    g = pd.DataFrame(rows)
    g = g.iloc[g.tier.map({"near": 0, "far": 1, "other": 2}).argsort(kind="stable")].reset_index(drop=True)
    g["d_accuracy"] = g.gate_accuracy - g.tda_accuracy
    g["d_ece"] = g.gate_ece - g.tda_ece
    return g


# --- numbers ----------------------------------------------------------------------------------
def numbers(d) -> dict:
    n = {}
    for m, key in (("clipzs", "Clip"), ("tda", "Tda"), ("cal_tda_agree", "Gate")):
        n[f"main{key}Acc"] = f(prot(d, "mean", m, "accuracy"))
        n[f"main{key}Ece"] = f(prot(d, "mean", m, "ece"))
    for metric, key in (("accuracy", "Acc"), ("ece", "Ece")):
        p = paired(d, "cal_tda_agree", metric)
        n[f"mainD{key}"] = f(p.mean_diff, 2, sign=True)
        n[f"mainD{key}Lo"] = f(p.ci90_lo, 2, sign=True)
        n[f"mainD{key}Hi"] = f(p.ci90_hi, 2, sign=True)
        n[f"mainD{key}Wins"] = f"{int(p.wins)}/{int(p.n)}"
    n["mainSeeds"] = str(int(paired(d, "cal_tda_agree", "ece").n))
    n["mainTdaOverClipAcc"] = f(prot(d, "mean", "tda", "accuracy") - prot(d, "mean", "clipzs", "accuracy"), 2, sign=True)
    n["mainTdaGainAcc"] = f(prot(d, "mean", "tda", "accuracy") - prot(d, "mean", "clipzs", "accuracy"))
    n["mainTdaOverClipEce"] = f(prot(d, "mean", "tda", "ece") - prot(d, "mean", "clipzs", "ece"), 2, sign=True)
    n["mainGateOverClipAcc"] = f(prot(d, "mean", "cal_tda_agree", "accuracy") - prot(d, "mean", "clipzs", "accuracy"), 2, sign=True)
    n["mainGateOverClipEce"] = f(prot(d, "mean", "cal_tda_agree", "ece") - prot(d, "mean", "clipzs", "ece"), 2, sign=True)

    for method, key in (("CLIP", "Clip"), ("TDA", "Tda"), ("agreement gate", "Gate"), ("read-first TDA", "ReadFirst"),
                        ("TDA + T (val)", "TdaT"), ("TDA + ZAEC", "TdaZaec"), ("gate + ZAEC", "GateZaec"),
                        ("DMN-weak", "Dmn"), ("ECALP", "Ecalp")):
        try:
            n[f"cmp{key}Acc"] = f(comp(d, method, "accuracy"))
            n[f"cmp{key}Ece"] = f(comp(d, method, "ece"))
        except KeyError:
            n[f"cmp{key}Acc"] = n[f"cmp{key}Ece"] = "n/a"
    excess = comp(d, "TDA", "ece") - comp(d, "CLIP", "ece")
    n["replayDiffMax"] = f(max(abs(p1_mean(d, m, metric, "mean_P2") - comp(d, cm, metric))
                               for m, cm in (("tda", "TDA"), ("cal_tda_agree", "agreement gate"))
                               for metric in ("accuracy", "ece")))
    n["readFirstExcessShare"] = f(100 * (comp(d, "TDA", "ece") - comp(d, "read-first TDA", "ece")) / excess, 0)

    n.update(sweep_numbers(d["matched"], d["alpha"]))

    v = d["val"][d["val"].dataset != "mean"]
    vm = d["val"][d["val"].dataset == "mean"].iloc[0]
    n["valSelfMin"], n["valSelfMax"] = f(v.self_vote_share.min(), 0), f(v.self_vote_share.max(), 0)
    n["valAgreeSelfMin"], n["valAgreeSelfMax"] = f(v.p_agree_given_self.min(), 0), f(v.p_agree_given_self.max(), 0)
    n["valAgreeNoSelfMin"], n["valAgreeNoSelfMax"] = f(v.p_agree_given_no_self.min(), 0), f(v.p_agree_given_no_self.max(), 0)
    n["valTdaEce"], n["valClipEce"], n["valTdaAcc"] = f(vm.tda_ece), f(vm.clip_ece), f(vm.tda_acc)
    n["valNoSelfEce"], n["valNoSelfAcc"] = f(read_first(d, "val", "ece")), f(read_first(d, "val", "acc"))
    n["valAgreeShare"] = f(vm.agree_share, 0)
    n["valAgreeNegMoved"] = f(vm.agree_neg_moved, 1)
    n["valAgreeTdaFlips"] = f(vm.agree_tda_flips, 1)
    for part, key in (("agree", "Agree"), ("disagree", "Disagree")):
        for k, kk in (("base", "Base"), ("tda", "Tda")):
            n[f"val{key}{kk}Acc"] = f(vm[f"{k}_acc_{part}"])
            n[f"val{key}{kk}Gap"] = f(vm[f"{k}_gap_{part}"], 1, sign=True)
    n["valGateAcc"], n["valGateEce"] = f(vm.gate_acc), f(vm.gate_ece)

    tm = d["test"][d["test"].dataset == "mean"].iloc[0]
    for k, key in (("clip", "Clip"), ("base", "Base"), ("gpos", "Gpos"), ("gate", "Gate"), ("tda", "Tda")):
        n[f"abl{key}Acc"], n[f"abl{key}Ece"] = f(tm[f"{k}_acc"]), f(tm[f"{k}_ece"])
    n["testAgreeShare"] = f(tm.agree_share, 0)
    n["testAgreeBaseGap"], n["testAgreeTdaGap"] = f(tm.base_gap_agree, 1, sign=True), f(tm.tda_gap_agree, 1, sign=True)
    n["testAgreeBaseAcc"], n["testAgreeTdaAcc"] = f(tm.base_acc_agree), f(tm.tda_acc_agree)

    g = generalization(d)
    g = g[g.tier != "other"]
    for tier, key in (("near", "Near"), ("far", "Far"), (None, "All")):
        sub = g if tier is None else g[g.tier == tier]
        n[f"gen{key}N"] = str(len(sub))
        n[f"gen{key}DEce"] = f(sub.d_ece.mean(), 2, sign=True)
        n[f"gen{key}DAcc"] = f(sub.d_accuracy.mean(), 2, sign=True)
        n[f"gen{key}EceWins"] = str(int((sub.d_ece < 0).sum()))
        for k, kk in (("tda", "Tda"), ("gate", "Gate")):
            n[f"gen{key}{kk}OverClipAcc"] = f((sub[f"{k}_accuracy"] - sub.clip_accuracy).mean(), 2, sign=True)
            n[f"gen{key}{kk}OverClipEce"] = f((sub[f"{k}_ece"] - sub.clip_ece).mean(), 2, sign=True)
    n["genAllMedDEce"] = f(g.d_ece.median(), 2, sign=True)
    n["totalDatasets"] = str(len(g) + len(d["test"][d["test"].dataset != "mean"]))
    for tier, key in (("near", "Near"), ("far", "Far")):
        n[f"{tier}Names"] = ", ".join(NAMES.get(x, x) for x in g[g.tier == tier].dataset)
        for k, kk in (("clip", "Clip"), ("tda", "Tda"), ("gate", "Gate")):
            for m, mk in (("accuracy", "Acc"), ("ece", "Ece")):
                n[f"gen{key}Mean{kk}{mk}"] = f(g[g.tier == tier][f"{k}_{m}"].mean())
    n.update(far_spread_numbers(g))
    n["reproMaxDelta"] = f(d["repro"].delta_gpu_vs_benchmark.abs().max())
    n["genWorstDAcc"] = f(g.d_accuracy.min(), 2, sign=True)
    worst = g.loc[g.d_accuracy.idxmin()]
    n["genWorstDropAcc"], n["genWorstDropName"] = f(-worst.d_accuracy), NAMES.get(worst.dataset, worst.dataset)
    far = g[g.tier == "far"]
    n["genFarTdaGainAcc"] = f((far.tda_accuracy - far.clip_accuracy).mean())
    n["genWorstDEce"] = f(g.d_ece.max(), 2, sign=True)

    rho = d["rho"]
    for sig, key in (("conf", "Conf"), ("over06", "Over"), ("share", "Share"), ("clip_gap", "Gap")):
        for ds, dk in (("all", "All"), ("test", "Test"), ("near", "Near"), ("far", "Far")):
            row = rho[(rho.signal == sig) & (rho.datasets == ds)]
            n[f"rho{key}{dk}"] = f(float(row.iloc[0].rho), 2, sign=True) if len(row) else "n/a"
    n["signalHead"], n["confCut"] = str(SIGNAL_HEAD), f(CONF_CUT, 1)
    n["rhoN"] = str(int(rho[(rho.datasets == "all")].iloc[0].n))
    n.update(no_info_numbers(d["ni_abl"], d["ni_auc"], d["ni_filter"]))
    n.update(rule_numbers(d["ni_rules"], d["ni_filter"]))
    n.update(agree_range_numbers(d["ni_abl_seed"]))
    n.update(corr_numbers(d["ni_corr"], d["ni_deciles"]))
    n.update(gap_numbers(d["gaps"], d["helps"]))
    return n


def sweep_values(matched: pd.DataFrame, curve: pd.DataFrame) -> dict:
    """TDA's alpha sweep (mean of the five test sets): its accuracy and ECE at alpha = 0, and the alpha at which its
    accuracy first reaches the gate's, with its ECE there (interpolated on the curve)."""
    gate = matched[(matched.dataset == "mean") & matched.method.str.startswith("agreement gate")].iloc[0]
    c = curve[curve.dataset == "mean"].sort_values("alpha").reset_index(drop=True)
    i = int((c.acc >= gate.acc).to_numpy().argmax())
    if i == 0 or c.acc[i] < gate.acc:
        raise ValueError("TDA's alpha curve never crosses the gate's accuracy from below")
    w = (gate.acc - c.acc[i - 1]) / (c.acc[i] - c.acc[i - 1])
    return {"zero_acc": c.acc[0], "zero_ece": c.ece[0], "gate_acc": gate.acc,
            "alpha": c.alpha[i - 1] + w * (c.alpha[i] - c.alpha[i - 1]),
            "ece": c.ece[i - 1] + w * (c.ece[i] - c.ece[i - 1])}


def sweep_numbers(matched: pd.DataFrame, curve: pd.DataFrame) -> dict:
    v = sweep_values(matched, curve)
    return {"sweepZeroAcc": f(v["zero_acc"], 1), "sweepZeroEce": f(v["zero_ece"], 1), "sweepGateAcc": f(v["gate_acc"], 1),
            "sweepMatchAlpha": f(v["alpha"], 1), "matchedTdaEce": f(v["ece"], 1)}


def far_spread(g: pd.DataFrame):
    """The far sets with the three largest ECE reductions (gate - TDA), and the rest."""
    far = g[g.tier == "far"].sort_values("d_ece")
    return far.iloc[:3], far.iloc[3:]


def far_spread_numbers(g: pd.DataFrame) -> dict:
    """How the far tier's mean ECE change (gate - TDA) is spread: the three largest reductions and the range of the rest."""
    big, rest = far_spread(g)
    names = [NAMES.get(x, x) for x in big.dataset]
    return {"farBigNames": ", ".join(names[:-1]) + " and " + names[-1],
            "farBigLo": f(-big.d_ece.max(), 1), "farBigHi": f(-big.d_ece.min(), 1),
            "farRestLo": f(rest.d_ece.min(), 1, sign=True), "farRestHi": f(rest.d_ece.max(), 1, sign=True),
            "farRestN": str(len(rest))}


NI_TIERS = (("val", "Val"), ("test", "Test"), ("near", "Near"), ("far", "Far"))
ABL_TIERS = (("test", "Test"), ("near", "Near"), ("far", "Far"))
NI_VARIANTS = (("clip", "CLIP"), ("base", "+ negative cache ($b$)"),
               ("apos", "$b$ + positive cache on agreeing samples"),
               ("gate", "$b$ + positive cache on disagreeing samples (gate)"),
               ("rand", "$b$ + positive cache on as many random samples"),
               ("tda", "$b$ + positive cache on all samples (TDA)"))


def gap_numbers(gaps: pd.DataFrame, helps: pd.DataFrame) -> dict:
    """Over- vs underconfidence on the design test, near and far sets: how often TDA raises CLIP's confidence - accuracy
    gap and the gate lowers TDA's, tier means, and the sets where TDA's ECE beats the gate's, split by CLIP's sign."""
    g = gaps[gaps.tier.isin(("test", "near", "far"))]
    n = {"gapN": str(len(g)), "gapTdaUpN": str(int((g.tda_gap > g.clip_gap).sum())),
         "gapGateDownN": str(int((g.gate_gap < g.tda_gap).sum())),
         "gapTdaOverN": str(int((g.tda_gap > 0).sum())), "gapGateOverN": str(int((g.gate_gap > 0).sum()))}
    for tier, tk in (("test", "Test"), ("near", "Near"), ("far", "Far")):
        sub = g[g.tier == tier]
        for k, kk in (("clip", "Clip"), ("tda", "Tda"), ("gate", "Gate")):
            n[f"gap{tk}{kk}"] = f(sub[f"{k}_gap"].mean(), 1, sign=True)
    lose = g.merge(helps[["dataset", "d_ece"]], on="dataset")
    lose = lose[lose.d_ece > 0]
    n["gapLoseUnderNames"] = ", ".join(NAMES.get(x, x) for x in lose[lose.clip_gap < 0].dataset)
    n["gapLoseOtherNames"] = ", ".join(NAMES.get(x, x) for x in lose[lose.clip_gap >= 0].dataset)
    n["gapLoseN"], n["gapLoseUnderN"] = str(len(lose)), str(int((lose.clip_gap < 0).sum()))
    n["gapUnderN"] = str(int((g.clip_gap < 0).sum()))
    n["gapLoseGateMax"] = f(lose.gate_gap.max(), 1, sign=True)  # the gate's largest gap where it loses in ECE
    return n


def rule_numbers(rules: pd.DataFrame, filt: pd.DataFrame) -> dict:
    """Best gain of any val-chosen one-score rule on val and on each held-out tier, and the oracle's range over the
    tiers other than val."""
    orc = filt[filt.tier != "val"].oracle_block
    n = {f"niRuleBest{t.capitalize()}": f(rules[rules.tier == t].gain.max(), 2, sign=True)
         for t in ("val", "near", "far")}
    n["niOracleBlockLo"], n["niOracleBlockHi"] = f(orc.min(), 1), f(orc.max(), 1)
    return n


def agree_range_numbers(abl_seed: pd.DataFrame) -> dict:
    """Range over datasets (every tier) of the share of samples whose cache agrees with CLIP, mean over seeds."""
    per = abl_seed.groupby(["tier", "dataset"]).agree_share.mean()
    return {"niAgreeDsLo": f(per.min(), 0), "niAgreeDsHi": f(per.max(), 0)}


CORR_SCORES = (("clip_conf", "Clip"), ("cache_conf", "Cache"), ("one_minus_js", "Js"))


def corr_numbers(corrs: pd.DataFrame, deciles: pd.DataFrame) -> dict:
    """Held-out correlations between a flip being right and the three discussed scores, and the lowest right-flip share
    of any of their deciles."""
    held = ("near", "far")
    c = corrs[corrs.measure.isin([k for k, _ in CORR_SCORES]) & corrs.tier.isin(held)]
    n = {f"niCorr{kk}{t.capitalize()}": f(c[(c.measure == k) & (c.tier == t)].rho.item(), 2, sign=True)
         for k, kk in CORR_SCORES for t in held}
    n["niCorrMaxAbs"] = f(c.rho.abs().max(), 2)
    n["niCorrDsMaxAbs"] = f(max(c.rho_min.abs().max(), c.rho_max.abs().max()), 2)
    dec = deciles[deciles.measure.isin([k for k, _ in CORR_SCORES])]
    for t in held:
        n[f"niDecileMin{t.capitalize()}"] = f(dec[dec.tier == t].right_pct.min(), 0)
        n[f"niDecileAllMin{t.capitalize()}"] = f(deciles[deciles.tier == t].right_pct.min(), 0)
    return n


def no_info_numbers(abl: pd.DataFrame, aucs: pd.DataFrame, filt: pd.DataFrame) -> dict:
    """Macros for the ablation by agreement group and the flip-information analysis."""
    a, n = abl.set_index("tier"), {}
    tiers = a.loc[[t for t, _ in NI_TIERS]]
    n["niAgreeShareLo"], n["niAgreeShareHi"] = f(tiers.agree_share.min(), 0), f(tiers.agree_share.max(), 0)
    for tier, tk in NI_TIERS:
        r = a.loc[tier]
        n[f"niAgreeShare{tk}"] = f(r.agree_share, 0)
        for part, pk in (("agree", "Agree"), ("disagree", "Dis")):
            for k, kk in (("base", "Base"), ("tda", "Tda")):
                n[f"ni{pk}{kk}Acc{tk}"] = f(r[f"{k}_acc_{part}"], 1)
                n[f"ni{pk}{kk}Gap{tk}"] = f(r[f"{k}_gap_{part}"], 1, sign=True)
        fr = filt.set_index("tier").loc[tier]
        n[f"niFilter{tk}"] = f(fr.filter_gain, 2, sign=True)
        n[f"niFilterU{tk}"] = f(abs(fr.filter_gain))
        n[f"niOracleBlock{tk}"], n[f"niOraclePick{tk}"] = f(fr.oracle_block, 1), f(fr.oracle_pick, 1)
        n[f"niCombAuc{tk}"] = f(aucs[(aucs.measure == "combined") & (aucs.tier == tier)].auc.item())
    for tier, tk in ABL_TIERS:
        for k, kk in (("gate", "Gate"), ("rand", "Rand"), ("tda", "Tda")):
            n[f"abl{kk}Acc{tk}"], n[f"abl{kk}Ece{tk}"] = f(a.loc[tier, f"{k}_acc"], 1), f(a.loc[tier, f"{k}_ece"], 1)
    held = a.loc[[t for t, _ in ABL_TIERS]]
    n["niAposMaxDAcc"] = f((held.apos_acc - held.base_acc).abs().max(), 2)
    n["niAgreeMaxDAcc"] = f((a.tda_acc_agree - a.base_acc_agree).abs().max(), 2)
    single = aucs[aucs.measure != "combined"]
    held = single[single.tier.isin(("near", "far"))].auc
    n["niHeldAucLo"], n["niHeldAucHi"] = f(held.min()), f(held.max())
    n["niFilterMax"] = f(filt.filter_gain.abs().max(), 2)
    n["niOraclePickLo"], n["niOraclePickHi"] = f(filt.oracle_pick.min(), 1), f(filt.oracle_pick.max(), 1)
    n["niThresh"] = f(float(filt.threshold.iloc[0]))
    return n


# --- tables -----------------------------------------------------------------------------------
def pm(mean, std, d=2):
    return f"{f(mean, d)}\\,{{\\scriptsize$\\pm${f(std, d)}}}"


def tab_main(d) -> str:
    dsets = ["dtd", "flower102", "pets", "eurosat", "aircraft", "mean"]
    head = " & ".join(NAMES.get(s, "Mean") for s in dsets)
    lines = ["\\begin{tabular}{ll" + "r" * len(dsets) + "}", "\\toprule",
             f" & method & {head} \\\\", "\\midrule"]
    for metric, lab in (("accuracy", "Acc."), ("ece", "ECE")):
        pick = max if metric == "accuracy" else min
        best = {s: pick(round(prot(d, s, m, metric), 2) for m in METHOD) for s in dsets}
        for i, (m, name) in enumerate(METHOD.items()):
            cells = []
            for s in dsets:
                mean, std = prot(d, s, m, metric), prot(d, s, m, metric, "std")
                cell = f(mean) if m == "clipzs" else pm(mean, std)  # CLIP does not depend on the stream order
                cells.append(f"\\textbf{{{cell}}}" if round(mean, 2) == best[s] else cell)
            lines.append(f"{lab if i == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if metric == "accuracy" else "\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


COMPARISON_ROWS = (("CLIP", "CLIP"), ("TDA", "TDA"), ("agreement gate", "\\textbf{agreement gate}"),
                   ("read-first TDA", "read-first TDA"), ("TDA + T (val)", "TDA + temperature (fit on val)"),
                   ("TDA + ZAEC", "TDA + ZAEC"), ("gate + ZAEC", "gate + ZAEC"),
                   ("DMN-weak", "DMN-weak"), ("ECALP", "ECALP"))


def tab_comparison(d) -> str:
    lines = ["\\begin{tabular}{lrrrrc}", "\\toprule",
             "method & Acc. & ECE & $\\Delta$Acc. & $\\Delta$ECE & ECE wins \\\\", "\\midrule"]
    for method, label in COMPARISON_ROWS:
        try:
            acc, ece = comp(d, method, "accuracy"), comp(d, method, "ece")
        except KeyError:
            lines.append(f"{label} & \\multicolumn{{5}}{{c}}{{(running)}} \\\\")
            continue
        if method == "TDA":
            lines.append(f"{label} & {f(acc)} & {f(ece)} & -- & -- & -- \\\\")
            continue
        pa, pe = comp_paired(d, method, "accuracy"), comp_paired(d, method, "ece")
        lines.append(f"{label} & {f(acc)} & {f(ece)} & {f(pa.mean_diff, 2, True)} & {f(pe.mean_diff, 2, True)}"
                     f" & {int(pe.wins)}/{int(pe.n)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def tab_comparison_per_dataset(d) -> str:
    dsets = ["dtd", "flower102", "pets", "eurosat", "aircraft"]
    lines = ["\\begin{tabular}{l" + "r" * len(dsets) + "}", "\\toprule",
             "ECE (\\%) & " + " & ".join(NAMES[s] for s in dsets) + " \\\\", "\\midrule"]
    for method, label in COMPARISON_ROWS:
        cells = []
        for s in dsets:
            try:
                cells.append(f(comp(d, method, "ece", s)))
            except KeyError:
                cells.append("--")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


GEN_TIERS = ("near", "far")


def bold_if(v: float, on: bool) -> str:
    return "\\textbf{" + f(v) + "}" if on else f(v)


def tab_generalization(d, tiers=GEN_TIERS) -> str:
    """Per-dataset rows plain; on each tier's mean row the best accuracy and the best ECE are bold. Last block: the
    signed confidence - accuracy gap (positive: overconfident)."""
    g = generalization(d).merge(d["gaps"].drop(columns="tier"), on="dataset", how="left")
    lines = ["\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
             " & & \\multicolumn{3}{c}{accuracy} & \\multicolumn{3}{c}{ECE} & \\multicolumn{3}{c}{gap} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
             "tier & dataset & CLIP & TDA & gate & CLIP & TDA & gate & CLIP & TDA & gate \\\\", "\\midrule"]
    gap = lambda r: " & ".join(f(r[f"{k}_gap"], 1, sign=True) for k in ("clip", "tda", "gate"))  # noqa: E731
    tiers = [t for t in tiers if (g.tier == t).any()]
    for tier in tiers:
        sub = g[g.tier == tier]
        for i, (_, r) in enumerate(sub.iterrows()):
            lines.append(f"{tier if i == 0 else ''} & {NAMES.get(r.dataset, r.dataset)} & {f(r.clip_accuracy)} & "
                         f"{f(r.tda_accuracy)} & {f(r.gate_accuracy)} & {f(r.clip_ece)} & {f(r.tda_ece)} & "
                         f"{f(r.gate_ece)} & {gap(r)} \\\\")
        mean = sub.mean(numeric_only=True)
        acc = [round(mean[f"{k}_accuracy"], 2) for k in ("clip", "tda", "gate")]
        ece = [round(mean[f"{k}_ece"], 2) for k in ("clip", "tda", "gate")]
        cells = [bold_if(v, v == max(acc)) for v in acc] + [bold_if(v, v == min(ece)) for v in ece] + [gap(mean)]
        lines.append(f" & \\emph{{mean ({len(sub)})}} & " + " & ".join(cells) + " \\\\")
        lines.append("\\bottomrule" if tier == tiers[-1] else "\\midrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def tab_p1(d) -> str:
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{accuracy} & \\multicolumn{2}{c}{ECE} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "method & per-dataset & single & per-dataset & single \\\\", "\\midrule"]
    for m, label in (("clipzs", "CLIP"), ("tda", "TDA"), ("cal_tda_agree", "agreement gate")):
        cells = [f(p1_mean(d, m, metric, col)) for metric in ("accuracy", "ece") for col in ("mean_P1", "mean_P2")]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def tab_ablation_groups(d) -> str:
    """Accuracy / ECE of the cache ablation on each evaluation tier, and the share of samples the gate opens on."""
    a = d["ni_abl"].set_index("tier")
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "accuracy / ECE & design (test) & near & far \\\\", "\\midrule"]
    for k, label in NI_VARIANTS:
        cells = [f"{f(a.loc[t, f'{k}_acc'], 1)} / {f(a.loc[t, f'{k}_ece'], 1)}" for t, _ in ABL_TIERS]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    share = " & ".join(f"{f(a.loc[t, 'disagree_share'], 0)}\\%" for t, _ in ABL_TIERS)
    lines += [f"samples the gate opens on (disagreeing) & {share} \\\\", "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def write(root: Path = REPO, out: Path = OUT) -> list[Path]:
    d = load(root)
    out.mkdir(parents=True, exist_ok=True)
    files = {"numbers.tex": "\n".join(macro(k, v) for k, v in numbers(d).items()) + "\n",
             "tab_main.tex": tab_main(d) + "\n", "tab_comparison.tex": tab_comparison(d) + "\n",
             "tab_generalization.tex": tab_generalization(d) + "\n",
             "tab_comparison_per_dataset.tex": tab_comparison_per_dataset(d) + "\n",
             "tab_p1.tex": tab_p1(d) + "\n", "tab_ablation_groups.tex": tab_ablation_groups(d) + "\n"}
    paths = []
    for name, text in files.items():
        (out / name).write_text("% generated by analysis/report_assets.py; do not edit\n" + text, encoding="utf-8")
        paths.append(out / name)
    return paths


if __name__ == "__main__":
    print("wrote", *[p.name for p in write()])
