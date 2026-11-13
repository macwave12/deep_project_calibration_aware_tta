"""Every number and table of the course report (T16), generated from tracked CSVs.

Writes `report/generated/numbers.tex` (one `\\newcommand` per number the prose uses) and
`report/generated/tab_*.tex` (the tables). The prose never types a result by hand (RULES 9).

Sources (all tracked):
  results/protocols_10seeds.csv, results/analysis/protocols_paired_10seeds.csv   main table (test, 10 seeds)
  results/analysis/t16/comparison{,_paired}.csv                                comparison (test, 3 seeds)
  results/analysis/t14/dev_protocols.csv, results/analysis/t16/fresh.csv       generalization
  results/analysis/t16/where_it_helps{,_rho}.csv                               where it helps
  results/analysis/story/summary_{val,test}.csv                                diagnosis (val), ablation (test)
  results/analysis/t14/matched_test.csv                                        matched comparison
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

OUT = REPO / "report" / "generated"
RES = REPO / "results"
AN = RES / "analysis"
METHOD = {"clipzs": "CLIP", "tda": "TDA", "cal_tda_agree": "agreement gate"}
EARLIER = {"cal_tda_margin": "margin admission", "cal_tda_fusion": "probability fusion",
           "cal_tda_gate": "two-way-tie fusion gate", "cal_tda_temp": "LOO temperature",
           "cal_tda_gate_temp": "tie gate + LOO temperature"}


def f(x, d=2, sign=False) -> str:
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
        "rho": pd.read_csv(an / "t16" / "where_it_helps_rho.csv"),
        "val": pd.read_csv(an / "story" / "summary_val.csv"),
        "test": pd.read_csv(an / "story" / "summary_test.csv"),
        "matched": pd.read_csv(an / "t14" / "matched_test.csv"),
        "p1": pd.read_csv(an / "p1_appendix_p1_vs_p2.csv"),
        "repro": pd.read_csv(an / "p1_appendix_reproduction.csv"),
        "switch": pd.read_csv(an / "t16" / "switch.csv"),
        "switch_sweep": pd.read_csv(an / "t16" / "switch_sweep.csv"),
        "ni_abl": pd.read_csv(an / "t16" / "no_info_ablation.csv"),
        "ni_auc": pd.read_csv(an / "t16" / "no_info_auc.csv"),
        "ni_filter": pd.read_csv(an / "t16" / "no_info_filter.csv"),
        "ni_rates": pd.read_csv(an / "t16" / "no_info_rates.csv"),
    }


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
    c = d["comp"]
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
    """One row per dev and fresh dataset: CLIP, TDA, gate accuracy / ECE (P2, 3 seeds) and deltas vs TDA."""
    rows = []
    dev = d["dev"]
    for ds in dict.fromkeys(dev.dataset[dev.dataset != "mean"]):
        row = {"dataset": ds, "tier": "dev"}
        for m, k in (("clipzs", "clip"), ("tda", "tda"), ("cal_tda_agree", "gate")):
            for metric in ("accuracy", "ece"):
                sel = dev[(dev.protocol == "P2") & (dev.dataset == ds) & (dev.method == m) & (dev.metric == metric)]
                row[f"{k}_{metric}"] = float(sel.iloc[0]["mean"])
        rows.append(row)
    fresh = d["fresh"][d["fresh"].dataset != "mean"]
    for _, r in fresh.iterrows():
        rows.append({"dataset": r.dataset, "tier": "fresh",
                     **{f"{k}_{m}": r[f"{k}_{m}"] for k in ("clip", "tda", "gate") for m in ("accuracy", "ece")}})
    g = pd.DataFrame(rows)
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
    n["mainTdaOverClipEce"] = f(prot(d, "mean", "tda", "ece") - prot(d, "mean", "clipzs", "ece"), 2, sign=True)

    for method, key in (("CLIP", "Clip"), ("TDA", "Tda"), ("agreement gate", "Gate"), ("read-first TDA", "ReadFirst"),
                        ("TDA + T (val)", "TdaT"), ("TDA + ZAEC", "TdaZaec"), ("gate + ZAEC", "GateZaec"),
                        ("DMN-weak", "Dmn"), ("ECALP", "Ecalp")):
        try:
            n[f"cmp{key}Acc"] = f(comp(d, method, "accuracy"))
            n[f"cmp{key}Ece"] = f(comp(d, method, "ece"))
        except KeyError:
            n[f"cmp{key}Acc"] = n[f"cmp{key}Ece"] = "n/a"
    excess = comp(d, "TDA", "ece") - comp(d, "CLIP", "ece")
    n["readFirstExcessShare"] = f(100 * (comp(d, "TDA", "ece") - comp(d, "read-first TDA", "ece")) / excess, 0)

    m = d["matched"]
    gate = m[(m.dataset == "mean") & m.method.str.startswith("agreement gate")].iloc[0]
    n["matchedTdaEce"] = f(gate.tda_best_ece_at_same_acc)

    v = d["val"][d["val"].dataset != "mean"]
    vm = d["val"][d["val"].dataset == "mean"].iloc[0]
    n["valSelfMin"], n["valSelfMax"] = f(v.self_vote_share.min(), 0), f(v.self_vote_share.max(), 0)
    n["valAgreeSelfMin"], n["valAgreeSelfMax"] = f(v.p_agree_given_self.min(), 0), f(v.p_agree_given_self.max(), 0)
    n["valAgreeNoSelfMin"], n["valAgreeNoSelfMax"] = f(v.p_agree_given_no_self.min(), 0), f(v.p_agree_given_no_self.max(), 0)
    n["valTdaEce"], n["valNoSelfEce"], n["valClipEce"] = f(vm.tda_ece), f(vm.noself_ece), f(vm.clip_ece)
    n["valTdaAcc"], n["valNoSelfAcc"] = f(vm.tda_acc), f(vm.noself_acc)
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
    for tier, key in (("dev", "Dev"), ("fresh", "Fresh"), (None, "All")):
        sub = g if tier is None else g[g.tier == tier]
        n[f"gen{key}N"] = str(len(sub))
        n[f"gen{key}DEce"] = f(sub.d_ece.mean(), 2, sign=True)
        n[f"gen{key}DAcc"] = f(sub.d_accuracy.mean(), 2, sign=True)
        n[f"gen{key}EceWins"] = str(int((sub.d_ece < 0).sum()))
    n["totalDatasets"] = str(len(g) + len(d["test"][d["test"].dataset != "mean"]))
    n["freshNames"] = ", ".join(NAMES.get(x, x) for x in g[g.tier == "fresh"].dataset)
    n["genFreshMeanTdaEce"] = f(g[g.tier == "fresh"].tda_ece.mean())
    n["genFreshMeanGateEce"] = f(g[g.tier == "fresh"].gate_ece.mean())
    n["reproMaxDelta"] = f(d["repro"].delta_gpu_vs_benchmark.abs().max())
    n["genWorstDAcc"] = f(g.d_accuracy.min(), 2, sign=True)
    n["genWorstDEce"] = f(g.d_ece.max(), 2, sign=True)
    inet = g[g.dataset == "imagenet"]
    if len(inet):
        r = inet.iloc[0]
        n["inetTdaAcc"], n["inetTdaEce"] = f(r.tda_accuracy), f(r.tda_ece)
        n["inetGateAcc"], n["inetGateEce"] = f(r.gate_accuracy), f(r.gate_ece)

    rho = d["rho"]
    for sig, key in (("conf", "Conf"), ("over06", "Over"), ("share", "Share"), ("clip_gap", "Gap")):
        for ds, dk in (("all", "All"), ("fresh", "Fresh"), ("test+dev", "Seen")):
            row = rho[(rho.signal == sig) & (rho.datasets == ds)]
            n[f"rho{key}{dk}"] = f(float(row.iloc[0].rho), 2, sign=True) if len(row) else "n/a"
    n["signalHead"], n["confCut"] = str(SIGNAL_HEAD), f(CONF_CUT, 1)
    n["rhoN"] = str(int(rho[(rho.datasets == "all")].iloc[0].n))
    n.update(switch_numbers(d["switch"], d["switch_sweep"]))
    n.update(no_info_numbers(d["ni_abl"], d["ni_auc"], d["ni_filter"]))
    rates = d["ni_rates"]
    n["niAlphaMinRate"] = f(rates[(rates.measure == "alpha_needed") & (rates.n >= 30)].right_pct.min(), 1)
    return n


NI_TIERS = (("val", "Val"), ("test", "Test"), ("dev", "Dev"), ("fresh", "Fresh"))
NI_VARIANTS = (("clip", "CLIP"), ("base", "+ negative cache ($b$)"),
               ("apos", "$b$ + positive cache, agreeing only"), ("gate", "$b$ + positive cache, disagreeing only (gate)"),
               ("tda", "$b$ + positive cache, all (TDA)"), ("gpos", "CLIP + positive cache, disagreeing only"))


def no_info_numbers(abl: pd.DataFrame, aucs: pd.DataFrame, filt: pd.DataFrame) -> dict:
    """Macros for the ablation by agreement group and the flip-information analysis."""
    a, n = abl.set_index("tier"), {}
    for tier, tk in NI_TIERS:
        r = a.loc[tier]
        n[f"niAgreeShare{tk}"] = f(r.agree_share, 0)
        for part, pk in (("agree", "Agree"), ("disagree", "Dis")):
            for k, kk in (("base", "Base"), ("tda", "Tda")):
                n[f"ni{pk}{kk}Acc{tk}"] = f(r[f"{k}_acc_{part}"], 1)
                n[f"ni{pk}{kk}Gap{tk}"] = f(r[f"{k}_gap_{part}"], 1, sign=True)
        fr = filt.set_index("tier").loc[tier]
        n[f"niFilter{tk}"] = f(round(fr.filter_gain, 2) + 0.0, 2, sign=True)  # no "-0.00"
        n[f"niOracleBlock{tk}"], n[f"niOraclePick{tk}"] = f(fr.oracle_block, 1), f(fr.oracle_pick, 1)
        n[f"niCombAuc{tk}"] = f(aucs[(aucs.measure == "combined") & (aucs.tier == tier)].auc.item())
    n["niAposMaxDAcc"] = f((a.apos_acc - a.base_acc).abs().max(), 2)
    n["niAgreeMaxDAcc"] = f((a.tda_acc_agree - a.base_acc_agree).abs().max(), 2)
    single = aucs[aucs.measure != "combined"]
    held = single[single.tier.isin(("dev", "fresh"))].auc
    n["niHeldAucLo"], n["niHeldAucHi"] = f(held.min()), f(held.max())
    n["niFilterMax"] = f(filt.filter_gain.abs().max(), 2)
    n["niOraclePickLo"], n["niOraclePickHi"] = f(filt.oracle_pick.min(), 1), f(filt.oracle_pick.max(), 1)
    n["niThresh"] = f(float(filt.threshold.iloc[0]))
    return n


SWITCH_METHODS = (("TDA", "Tda"), ("TDA + ZAEC", "Zaec"), ("agreement gate", "Gate"), ("confidence switch", "Sw"))
TIERS = (("val", "Val"), ("dev", "Dev"), ("test", "Test"), ("fresh", "Fresh"))  # design tiers first


def switch_numbers(sw: pd.DataFrame, sweep: pd.DataFrame) -> dict:
    """Macros for the confidence switch: threshold, per-tier means, the datasets it switches on, ECE deltas, and the
    thresholds the fresh tier alone would have preferred (within 0.1 ECE of its minimum)."""
    n = {"swThresh": f(float(sw.threshold.iloc[0])), "swPlateauLo": f(float(sw.plateau_lo.iloc[0])),
         "swPlateauHi": f(float(sw.plateau_hi.iloc[0]))}
    mean = sw[sw.dataset == "mean"]
    for tier, tk in TIERS:
        for method, mk in SWITCH_METHODS:
            row = mean[(mean.tier == tier) & (mean.method == method)].iloc[0]
            n[f"sw{mk}{tk}Ece"], n[f"sw{mk}{tk}Acc"] = f(row.ece), f(row.accuracy)
        gate = mean[(mean.tier == tier) & (mean.method == "agreement gate")].iloc[0]
        swr = mean[(mean.tier == tier) & (mean.method == "confidence switch")].iloc[0]
        n[f"swD{tk}Ece"] = f(swr.ece - gate.ece, 2, sign=True)
    per = sw[(sw.dataset != "mean") & (sw.method == "confidence switch")]
    on = per[per.switch_on_pct > 50]
    n["swOnNames"] = ", ".join(NAMES.get(x, x) for x in dict.fromkeys(on[on.tier != "val"].dataset))
    n["swOnN"] = str(on[on.tier != "val"].dataset.nunique())
    n["swHeldN"] = str(per[per.tier != "val"].dataset.nunique())
    n["swFreshOnN"] = str(on[on.tier == "fresh"].dataset.nunique())
    off = per[per.switch_on_pct <= 50]
    n["swMinOnConf"] = f(on.clip_conf.min() / 100)
    n["swMaxOffConf"] = f(off.clip_conf.max() / 100)
    fr = sweep[sweep.tier == "fresh"]
    best = fr[fr.ece <= fr.ece.min() + 0.1]
    n["swFreshBestEce"] = f(fr.ece.min())
    n["swFreshBestLo"], n["swFreshBestHi"] = f(best.threshold.min()), f(best.threshold.max())
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
        for i, (m, name) in enumerate(METHOD.items()):
            cells = []
            for s in dsets:
                mean, std = prot(d, s, m, metric), prot(d, s, m, metric, "std")
                cell = f(mean) if m == "clipzs" else pm(mean, std)
                cells.append(f"\\textbf{{{cell}}}" if m == "cal_tda_agree" else cell)
            lines.append(f"{lab if i == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if metric == "accuracy" else "\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


COMPARISON_ROWS = (("CLIP", "CLIP zero-shot"), ("TDA", "TDA (reference)"), ("agreement gate", "\\textbf{agreement gate (ours)}"),
                   ("read-first TDA", "read-before-write TDA"), ("TDA + T (val)", "TDA + temperature (val-fitted)"),
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


def tab_generalization(d) -> str:
    g = generalization(d)
    lines = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
             " & & \\multicolumn{3}{c}{accuracy} & \\multicolumn{3}{c}{ECE} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "tier & dataset & CLIP & TDA & gate & CLIP & TDA & gate \\\\", "\\midrule"]
    for tier in ("dev", "fresh"):
        sub = g[g.tier == tier]
        for i, (_, r) in enumerate(sub.iterrows()):
            best = "\\textbf{" + f(r.gate_ece) + "}" if r.gate_ece < r.tda_ece else f(r.gate_ece)
            lines.append(f"{tier if i == 0 else ''} & {NAMES.get(r.dataset, r.dataset)} & {f(r.clip_accuracy)} & "
                         f"{f(r.tda_accuracy)} & {f(r.gate_accuracy)} & {f(r.clip_ece)} & {f(r.tda_ece)} & {best} \\\\")
        mean = sub.mean(numeric_only=True)
        lines.append(f" & \\emph{{mean ({len(sub)})}} & {f(mean.clip_accuracy)} & {f(mean.tda_accuracy)} & "
                     f"{f(mean.gate_accuracy)} & {f(mean.clip_ece)} & {f(mean.tda_ece)} & {f(mean.gate_ece)} \\\\")
        lines.append("\\midrule" if tier == "dev" else "\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def tab_earlier(d) -> str:
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "variant (P2, test, mean of 5) & Acc. & ECE & $\\Delta$Acc. & $\\Delta$ECE \\\\", "\\midrule"]
    for m, label in {**{"cal_tda_agree": "\\textbf{agreement gate (final)}"}, **EARLIER}.items():
        acc, ece = prot(d, "mean", m, "accuracy"), prot(d, "mean", m, "ece")
        pa, pe = paired(d, m, "accuracy"), paired(d, m, "ece")
        lines.append(f"{label} & {f(acc)} & {f(ece)} & {f(pa.mean_diff, 2, True)} & {f(pe.mean_diff, 2, True)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def tab_p1(d) -> str:
    p = d["p1"][d["p1"].dataset == "mean"]
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "method (test, mean of 5, 3 seeds) & Acc.\\ P1 & Acc.\\ P2 & ECE P1 & ECE P2 \\\\", "\\midrule"]
    for m, label in (("clipzs", "CLIP"), ("tda", "TDA"), ("cal_tda_agree", "agreement gate")):
        a = p[(p.method == m) & (p.metric == "accuracy")].iloc[0]
        e = p[(p.method == m) & (p.metric == "ece")].iloc[0]
        lines.append(f"{label} & {f(a.mean_P1)} & {f(a.mean_P2)} & {f(e.mean_P1)} & {f(e.mean_P2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def tab_switch(d) -> str:
    mean = d["switch"][d["switch"].dataset == "mean"]
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "ECE (acc.), P2, 3 seeds & val$^\\ast$ & dev$^\\ast$ & test & fresh \\\\", "\\midrule"]
    for method, _ in SWITCH_METHODS:
        cells = []
        for tier, _ in TIERS:
            row = mean[(mean.tier == tier) & (mean.method == method)].iloc[0]
            cells.append(f"{f(row.ece)} ({f(row.accuracy, 1)})")
        lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def tab_ablation_groups(d) -> str:
    """Accuracy / ECE of the cache ablation on every tier, then the agreeing / disagreeing groups."""
    a = d["ni_abl"].set_index("tier")
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Accuracy / ECE, P2, 3 seeds & val & test & dev & fresh \\\\", "\\midrule"]
    for k, label in NI_VARIANTS:
        cells = [f"{f(a.loc[t, f'{k}_acc'], 1)} / {f(a.loc[t, f'{k}_ece'], 1)}" for t, _ in NI_TIERS]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for part, label in (("agree", "agreeing"), ("disagree", "disagreeing")):
        share = " & ".join(f"{f(a.loc[t, f'{part}_share'], 0)}\\%" for t, _ in NI_TIERS)
        lines.append(f"{label}: share of samples & {share} \\\\")
        acc = " & ".join(f"{f(a.loc[t, f'base_acc_{part}'], 1)} $\\to$ {f(a.loc[t, f'tda_acc_{part}'], 1)}"
                         for t, _ in NI_TIERS)
        lines.append(f"\\quad accuracy, $b \\to$ TDA & {acc} \\\\")
        gap = " & ".join(f"{f(a.loc[t, f'base_gap_{part}'], 1, sign=True)} $\\to$ "
                         f"{f(a.loc[t, f'tda_gap_{part}'], 1, sign=True)}" for t, _ in NI_TIERS)
        lines.append(f"\\quad conf.$-$acc.\\ gap, $b \\to$ TDA & {gap} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def tab_switch_design(d) -> str:
    """Design sets by CLIP confidence: CLIP's accuracy minus confidence (labels, diagnostic) and the switch state."""
    sw = d["switch"]
    per = sw[sw.tier.isin(("val", "dev")) & (sw.method == "confidence switch") & (sw.dataset != "mean")]
    per = per.sort_values("clip_conf")
    lines = ["\\begin{tabular}{llrrc}", "\\toprule",
             "Design set & split & CLIP conf. & CLIP acc.$-$conf. & switch on \\\\", "\\midrule"]
    for _, r in per.iterrows():
        on = "yes" if r.switch_on_pct > 50 else "no"
        lines.append(f"{NAMES.get(r.dataset, r.dataset)} & {r.tier} & {f(r.clip_conf, 1)} & "
                     f"{f(r.clip_gap, 1, sign=True)} & {on} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def write(root: Path = REPO, out: Path = OUT) -> list[Path]:
    d = load(root)
    out.mkdir(parents=True, exist_ok=True)
    files = {"numbers.tex": "\n".join(macro(k, v) for k, v in numbers(d).items()) + "\n",
             "tab_main.tex": tab_main(d) + "\n", "tab_comparison.tex": tab_comparison(d) + "\n",
             "tab_generalization.tex": tab_generalization(d) + "\n", "tab_earlier.tex": tab_earlier(d) + "\n",
             "tab_comparison_per_dataset.tex": tab_comparison_per_dataset(d) + "\n",
             "tab_p1.tex": tab_p1(d) + "\n", "tab_switch.tex": tab_switch(d) + "\n",
             "tab_switch_design.tex": tab_switch_design(d) + "\n",
             "tab_ablation_groups.tex": tab_ablation_groups(d) + "\n"}
    paths = []
    for name, text in files.items():
        (out / name).write_text("% generated by analysis/report_assets.py; do not edit\n" + text, encoding="utf-8")
        paths.append(out / name)
    return paths


if __name__ == "__main__":
    print("wrote", *[p.name for p in write()])
