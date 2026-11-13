"""Is there information beyond binary agreement? (T16; P2, seeds 0/1/2, exact trace re-evaluation)

1. Ablation by agreement group. TDA's positive cache is split into the samples where its top class agrees with CLIP's
   (gate closed) and those where it disagrees (gate open). Variants per tier: CLIP, + negative cache, + positive cache
   on agreeing samples only, + positive cache on disagreeing samples only (= the gate), + both (= TDA), and the gated
   positive cache without the negative cache. Per group: accuracy and confidence-accuracy gap without / with the
   positive cache.
2. Flip information. A flip is a sample where TDA's decision differs from CLIP's; it is right if TDA is right and
   wrong if CLIP was right. Label-free per-sample measures of CLIP's scores, the cache's scores and their relation are
   scored alone (AUC, right-flip rate by octile) and combined (logistic regression fitted on the val flips). The
   combination also defines a label-free flip filter (revert to CLIP below a threshold chosen on val); its accuracy
   change is compared with the oracle filter (block every wrong flip) and the oracle pick (right one of CLIP's and the
   cache's top class).

    python analysis/no_info.py  -> results/analysis/t16/no_info_*.csv, analysis/figures/no_info_*.png
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.adaptive_alpha import disagree_mask, load_runs, metrics, trace_path  # noqa: E402
from utils.feature_cache import DATASETS, DEV_DATASETS, FRESH_DATASETS  # noqa: E402

ALPHA = 2.0
OUT = REPO / "results" / "analysis" / "t16"
FIG = REPO / "analysis" / "figures"
REPORT_FIG = REPO / "report" / "figures"
TIERS = ("val", "test", "dev", "fresh")
VARIANTS = ("clip", "base", "apos", "gate", "tda", "gpos")
LABELS = {"clip": "CLIP", "base": "+ negative cache", "apos": "+ positive cache, agreeing samples only",
          "gate": "+ positive cache, disagreeing samples only (gate)", "tda": "+ positive cache, all samples (TDA)",
          "gpos": "CLIP + positive cache, disagreeing only (no negative)"}
# measure -> (group, label); every measure is label-free
MEASURES = {
    "clip_conf": ("CLIP", "confidence"),
    "clip_margin": ("CLIP", "top-1 − top-2 margin"),
    "clip_p_cache_class": ("CLIP", "prob. of the cache's class"),
    "cache_conf": ("cache", "confidence"),
    "cache_mass": ("cache", "mass of its top class"),
    "cache_support": ("cache", "weight on CLIP's top-1"),
    "rho": ("relation", "ρ(cache, CLIP) over classes"),
    "one_minus_js": ("relation", "1 − JS(cache, CLIP)"),
    "weight_gap": ("relation", "cache weight: CLIP top-1 − own top"),
    "alpha_needed": ("relation", "α needed to flip"),
}
TIER_COLORS = {"val": "#9ecae1", "test": "#3182bd", "dev": "#31a354", "fresh": "#e6550d"}
MIN_CLASS = 20  # flips of each kind for a per-dataset AUC


def variant_logits(run, alpha: float = ALPHA) -> dict:
    dis = disagree_mask(run)[:, None]
    agree = ~dis
    return {"clip": run.clip, "base": run.base, "apos": run.base + alpha * agree * run.cache,
            "gate": run.base + alpha * dis * run.cache, "tda": run.base + alpha * run.cache,
            "gpos": run.clip + alpha * dis * run.cache}


def _gap(logits, labels, mask) -> float:
    p = logits[mask].softmax(-1)
    return 100 * float((p.max(-1).values - (p.argmax(-1) == labels[mask]).double()).mean())


def _acc(logits, labels, mask) -> float:
    return 100 * float((logits[mask].argmax(-1) == labels[mask]).double().mean())


def ablation_rows(runs, tier: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        lg = variant_logits(r)
        row = {"tier": tier, "dataset": r.dataset, "seed": r.seed}
        for k, o in lg.items():
            m = metrics(o.softmax(-1), r.labels)
            row[f"{k}_acc"], row[f"{k}_ece"] = m["acc"], m["ece"]
        dis = disagree_mask(r)
        for part, mask in (("agree", ~dis), ("disagree", dis)):
            row[f"{part}_share"] = 100 * float(mask.double().mean())
            for k in ("base", "tda"):
                row[f"{k}_acc_{part}"] = _acc(lg[k], r.labels, mask) if mask.any() else np.nan
                row[f"{k}_gap_{part}"] = _gap(lg[k], r.labels, mask) if mask.any() else np.nan
            row[f"changed_{part}"] = 100 * float((lg["tda"].argmax(-1) != lg["base"].argmax(-1))[mask].double().mean()) \
                if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def flip_records(run, alpha: float = ALPHA) -> pd.DataFrame:
    """One row per flip (TDA's decision != CLIP's) whose outcome is decided: right (TDA right) or wrong (CLIP right)."""
    y = run.labels
    p = run.clip.softmax(-1)
    top2 = p.topk(2, dim=-1)
    p0 = top2.indices[:, 0]
    p1 = (run.base + alpha * run.cache).argmax(-1)
    c = run.cache.clamp_min(0)
    has = c.sum(-1) > 0
    pc = torch.where(has[:, None], c / c.sum(-1, keepdim=True).clamp_min(1e-300), torch.full_like(c, 1 / c.shape[1]))
    ct = pc.argmax(-1)
    pcs = 0.99 * pc + 0.01 / c.shape[1]
    m = 0.5 * (p + pcs)
    js = 0.5 * ((p * (p.clamp_min(1e-300).log() - m.log())).sum(-1) + (pcs * (pcs.log() - m.log())).sum(-1)) / np.log(2)
    gap = run.base.gather(1, p0[:, None]).squeeze(1) - run.base.gather(1, ct[:, None]).squeeze(1)
    adv = (c.gather(1, ct[:, None]) - c.gather(1, p0[:, None])).squeeze(1)
    s = {"clip_conf": top2.values[:, 0], "clip_margin": top2.values[:, 0] - top2.values[:, 1],
         "clip_p_cache_class": p.gather(1, ct[:, None]).squeeze(1),
         "cache_conf": pc.max(-1).values, "cache_mass": c.max(-1).values,
         "cache_support": pc.gather(1, p0[:, None]).squeeze(1),
         "rho": _corr(run.cache, run.clip), "one_minus_js": 1 - js,
         "weight_gap": pc.gather(1, p0[:, None]).squeeze(1) - pc.max(-1).values,
         "alpha_needed": torch.where(adv > 0, gap / adv.clamp_min(1e-12), torch.full_like(gap, float("inf")))}
    flip = p1 != p0
    right, wrong = flip & (p1 == y), flip & (p0 == y)
    keep = right | wrong
    out = pd.DataFrame({k: v[keep].numpy() for k, v in s.items()})
    out["right"] = right[keep].numpy()
    out["dataset"], out["seed"], out["n"] = run.dataset, run.seed, len(y)
    return out


def auc(right, score) -> float:
    """P(score of a right flip > score of a wrong flip), ties counted half."""
    right = np.asarray(right, bool)
    r = rankdata(np.asarray(score, float))
    n1, n0 = right.sum(), (~right).sum()
    return float((r[right].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _features(frame: pd.DataFrame) -> np.ndarray:
    x = frame[list(MEASURES)].to_numpy(float).copy()
    j = list(MEASURES).index("alpha_needed")
    x[:, j] = np.arcsinh(np.nan_to_num(x[:, j], posinf=1e4))  # heavy-tailed and signed
    return x


def fit_logistic(frame: pd.DataFrame, l2: float = 1e-3):
    """Logistic regression on standardized features; returns a scorer frame -> P(right)."""
    x = _features(frame)
    mu, sd = x.mean(0), x.std(0) + 1e-12
    xt = torch.tensor((x - mu) / sd)
    yt = torch.tensor(frame.right.to_numpy(float))
    w = torch.zeros(xt.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = xt @ w[:-1] + w[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    wd = w.detach()

    def score(f: pd.DataFrame) -> np.ndarray:
        z = torch.tensor((_features(f) - mu) / sd) @ wd[:-1] + wd[-1]
        return torch.sigmoid(z).numpy()

    return score


def filter_gain(frame: pd.DataFrame, p_right: np.ndarray, threshold: float) -> pd.DataFrame:
    """Accuracy change (points) per dataset and seed from reverting flips with P(right) < threshold to CLIP."""
    blocked = p_right < threshold
    g = frame.assign(gain=np.where(blocked, np.where(frame.right, -1.0, 1.0), 0.0))
    return (g.groupby(["dataset", "seed"]).agg(gain=("gain", "sum"), n=("n", "first"))
            .assign(accuracy=lambda d: 100 * d.gain / d.n).reset_index())


def oracle_rows(run, tier: str, alpha: float = ALPHA) -> dict:
    y = run.labels
    p0 = run.clip.argmax(-1)
    p1 = (run.base + alpha * run.cache).argmax(-1)
    ct = run.cache.clamp_min(0).argmax(-1)
    has = run.cache.clamp_min(0).sum(-1) > 0
    tda_ok = p1 == y
    block = tda_ok | ((p1 != p0) & (p0 == y))
    pick = tda_ok | (p0 == y) | (has & (ct == y))
    n = len(y)
    return {"tier": tier, "dataset": run.dataset, "seed": run.seed,
            "oracle_block": 100 * float((block & ~tda_ok).double().sum()) / n,
            "oracle_pick": 100 * float((pick & ~tda_ok).double().sum()) / n}


def tier_mean(frame: pd.DataFrame, col: str) -> float:
    """Unweighted mean over datasets per seed, then over seeds."""
    return float(frame.groupby("seed")[col].mean().mean())


def info_tables(flips: pd.DataFrame, oracle: pd.DataFrame, design: str = "val"):
    """AUC per measure and tier (pooled and range over datasets), octile right-flip rates, and the learned filter."""
    auc_rows, rate_rows = [], []
    for key, (group, label) in MEASURES.items():
        edges = np.unique(np.quantile(flips[key].replace(np.inf, np.nan).dropna(), np.linspace(0, 1, 9)))
        for tier, d in flips.groupby("tier"):
            per = [auc(dd.right, dd[key]) for _, dd in d.groupby("dataset")
                   if dd.right.sum() >= MIN_CLASS and (~dd.right).sum() >= MIN_CLASS]
            auc_rows.append({"measure": key, "group": group, "label": label, "tier": tier,
                             "auc": auc(d.right, d[key]), "auc_ds_min": min(per, default=np.nan),
                             "auc_ds_max": max(per, default=np.nan)})
            b = np.clip(np.digitize(d[key].clip(upper=edges[-1]), edges[1:-1]), 0, len(edges) - 2)
            for i in range(len(edges) - 1):
                sel = d.right[b == i]
                rate_rows.append({"measure": key, "tier": tier, "bin": i, "lo": edges[i], "hi": edges[i + 1],
                                  "n": len(sel), "right_pct": 100 * sel.mean() if len(sel) else np.nan})
    scorer = fit_logistic(flips[flips.tier == design])
    p = scorer(flips)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    dz = flips.tier == design
    val_gain = {t: tier_mean(filter_gain(flips[dz], p[dz.to_numpy()], t), "accuracy") for t in grid}
    best = max(val_gain, key=val_gain.get)
    comb = []
    for tier, d in flips.groupby("tier"):
        sel = (flips.tier == tier).to_numpy()
        auc_rows.append({"measure": "combined", "group": "combined", "label": f"all ten combined (fitted on {design})",
                         "tier": tier, "auc": auc(d.right, p[sel]), "auc_ds_min": np.nan, "auc_ds_max": np.nan})
        o = oracle[oracle.tier == tier]
        comb.append({"tier": tier, "threshold": best, "filter_gain": tier_mean(filter_gain(d, p[sel], best), "accuracy"),
                     "oracle_block": tier_mean(o, "oracle_block"), "oracle_pick": tier_mean(o, "oracle_pick"),
                     "right_share": 100 * d.right.mean(), "flips": len(d)})
    return pd.DataFrame(auc_rows), pd.DataFrame(rate_rows), pd.DataFrame(comb)


def fig_auc(aucs: pd.DataFrame, path: Path):
    keys = list(MEASURES) + ["combined"]
    labels = [f"{MEASURES[k][0]}: {MEASURES[k][1]}" if k in MEASURES else "all ten combined (fit on val)"
              for k in keys]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    y = np.arange(len(keys))[::-1]
    for i, tier in enumerate(TIERS):
        d = aucs[aucs.tier == tier].set_index("measure").loc[keys]
        off = (i - 1.5) * 0.17
        ax.hlines(y + off, d.auc_ds_min, d.auc_ds_max, color=TIER_COLORS[tier], lw=1.6, alpha=0.4)
        ax.plot(d.auc, y + off, "o", color=TIER_COLORS[tier], ms=4.5, label=tier)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.axvspan(0.45, 0.55, color="#dddddd", alpha=0.6, zorder=0)
    ax.set_yticks(y, labels, fontsize=7.5)
    for i in (2.5, 5.5, 9.5):  # between the CLIP, cache, relation and combined groups
        ax.axhline(len(keys) - 1 - i, color="#999999", lw=0.6)
    ax.set_xlim(0.2, 0.82)
    ax.set_xlabel("AUC, right vs wrong flips (dot: pooled; line: range over datasets)", fontsize=7.5)
    ax.set_title("Label-free scores of CLIP, of the cache and of their relation\n"
                 "do not tell right flips from wrong ones on new datasets", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_rates(rates: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), sharey=True)
    for ax, (key, (group, label)) in zip(axes.flat, MEASURES.items()):
        for tier in TIERS:
            d = rates[(rates.measure == key) & (rates.tier == tier) & (rates.n >= 30)]
            ax.plot(d.bin, d.right_pct, "o-", color=TIER_COLORS[tier], ms=3, lw=1.2, label=tier)
        ax.axhline(50, color="k", lw=0.6, ls="--")
        ax.set_title(f"{group}: {label}", fontsize=8)
        ax.set_xticks(range(8), [f"{i + 1}" for i in range(8)], fontsize=7)
        ax.set_ylim(20, 100)
    for ax in axes[:, 0]:
        ax.set_ylabel("% right flips", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("octile of the score (all tiers)", fontsize=7.5)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace-root", default="replay_results/t14")
    ap.add_argument("--fresh-root", default="replay_results/t16_fresh")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)
    spec = {"val": (a.trace_root, DATASETS, "val"), "test": (a.trace_root, DATASETS, "test"),
            "dev": (a.trace_root, DEV_DATASETS, "test"), "fresh": (a.fresh_root, FRESH_DATASETS, "test")}
    abl, flips, orc, skipped = [], [], [], []
    for tier, (root, datasets, split) in spec.items():
        for ds in datasets:
            if not all(trace_path(root, ds, s, split, "tda_eq_clean").exists() for s in a.seeds):
                skipped.append(ds)
                continue
            runs = load_runs(root, [ds], a.seeds, split, "tda_eq_clean")
            abl.append(ablation_rows(runs, tier))
            flips += [flip_records(r).assign(tier=tier) for r in runs]
            orc += [oracle_rows(r, tier) for r in runs]
    abl, flips, orc = pd.concat(abl, ignore_index=True), pd.concat(flips, ignore_index=True), pd.DataFrame(orc)
    aucs, rates, comb = info_tables(flips, orc)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in abl.columns if c not in ("tier", "dataset", "seed")]
    abl_mean = abl.groupby(["tier", "seed"])[cols].mean().groupby("tier").mean().reset_index()
    abl.round(4).to_csv(out / "no_info_ablation_per_seed.csv", index=False)
    abl_mean.round(4).to_csv(out / "no_info_ablation.csv", index=False)
    aucs.round(4).to_csv(out / "no_info_auc.csv", index=False)
    rates.round(4).to_csv(out / "no_info_rates.csv", index=False)
    comb.round(4).to_csv(out / "no_info_filter.csv", index=False)
    FIG.mkdir(parents=True, exist_ok=True)
    fig_auc(aucs, FIG / "no_info_auc.png")
    fig_rates(rates, FIG / "no_info_rates.png")
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    for name in ("no_info_auc.png", "no_info_rates.png"):
        shutil.copy(FIG / name, REPORT_FIG / name)
    print("skipped (no full trace):", ", ".join(skipped) or "none")
    print(abl_mean.set_index("tier")[[f"{k}_{m}" for k in VARIANTS for m in ("acc", "ece")]].round(2).T)
    print(abl_mean.set_index("tier")[[c for c in cols if "agree" in c]].round(2).T)
    print(aucs.pivot(index="measure", columns="tier", values="auc")[list(TIERS)].round(3))
    print(comb.round(3))


if __name__ == "__main__":
    main()
