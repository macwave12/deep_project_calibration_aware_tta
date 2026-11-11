"""T14: dataset-independent adaptive cache weight alpha(x), evaluated offline from traces.

A `--trace full` replay of TDA (`configs/method/tda_equivalent.yaml`) stores, per step, the
logits as the method used them: `base_logits` (CLIP minus negative cache) and the unscaled
positive-cache logits `c`. Cache contents never depend on alpha (tested in
`tests/test_cal_tda_trace.py`), so any per-sample rule gives exactly
`softmax(base + alpha(x) * c)` without re-running the stream.

Labels are joined here, after the stream (RULES 2); every signal is label-free.

Library: `load_runs`, `signals`, `nll_matrix`, rule families (`RULES`), `fit_rule`, `evaluate`.
CLI: `python analysis/adaptive_alpha.py {signals,oracle,ladder} ...` -> results/analysis/adaptive_alpha/
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.calibration import compute_accuracy, compute_ece  # noqa: E402
from utils.feature_cache import FeatureStore  # noqa: E402

ALPHA_GRID = torch.round(torch.arange(0.0, 4.0001, 0.1), decimals=1)
SHOT_CAPACITY = 3  # TDA's positive-cache slots per class
OUT = REPO / "results" / "analysis" / "adaptive_alpha"
SIGNALS = ["clip_conf", "clip_conf_rel", "clip_ent", "clip_margin", "cache_conf", "cache_conf_rel",
           "cache_top_mass", "agree", "kl", "js", "cache_support_clip_top", "rank_clip_top",
           "self_vote"]


@dataclass
class Run:
    dataset: str
    seed: int
    base: torch.Tensor      # (N, C) float64: CLIP logits minus negative cache, as used
    cache: torch.Tensor     # (N, C) float64: unscaled positive-cache logits
    clip: torch.Tensor      # (N, C) float64: raw CLIP logits
    labels: torch.Tensor    # (N,) int64, stream order (analysis only)
    self_vote: torch.Tensor  # (N,) bool
    s: dict = field(default_factory=dict)  # signals, filled by `signals`

    @property
    def n_classes(self) -> int:
        return self.base.shape[1]


def trace_path(root: Path, dataset: str, seed: int, split: str, name: str) -> Path:
    d = Path(root)
    if split != "test":
        d = d / split
    if seed != 0:
        d = d / f"seed{seed}"
    return d / f"{dataset}_{name}.trace.pt"


def load_runs(root, datasets, seeds, split="val", name="tda_eq_clean", features="features") -> list[Run]:
    runs = []
    for dataset in datasets:
        store = FeatureStore(Path(features), dataset)
        labels_all = store.labels(split)
        for seed in seeds:
            trace = torch.load(trace_path(root, dataset, seed, split, name), weights_only=True)
            if not trace.get("full_vectors"):
                raise ValueError(f"{dataset} seed {seed}: trace lacks full vectors (--trace full)")
            st = trace["steps"]
            order = store.permutation(split, seed)
            run = Run(dataset, seed, st["base_logits"].double(), st["cache_logits"].double(),
                      st["clip_logits"].double(), labels_all[order], st["in_pos_cache"].bool())
            signals(run)
            runs.append(run)
    return runs


def signals(run: Run) -> dict:
    """Label-free, scale-free per-sample signals (T14 guardrail 1)."""
    C = run.n_classes
    log_c = math.log(C)
    p = run.clip.softmax(-1)
    top2 = p.topk(2, dim=-1)
    clip_top = top2.indices[:, 0]
    c = run.cache.clamp_min(0)
    mass = c.sum(-1, keepdim=True)
    has = mass.squeeze(-1) > 0
    # c_k = sum_{i in k} exp(-beta (1 - cos_i)): kernel class weights, so c / sum(c) is the
    # cache's own posterior (not softmax(c), which is near-flat).
    p_cache = torch.where(has[:, None], c / mass.clamp_min(1e-300), torch.full_like(c, 1.0 / C))
    p_cache_s = 0.99 * p_cache + 0.01 / C
    m = 0.5 * (p + p_cache_s)
    kl = (p * (p.clamp_min(1e-300).log() - p_cache_s.log())).sum(-1) / log_c
    js = 0.5 * ((p * (p.clamp_min(1e-300).log() - m.log())).sum(-1)
                + (p_cache_s * (p_cache_s.log() - m.log())).sum(-1)) / math.log(2)
    support = p_cache.gather(1, clip_top[:, None]).squeeze(1)
    rank = (p_cache > support[:, None]).sum(-1).double() / (C - 1)
    cache_conf = p_cache.max(-1).values
    run.s = {
        "clip_conf": top2.values[:, 0],
        "clip_conf_rel": (top2.values[:, 0] - 1.0 / C) / (1 - 1.0 / C),
        "clip_ent": -(p * p.clamp_min(1e-300).log()).sum(-1) / log_c,
        "clip_margin": top2.values[:, 0] - top2.values[:, 1],
        "cache_conf": cache_conf,
        "cache_conf_rel": (cache_conf - 1.0 / C) / (1 - 1.0 / C),
        "cache_top_mass": c.max(-1).values / SHOT_CAPACITY,
        "agree": (p_cache.argmax(-1) == clip_top).double() * has.double(),
        "kl": kl,
        "js": js,
        "cache_support_clip_top": support,
        "rank_clip_top": rank,
        "self_vote": run.self_vote.double(),
    }
    run.s = {k: v.double() for k, v in run.s.items()}
    return run.s


def probs_at(run: Run, alpha) -> torch.Tensor:
    """softmax(base + alpha * c); alpha scalar or (N,)."""
    a = torch.as_tensor(alpha, dtype=torch.float64)
    if a.dim() == 1:
        a = a[:, None]
    return (run.base + a * run.cache).softmax(-1)


def nll_matrix(run: Run, grid=ALPHA_GRID) -> torch.Tensor:
    """(N, len(grid)) per-sample NLL at each constant alpha."""
    out = []
    y = run.labels[:, None]
    for a in grid:
        lp = (run.base + float(a) * run.cache).log_softmax(-1)
        out.append(-lp.gather(1, y).squeeze(1))
    return torch.stack(out, 1)


def roc_auc_score(positive, score) -> float:
    """AUROC via the Mann-Whitney U statistic (ties get average ranks)."""
    from scipy.stats import rankdata
    positive = np.asarray(positive, dtype=bool)
    ranks = rankdata(score)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    pn, yn = probs.numpy(), labels.numpy()
    nll = float(-probs.gather(1, labels[:, None]).clamp_min(1e-300).log().mean())
    return {"acc": 100 * compute_accuracy(pn, yn), "ece": 100 * compute_ece(pn, yn), "nll": nll}


# --- rule families ------------------------------------------------------------------------
# Each rule maps a dict of signals (tensors) to alpha(x) >= 0 from a flat parameter vector.

def _sp(x):
    return torch.nn.functional.softplus(x)


@dataclass
class Rule:
    name: str
    inputs: tuple            # signal names
    n_params: int
    fn: callable              # (params, [signal tensors]) -> alpha
    init: callable            # () -> initial params tensor
    complexity: int           # ladder position

    def alpha(self, params, s: dict) -> torch.Tensor:
        return self.fn(params, [s[k] for k in self.inputs])

    def temperature(self, params) -> torch.Tensor:
        """Global temperature: the last parameter is log T (amendment 2026-11-10)."""
        return params[-1].exp()

    def logits(self, params, run) -> torch.Tensor:
        a = self.alpha(params, run.s)
        if a.numel() == 1:
            a = a.expand(run.labels.shape[0])
        return (run.base + a[:, None] * run.cache) / self.temperature(params)


def const_rule():
    return Rule("const", (), 1, lambda p, xs: _sp(p[0]).expand(1), lambda: torch.tensor([1.9]), 0)


def logistic_rule(sig):
    # alpha = A * sigmoid(k * (s - t)); k free in sign, so monotone either way.
    return Rule(f"logistic[{sig}]", (sig,), 3,
                lambda p, xs: _sp(p[0]) * torch.sigmoid(p[1] * (xs[0] - p[2])),
                lambda: torch.tensor([2.0, 0.0, 0.5]), 2)


def step_rule(sig, t):
    return Rule(f"step[{sig}<{t:.3g}]", (sig,), 2,
                lambda p, xs: torch.where(xs[0] < t, _sp(p[0]), _sp(p[1])),
                lambda: torch.tensor([1.9, 1.9]), 1)


def logistic2_rule(s1, s2):
    return Rule(f"logistic2[{s1},{s2}]", (s1, s2), 4,
                lambda p, xs: _sp(p[0]) * torch.sigmoid(p[1] * xs[0] + p[2] * xs[1] + p[3]),
                lambda: torch.tensor([2.0, 0.0, 0.0, 0.0]), 3)


def linear_all_rule(sigs, mean, std):
    sigs = tuple(sigs)
    m = torch.tensor(mean, dtype=torch.float64)
    sd = torch.tensor(std, dtype=torch.float64)
    return Rule(f"logistic_all[{len(sigs)}]", sigs, len(sigs) + 2,
                lambda p, xs: _sp(p[0]) * torch.sigmoid(
                    ((torch.stack(xs, 1) - m) / sd) @ p[2:2 + len(sigs)] + p[1]),
                lambda: torch.cat([torch.tensor([2.0, 0.0]), torch.zeros(len(sigs))]), 4)


def dataset_loss(rule: Rule, params, runs) -> torch.Tensor:
    """Mean NLL with equal weight per dataset (pre-registered, DECISIONS 2026-11-10)."""
    per_ds = {}
    for r in runs:
        lp = rule.logits(params, r).log_softmax(-1)
        per_ds.setdefault(r.dataset, []).append(-lp.gather(1, r.labels[:, None]).mean())
    return torch.stack([torch.stack(v).mean() for v in per_ds.values()]).mean()


def fit_rule(rule: Rule, runs, steps=200) -> torch.Tensor:
    params = torch.cat([rule.init().double(), torch.zeros(1, dtype=torch.float64)])
    params = params.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([params], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = dataset_loss(rule, params, runs)
        loss.backward()
        return loss

    opt.step(closure)
    return params.detach()


def evaluate(rule: Rule, params, runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        a = rule.alpha(params, r.s)
        m = metrics(rule.logits(params, r).softmax(-1), r.labels)
        rows.append({"dataset": r.dataset, "seed": r.seed, "rule": rule.name,
                     "alpha_mean": float(a.mean()), "T": float(rule.temperature(params)), **m})
    return pd.DataFrame(rows)


def baseline_rows(runs, alphas=(0.0, 2.0)) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = metrics(r.clip.softmax(-1), r.labels)
        rows.append({"dataset": r.dataset, "seed": r.seed, "rule": "clip", "alpha_mean": 0.0, **m})
        for a in alphas:
            m = metrics(probs_at(r, a), r.labels)
            rows.append({"dataset": r.dataset, "seed": r.seed, "rule": f"alpha={a:g}",
                         "alpha_mean": a, **m})
        nll = nll_matrix(r)
        best = nll.argmin(1)
        oracle = ALPHA_GRID[best]
        m = metrics(probs_at(r, oracle.double()), r.labels)
        rows.append({"dataset": r.dataset, "seed": r.seed, "rule": "oracle", "alpha_mean": float(oracle.mean()), **m})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per rule: per-dataset mean over seeds, then the unweighted dataset mean."""
    per_ds = frame.groupby(["rule", "dataset"], as_index=False)[["acc", "ece", "nll", "alpha_mean"]].mean()
    mean = per_ds.groupby("rule", as_index=False)[["acc", "ece", "nll", "alpha_mean"]].mean()
    mean.insert(1, "dataset", "mean")
    return pd.concat([per_ds, mean], ignore_index=True)


# --- CLI ----------------------------------------------------------------------------------

def _runs(args):
    return load_runs(args.trace_root, args.datasets, args.seeds, args.split, args.name, args.features)


def cmd_oracle(args):
    """Steps 3-4: oracle alpha*, benefit vs signals, signal ranking."""
    from scipy.stats import spearmanr

    runs = _runs(args)
    rows, rank_rows = [], []
    i2 = int(torch.nonzero(ALPHA_GRID == 2.0)[0])
    for r in runs:
        nll = nll_matrix(r)
        a_star = ALPHA_GRID[nll.argmin(1)].double()
        benefit = nll[:, 0] - nll[:, i2]
        df = pd.DataFrame({k: v.numpy() for k, v in r.s.items()})
        df["alpha_star"] = a_star.numpy()
        df["benefit"] = benefit.numpy()
        df["nll_0"] = nll[:, 0].numpy()
        df["nll_2"] = nll[:, i2].numpy()
        df["nll_star"] = nll.min(1).values.numpy()
        df["dataset"], df["seed"] = r.dataset, r.seed
        rows.append(df)
        for sname in SIGNALS:
            x = df[sname].to_numpy()
            pos = (df["benefit"] > 0).to_numpy()
            auc = roc_auc_score(pos, x) if 0 < pos.sum() < len(pos) and np.ptp(x) > 0 else np.nan
            rho = spearmanr(x, df["alpha_star"]).correlation if np.ptp(x) > 0 else np.nan
            rank_rows.append({"dataset": r.dataset, "seed": r.seed, "signal": sname,
                              "auroc_benefit": auc, "spearman_alpha_star": rho})
    samples = pd.concat(rows, ignore_index=True)
    ranking = pd.DataFrame(rank_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out / f"samples_{args.split}.parquet") if _has_parquet() else \
        samples.to_csv(out / f"samples_{args.split}.csv.gz", index=False)
    rk = ranking.groupby(["signal", "dataset"], as_index=False)[["auroc_benefit", "spearman_alpha_star"]].mean()
    pooled = rk.groupby("signal", as_index=False)[["auroc_benefit", "spearman_alpha_star"]].agg(["mean", "min", "max"])
    pooled.columns = ["_".join(c).strip("_") for c in pooled.columns]
    rk.round(4).to_csv(out / f"signal_ranking_{args.split}.csv", index=False)
    pooled.round(4).to_csv(out / f"signal_ranking_pooled_{args.split}.csv", index=False)
    base = summarize(baseline_rows(runs))
    base.round(4).to_csv(out / f"baselines_{args.split}.csv", index=False)
    print(pooled.round(3).sort_values("auroc_benefit_mean").to_string(index=False))
    print(base[base.dataset == "mean"].round(3).to_string(index=False))


def binned_optimal(runs, signal, edges, grid=ALPHA_GRID) -> pd.DataFrame:
    """Per dataset and bin of `signal`: the constant alpha minimising the bin's mean NLL
    (pooled over seeds), the bin's NLL at alpha 0 / 2 / optimum, and its share of samples.
    Unlike the per-sample alpha*, this is an estimand a rule can actually reach."""
    rows = []
    by_ds = {}
    for r in runs:
        by_ds.setdefault(r.dataset, []).append(r)
    i2 = int(torch.nonzero(grid == 2.0)[0])
    for dataset, rs in by_ds.items():
        nll = torch.cat([nll_matrix(r, grid) for r in rs])
        s = torch.cat([r.s[signal] for r in rs])
        correct0 = torch.cat([(r.clip.argmax(-1) == r.labels) for r in rs]).double()
        b = torch.bucketize(s, torch.as_tensor(edges[1:-1], dtype=torch.float64))
        for k in range(len(edges) - 1):
            m = b == k
            if m.sum() < 20:
                continue
            curve = nll[m].mean(0)
            j = int(curve.argmin())
            rows.append({"dataset": dataset, "signal": signal, "bin": k,
                         "lo": edges[k], "hi": edges[k + 1], "s_mean": float(s[m].mean()),
                         "share": float(m.double().mean()), "n": int(m.sum()),
                         "clip_acc": float(correct0[m].mean()),
                         "alpha_opt": float(grid[j]), "nll_0": float(curve[0]),
                         "nll_2": float(curve[i2]), "nll_opt": float(curve[j])})
    return pd.DataFrame(rows)


def cmd_binned(args):
    runs = _runs(args)
    edges = np.linspace(0, 1, 11)
    frames = [binned_optimal(runs, s, edges) for s in args.signals]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True)
    frame.round(4).to_csv(out / f"binned_alpha_{args.split}.csv", index=False)
    for s in args.signals:
        piv = frame[frame.signal == s].pivot(index="bin", columns="dataset", values="alpha_opt")
        print(f"== {s}: bin-optimal alpha (rows = bins of width 0.1)")
        print(piv.to_string())


STEP_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def ladder_candidates(train, signals=SIGNALS):
    """All rules per rung, built from training runs only (thresholds, standardization)."""
    pooled = {s: torch.cat([r.s[s] for r in train]) for s in signals}
    rungs = {0: [const_rule()], 1: [], 2: [logistic_rule(s) for s in signals], 4: []}
    for s in signals:
        for q in STEP_QUANTILES:
            thr = float(torch.quantile(pooled[s].float(), q))
            if float(pooled[s].min()) < thr < float(pooled[s].max()):
                rungs[1].append(step_rule(s, thr))
    mean = [float(pooled[s].mean()) for s in signals]
    std = [float(pooled[s].std()) or 1.0 for s in signals]
    rungs[4].append(linear_all_rule(signals, mean, std))
    return rungs


def fit_ladder(train, log=print):
    """Best rule per rung by training loss; rung 3 pairs the four best rung-2 signals."""
    rungs = ladder_candidates(train)
    best = {}
    scored2 = []
    for rung in (0, 1, 2, 4):
        results = []
        for rule in rungs[rung]:
            params = fit_rule(rule, train)
            loss = float(dataset_loss(rule, params, train))
            results.append((loss, rule, params))
            if rung == 2:
                scored2.append((loss, rule.inputs[0]))
        results.sort(key=lambda x: x[0])
        best[rung] = results[0]
        log(f"    rung {rung}: {results[0][1].name} train NLL {results[0][0]:.4f}")
    top = [s for _, s in sorted(scored2)[:4]]
    results = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            rule = logistic2_rule(top[i], top[j])
            params = fit_rule(rule, train)
            results.append((float(dataset_loss(rule, params, train)), rule, params))
    results.sort(key=lambda x: x[0])
    best[3] = results[0]
    log(f"    rung 3: {results[0][1].name} train NLL {results[0][0]:.4f}")
    return dict(sorted(best.items()))


def cmd_ladder(args):
    runs = _runs(args)
    datasets = list(dict.fromkeys(r.dataset for r in runs))
    rows, params_rows = [], []
    folds = [(d, [r for r in runs if r.dataset != d], [r for r in runs if r.dataset == d])
             for d in datasets] + [("all", runs, runs)]
    for held, train, test in folds:
        print(f"fold held-out={held}", flush=True)
        best = fit_ladder(train)
        for rung, (loss, rule, params) in best.items():
            ev = evaluate(rule, params, test)
            ev["fold"], ev["rung"], ev["train_nll"] = held, rung, loss
            rows.append(ev)
            params_rows.append({"fold": held, "rung": rung, "rule": rule.name, "train_nll": loss,
                                "T": float(rule.temperature(params)),
                                "params": " ".join(f"{float(x):.4f}" for x in params)})
    frame = pd.concat(rows, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(out / f"ladder_{args.split}.csv", index=False)
    pd.DataFrame(params_rows).to_csv(out / f"ladder_params_{args.split}.csv", index=False)
    lodo = frame[frame.fold != "all"]
    per_ds = lodo.groupby(["rung", "dataset"], as_index=False)[["acc", "ece", "nll", "alpha_mean"]].mean()
    summary = per_ds.groupby("rung", as_index=False)[["acc", "ece", "nll", "alpha_mean"]].mean()
    summary.round(4).to_csv(out / f"ladder_lodo_summary_{args.split}.csv", index=False)
    print("LODO (held-out dataset, mean of datasets):")
    print(summary.round(3).to_string(index=False))
    print(per_ds.pivot(index="dataset", columns="rung", values="nll").round(4).to_string())


# --- agreement gate (the T14 candidate) ----------------------------------------------------

def disagree_mask(run: Run) -> torch.Tensor:
    """Gate open: the cache is non-empty and its argmax differs from CLIP's (as online)."""
    has = run.cache.clamp_min(0).sum(-1) > 0
    return has & (run.s["agree"] == 0)


def agree_logits(run: Run, alpha_d=2.0, t_dis=1.0, t_agree=1.0, alpha_a=0.0) -> torch.Tensor:
    dis = disagree_mask(run)
    a = torch.where(dis, torch.as_tensor(alpha_d, dtype=torch.float64),
                    torch.as_tensor(alpha_a, dtype=torch.float64))
    t = torch.where(dis, torch.as_tensor(t_dis, dtype=torch.float64),
                    torch.as_tensor(t_agree, dtype=torch.float64))
    return (run.base + a[:, None] * run.cache) / t[:, None]


AGREE_VARIANTS = {
    # name: (free parameters, builder(params) -> kwargs for agree_logits)
    "agree": ((), lambda p: {}),
    "agree+T_dis": (("log_t_dis",), lambda p: {"t_dis": p[0].exp()}),
    "agree+T_dis+T_agree": (("log_t_dis", "log_t_agree"),
                            lambda p: {"t_dis": p[0].exp(), "t_agree": p[1].exp()}),
}


def fit_agree(variant, runs, steps=100):
    names, build = AGREE_VARIANTS[variant]
    if not names:
        return torch.zeros(0, dtype=torch.float64)
    params = torch.zeros(len(names), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([params], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def loss_fn():
        per_ds = {}
        for r in runs:
            lp = agree_logits(r, **build(params)).log_softmax(-1)
            per_ds.setdefault(r.dataset, []).append(-lp.gather(1, r.labels[:, None]).mean())
        return torch.stack([torch.stack(v).mean() for v in per_ds.values()]).mean()

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    return params.detach()


def cmd_agree(args):
    runs = _runs(args)
    datasets = list(dict.fromkeys(r.dataset for r in runs))
    rows, prm = [], []
    for r in runs:  # references
        for name, lg in (("clip", r.clip), ("alpha=0", r.base), ("tda", r.base + 2 * r.cache)):
            rows.append({"fold": "-", "dataset": r.dataset, "seed": r.seed, "rule": name,
                         **metrics(lg.softmax(-1), r.labels)})
    for held in datasets + ["all"]:
        train = [r for r in runs if r.dataset != held] if held != "all" else runs
        test = [r for r in runs if r.dataset == held] if held != "all" else runs
        for variant, (names, build) in AGREE_VARIANTS.items():
            params = fit_agree(variant, train)
            kw = build(params)
            prm.append({"fold": held, "rule": variant,
                        **{k: float(v) for k, v in kw.items()}})
            for r in test:
                rows.append({"fold": held, "dataset": r.dataset, "seed": r.seed, "rule": variant,
                             "gate_open": float(disagree_mask(r).double().mean()),
                             **metrics(agree_logits(r, **kw).softmax(-1), r.labels)})
        # TDA + one global temperature, the reference rung (amendment 2026-11-10)
        tda_t = fit_rule(Rule("tda+T", (), 0, lambda p, xs: torch.tensor([2.0], dtype=torch.float64),
                              lambda: torch.zeros(0), 0), train)
        T = float(tda_t[-1].exp())
        prm.append({"fold": held, "rule": "tda+T", "T": T})
        for r in test:
            rows.append({"fold": held, "dataset": r.dataset, "seed": r.seed, "rule": "tda+T",
                         **metrics(((r.base + 2 * r.cache) / T).softmax(-1), r.labels)})
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(out / f"agree_{args.split}.csv", index=False)
    pd.DataFrame(prm).round(4).to_csv(out / f"agree_params_{args.split}.csv", index=False)
    held_out = frame[frame.fold.isin(datasets + ["-"])]
    per_ds = held_out.groupby(["rule", "dataset"], as_index=False)[["acc", "ece", "nll"]].mean()
    mean = per_ds.groupby("rule", as_index=False)[["acc", "ece", "nll"]].mean()
    print("held-out (LODO; references unfitted), mean of datasets:")
    print(mean.round(3).to_string(index=False))
    print(per_ds.pivot(index="rule", columns="dataset", values="ece").round(2).to_string())
    print(pd.DataFrame(prm).round(3).to_string(index=False))


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["oracle", "binned", "ladder", "agree"])
    ap.add_argument("--trace-root", default="replay_results/t14")
    ap.add_argument("--features", default="features")
    ap.add_argument("--split", default="val")
    ap.add_argument("--name", default="tda_eq_clean")
    ap.add_argument("--datasets", nargs="+", default=["dtd", "flower102", "pets", "eurosat", "aircraft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--signals", nargs="+", default=["clip_conf", "clip_margin", "clip_ent", "js",
                                                     "cache_support_clip_top", "cache_conf"])
    args = ap.parse_args(argv)
    {"oracle": cmd_oracle, "binned": cmd_binned, "ladder": cmd_ladder, "agree": cmd_agree}[args.command](args)


if __name__ == "__main__":
    main()
