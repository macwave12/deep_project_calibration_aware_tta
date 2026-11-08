"""Derived quantities the deck quotes that are not columns of the summary CSVs.

Everything here is a pure function of tracked run records (`results/raw/**.json`)
and dataset configs (`configs/<dataset>.yaml`); `main()` writes the results as
tracked CSVs under `results/analysis/`, and the deck reads those CSVs. Nothing
here touches a label beyond what the scorer already wrote into each record's
`reliability` field (per-bin counts, mean confidence, accuracy).

    python analysis/deck_evidence.py

Outputs
  results/analysis/reliability_stats.csv   one row per run record: accuracy,
      mean top-1 confidence, gap = confidence - accuracy, ECE (all in %)
  results/analysis/dataset_constants.csv   per dataset: TDA's alpha/beta as
      the runs used them, cache constants from the dataset config, and the
      closed-form bounds the TDA and fusion slides quote
  analysis/figures/deck_reliability_<dataset>.png   CLIP vs TDA reliability
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml

try:  # running as a script puts analysis/ itself on sys.path
    from aggregate import DATASET_LABEL, DATASET_ORDER
except ImportError:  # imported as analysis.deck_evidence (tests)
    from analysis.aggregate import DATASET_LABEL, DATASET_ORDER

# Study name -> directory under results/raw holding `{dataset}_{variant}.json`.
STUDIES = {
    "headline": "",
    "ablation": "ablation",
    "ablation_gate": "ablation_gate",
    "exploration": "exploration",
}

# P2 "clean" protocol (pm/PLAN.md, KNOWLEDGE K4): TDA's own single searched
# setting, supp. Table 6. `tests/test_deck_evidence.py` pins it to the TDA
# defaults recorded in configs/method/tda.yaml so the two cannot drift.
PROTOCOL_P2 = {"alpha": 2.0, "beta": 5.0}


# --- Reliability-derived statistics -------------------------------------------
def reliability_summary(record: dict) -> dict:
    """Accuracy, mean confidence, gap and ECE (in %) from a record's `reliability` bins.

    The bins hold, per confidence bin, the sample count, the mean top-1
    confidence and the accuracy of the samples in it. Count-weighted means of
    those recover the run's overall mean confidence and accuracy exactly; ECE
    is the count-weighted mean of |acc - conf| over bins.
    """
    rel = record["reliability"]
    counts = rel["bin_count"]
    total = sum(counts)
    if total == 0:
        raise ValueError("reliability field holds no samples")
    conf = sum(n * c for n, c in zip(counts, rel["bin_conf"])) / total
    acc = sum(n * a for n, a in zip(counts, rel["bin_acc"])) / total
    ece = sum(n * abs(a - c) for n, a, c in zip(counts, rel["bin_acc"], rel["bin_conf"])) / total
    return {
        "n_samples": total,
        "accuracy": 100.0 * acc,
        "mean_confidence": 100.0 * conf,
        "gap": 100.0 * (conf - acc),
        "ece": 100.0 * ece,
    }


def reliability_stats(raw_root) -> pd.DataFrame:
    """One row per run record across all studies, in deck dataset order."""
    raw_root = Path(raw_root)
    rows = []
    for study, sub in STUDIES.items():
        directory = raw_root / sub if sub else raw_root
        for path in sorted(directory.glob("*.json")):
            dataset, variant = path.stem.split("_", 1)
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["dataset"] != dataset:
                raise ValueError(f"{path}: file name says {dataset}, record says {record['dataset']}")
            stats = reliability_summary(record)
            if stats["n_samples"] != record["n_samples"]:
                raise ValueError(f"{path}: bins hold {stats['n_samples']} samples, record {record['n_samples']}")
            rows.append({"study": study, "dataset": dataset, "variant": variant,
                         "n_classes": record["n_classes"], "n_bins": record["n_bins"], **stats})
    frame = pd.DataFrame(rows)
    frame["_d"] = frame["dataset"].map(lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 99)
    frame = frame.sort_values(["study", "_d", "variant"]).drop(columns="_d").reset_index(drop=True)
    for column in ("accuracy", "mean_confidence", "gap", "ece"):
        frame[column] = frame[column].round(2)
    return frame


# --- Closed-form bounds used on the TDA and fusion slides --------------------
def vote_halving_cosine(beta: float) -> float:
    """Cosine at which a positive-cache vote exp(-beta (1 - cos)) falls to one half."""
    return 1.0 - math.log(2.0) / beta


def additive_overturn_log_odds(alpha: float, shot_capacity: int) -> float:
    """Largest CLIP log-odds log(p1/p2) TDA's positive cache can overturn.

    The final logit is z_k + alpha * c_k with c_k = sum of at most `shot_capacity`
    votes, each in (0, 1] (cosine <= 1). The cache moves the difference z_1 - z_2
    by at most alpha * shot_capacity, and z_1 - z_2 = log(p1 / p2).
    """
    return alpha * shot_capacity


def mixture_overturn_gap(n_classes: int, max_cache_logit: float) -> float:
    """Largest CLIP top-2 probability gap a 50/50 mixture with the cache can overturn.

    Fused p = (p_clip + q)/2 with q = softmax(c). Class 2 overtakes class 1 iff
    p1 - p2 < q2 - q1. Cache logits are >= 0 and <= `max_cache_logit`, so q2 - q1
    is largest with c_2 at the maximum and every other class at 0:
    (e^L - 1) / (e^L + C - 1).
    """
    e = math.exp(max_cache_logit)
    return (e - 1.0) / (e + n_classes - 1.0)


def flat_cache_top_probability(n_classes: int, cache_logit: float) -> float:
    """Top entry of softmax(c) when one class has logit `cache_logit` and the rest 0."""
    e = math.exp(cache_logit)
    return e / (e + n_classes - 1.0)


def dataset_constants(raw_root, configs_dir) -> pd.DataFrame:
    """Per dataset: TDA's alpha/beta as the runs used them plus cache constants.

    alpha and beta are read from the TDA run record (what was actually used);
    shot capacities, negative-cache constants and the entropy band from
    `configs/<dataset>.yaml`, which `tests/test_configs.py` pins to upstream.
    """
    raw_root, configs_dir = Path(raw_root), Path(configs_dir)
    rows = []
    for dataset in DATASET_ORDER:
        tda = json.loads((raw_root / f"{dataset}_tda.json").read_text(encoding="utf-8"))
        hp = tda["hyperparams"]
        cfg = yaml.safe_load((configs_dir / f"{dataset}.yaml").read_text(encoding="utf-8"))
        pos, neg = cfg["positive"], cfg["negative"]
        if (pos["alpha"], pos["beta"]) != (hp["alpha"], hp["beta"]):
            raise ValueError(f"{dataset}: config alpha/beta disagree with the TDA run record")
        capacity = int(hp["shot_capacity"])
        n_classes = int(tda["n_classes"])
        rows.append({
            "dataset": dataset,
            "label": DATASET_LABEL[dataset],
            "n_classes": n_classes,
            "n_samples": int(tda["n_samples"]),
            "alpha": float(hp["alpha"]),
            "beta": float(hp["beta"]),
            "p2_alpha": PROTOCOL_P2["alpha"],
            "p2_beta": PROTOCOL_P2["beta"],
            "pos_capacity": capacity,
            "neg_capacity": int(neg["shot_capacity"]),
            "neg_alpha": float(neg["alpha"]),
            "neg_beta": float(neg["beta"]),
            "band_lower": float(neg["entropy_threshold"]["lower"]),
            "band_upper": float(neg["entropy_threshold"]["upper"]),
            "mask_lower": float(neg["mask_threshold"]["lower"]),
            "vote_half_cos": round(vote_halving_cosine(float(hp["beta"])), 4),
            "overturn_log_odds": additive_overturn_log_odds(float(hp["alpha"]), capacity),
            "overturn_odds": round(math.exp(additive_overturn_log_odds(float(hp["alpha"]), capacity)), 1),
            "mixture_overturn_gap": round(mixture_overturn_gap(n_classes, capacity), 4),
            "flat_cache_top_p_at_cap": round(flat_cache_top_probability(n_classes, capacity), 4),
        })
    return pd.DataFrame(rows)


# --- Figures -----------------------------------------------------------------
PAIR_STYLE = {
    "clipzs": {"label": "CLIP zero-shot", "marker": "o", "color": "#4C72B0"},
    "tda": {"label": "TDA", "marker": "s", "color": "#DD8452"},
}


def plot_reliability_pair(raw_root, dataset: str, out_path, methods=("clipzs", "tda")) -> Path:
    """Reliability curves of a few headline methods on one dataset, sized for a slide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drawn at the size it is shown on the slide (five across), so 12pt here is 12pt there.
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, linewidth=1.2)
    for method in methods:
        style = PAIR_STYLE[method]
        record = json.loads((Path(raw_root) / f"{dataset}_{method}.json").read_text(encoding="utf-8"))
        rel = record["reliability"]
        points = [(c, a) for c, a, n in zip(rel["bin_conf"], rel["bin_acc"], rel["bin_count"]) if n > 0]
        ax.plot([p[0] for p in points], [p[1] for p in points], marker=style["marker"],
                markersize=4.5, color=style["color"], linewidth=2.0, label=style["label"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(DATASET_LABEL[dataset], fontsize=14, fontweight="bold")
    ax.set_xlabel("confidence", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def write_all(raw_root="results/raw", configs_dir="configs", out_dir="results/analysis",
              figures_dir="analysis/figures") -> dict[str, Path]:
    out_dir, figures_dir = Path(out_dir), Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    path = out_dir / "reliability_stats.csv"
    reliability_stats(raw_root).to_csv(path, index=False)
    written["reliability_stats"] = path
    path = out_dir / "dataset_constants.csv"
    dataset_constants(raw_root, configs_dir).to_csv(path, index=False)
    written["dataset_constants"] = path
    for dataset in DATASET_ORDER:
        written[f"reliability_{dataset}"] = plot_reliability_pair(
            raw_root, dataset, figures_dir / f"deck_reliability_{dataset}.png")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="results/raw")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--out-dir", default="results/analysis")
    parser.add_argument("--figures-dir", default="analysis/figures")
    args = parser.parse_args()
    for key, path in write_all(args.raw_root, args.configs_dir, args.out_dir, args.figures_dir).items():
        print(f"wrote {path}  ({key})")


if __name__ == "__main__":
    main()
