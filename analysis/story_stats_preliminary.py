"""PRELIMINARY (story proposal, 2026-11-11; untested): self-vote counterfactual, the no-flip region and
the CLIP-cache agreement rate on the test and dev sets, from `tda_eq_clean` full traces (P2, seeds 0/1/2).
The self-vote removal subtracts 1 (the self affinity exp(0)) at CLIP's argmax, which is approximate
under fp16 autocast emulation. Output: results/analysis/t14/story_stats_preliminary.csv
"""
import sys, torch, numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.adaptive_alpha import load_runs, metrics, disagree_mask
from utils.feature_cache import DATASETS, DEV_DATASETS
ALPHA = 2.0
rows = []
for group, dss in (("test", DATASETS), ("dev", DEV_DATASETS)):
    runs = load_runs("replay_results/t14", dss, [0, 1, 2], "test", "tda_eq_clean")
    for r in runs:
        y = r.labels
        z = r.clip; p = z.softmax(-1)
        pred0 = z.argmax(-1)
        has = r.cache.clamp_min(0).sum(-1) > 0
        agree = has & (r.cache.argmax(-1) == pred0)
        selfv = r.self_vote
        # counterfactual: remove the self term (affinity ~1 at pred0) from c
        c_noself = r.cache.clone()
        c_noself[selfv, pred0[selfv]] -= 1.0
        c_noself = c_noself.clamp_min(0)
        tda = (r.base + ALPHA * r.cache).softmax(-1)
        tda_ns = (r.base + ALPHA * c_noself).softmax(-1)
        m_tda, m_ns = metrics(tda, y), metrics(tda_ns, y)
        # no-flip region: CLIP top-1 gap in raw logits larger than the most the caches can move it
        top2 = z.topk(2, -1).values; gap = top2[:, 0] - top2[:, 1]
        n = r.clip - r.base
        bound = ALPHA * r.cache.max(-1).values + (n.max(-1).values - n.min(-1).values)
        safe = gap > bound
        flip = (r.base + ALPHA * r.cache).argmax(-1) != pred0
        conf_clip = p.max(-1).values
        corr_clip = (pred0 == y).double()
        conf_tda = tda.max(-1).values
        rows.append(dict(group=group, dataset=r.dataset, seed=r.seed,
            clip_acc=100*float(corr_clip.mean()), clip_conf=100*float(conf_clip.mean()),
            agree_rate=100*float(agree[has].double().mean()),
            self_rate=100*float(selfv.double().mean()),
            p_agree_self=100*float(agree[selfv].double().mean()) if selfv.any() else np.nan,
            p_agree_noself=100*float(agree[~selfv].double().mean()),
            tda_acc=m_tda["acc"], tda_ece=m_tda["ece"], noself_acc=m_ns["acc"], noself_ece=m_ns["ece"],
            safe_share=100*float(safe.double().mean()), safe_flips=int((safe & flip).sum()),
            safe_gap_clip=100*float((conf_clip-corr_clip)[safe].mean()) if safe.any() else np.nan,
            safe_gap_tda=100*float((conf_tda-corr_clip)[safe].mean()) if safe.any() else np.nan,
            agree_gap_clip=100*float((conf_clip - corr_clip)[agree].mean()),
            agree_acc_clip=100*float(corr_clip[agree].mean()), agree_conf_clip=100*float(conf_clip[agree].mean())))
df = pd.DataFrame(rows).groupby(["group","dataset"], as_index=False).mean(numeric_only=True).drop(columns="seed")
pd.set_option("display.width", 250)
print(df.round(2).to_string(index=False))
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "story_stats.csv", index=False)
