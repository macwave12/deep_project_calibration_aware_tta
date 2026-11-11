import torch

from analysis.adaptive_alpha import Run, signals
from analysis.story import (confidence_signals, derive, fig_cache_ablation, fig_reliability, fig_step1,
                            fig_step2, fig_step3, fig_step5, per_dataset, summary_table, worst_case_safe)


def _run(dataset="dtd", seed=0, n=400, c=6):
    g = torch.Generator().manual_seed(seed + 7 * len(dataset))
    labels = torch.randint(0, c, (n,), generator=g)
    clip = torch.randn(n, c, generator=g, dtype=torch.float64) * 3
    clip[torch.arange(n), labels] += 2.5
    cache = torch.rand(n, c, generator=g, dtype=torch.float64) * 1.5
    cache[torch.arange(n), clip.argmax(-1)] += 0.8  # an echo: the cache tends to agree with CLIP
    neg = 0.1 * torch.rand(n, c, generator=g, dtype=torch.float64)
    self_vote = torch.rand(n, generator=g) < 0.3
    r = Run(dataset, seed, clip - neg, cache, clip, labels, self_vote)
    signals(r)
    return r


def test_gate_decisions_equal_clip_plus_negative_cache_where_the_cache_agrees():
    d = derive(_run())
    assert d["agree"].any()
    assert torch.equal(d["gate_pred"][d["agree"]], d["base_pred"][d["agree"]])
    assert torch.equal(d["gate_pred"][~d["agree"]], d["tda_pred"][~d["agree"]])


def test_lemma_agreement_never_changes_the_decision_and_never_lowers_confidence():
    # Lemma 2: argmax c = argmax b = y_hat  =>  TDA keeps y_hat and is at least as confident.
    r = _run()
    d = derive(r)
    m = d["agree"] & (d["base_pred"] == d["pred0"])
    assert torch.equal(d["tda_pred"][m], d["base_pred"][m])
    assert (d["tda_conf"][m] >= d["base_conf"][m] - 1e-12).all()


def test_worst_case_no_flip_region_is_never_flipped():
    r = _run()
    safe = worst_case_safe(r)
    tda_pred = (r.base + 2.0 * r.cache).argmax(-1)
    # the synthetic cache stays within S = 3 per class, so the bound applies
    assert float(r.cache.max()) <= 3.0
    assert torch.equal(tda_pred[safe], r.clip.argmax(-1)[safe])


def test_summary_table_and_figures(tmp_path):
    data = per_dataset([_run(ds, s) for ds in ("dtd", "pets") for s in (0, 1)])
    table = summary_table(data)
    assert list(table["dataset"]) == ["dtd", "pets", "mean"]
    for col in ("gate_ece", "tda_acc", "agree_share", "tda_gap_agree", "p_agree_given_self"):
        assert col in table and table[col].notna().all()
    assert ((table["agree_share"] >= 0) & (table["agree_share"] <= 100)).all()
    fig_step1(data, tmp_path / "a.png", "test")
    fig_step2(data, tmp_path / "b.png", "test", bins=4)
    fig_step3(data, tmp_path / "c.png", "test")
    fig_cache_ablation(table, tmp_path / "d.png", "test")
    fig_reliability(data, tmp_path / "e.png", "test")
    runs = [_run(ds, s) for ds in ("dtd", "pets", "eurosat") for s in (0, 1)]
    sig = confidence_signals(runs, head=100)
    assert set(sig.columns) == {"share", "over06", "conf"} and len(sig) == 3
    fig_step5(runs, {"test": per_dataset(runs)}, tmp_path / "f.png")
    assert all((tmp_path / f).stat().st_size > 0 for f in ("a.png", "b.png", "c.png", "d.png", "e.png", "f.png"))
