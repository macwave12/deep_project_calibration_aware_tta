import json

import torch

from analysis.comparison import entropy, record_rows, trace_logits, zaec
from tests.test_story import _run


def _logits(n=300, c=10, scale=4.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, generator=g, dtype=torch.float64) * scale


def test_zaec_restores_zero_shot_entropy_exactly_where_adaptation_lowered_it():
    zs, adapted = _logits(scale=2.0, seed=1), _logits(scale=6.0, seed=2)
    out = zaec(adapted, zs)
    lowered = entropy(adapted) < entropy(zs)
    assert lowered.any() and (~lowered).any()
    assert torch.allclose(entropy(out)[lowered], entropy(zs)[lowered], atol=1e-9)
    assert torch.equal(out[~lowered], adapted[~lowered])


def test_zaec_preserves_ranking_and_only_softens():
    zs, adapted = _logits(scale=2.0, seed=3), _logits(scale=8.0, seed=4)
    out = zaec(adapted, zs)
    assert torch.equal(out.argsort(-1), adapted.argsort(-1))
    t = adapted.norm(dim=-1) / out.norm(dim=-1)
    assert (t >= 1 - 1e-12).all()


def test_zaec_handles_very_sharp_adapted_logits():
    zs = _logits(scale=0.5, seed=5)
    adapted = _logits(scale=500.0, seed=6)
    out = zaec(adapted, zs)
    assert torch.allclose(entropy(out), entropy(zs), atol=1e-8)


def test_trace_logits_gate_is_tda_where_the_cache_disagrees_and_negative_only_elsewhere():
    from analysis.adaptive_alpha import disagree_mask
    r = _run()
    lg = trace_logits(r, temperature=1.5)
    dis = disagree_mask(r)
    assert dis.any() and (~dis).any()
    assert torch.allclose(lg["agreement gate"][dis], lg["TDA"][dis])
    assert torch.allclose(lg["agreement gate"][~dis], r.base[~dis])
    assert torch.allclose(lg["TDA + T (val)"], lg["TDA"] / 1.5)
    assert "read-first TDA" not in lg


def test_record_rows_reads_seed_folders_and_checks_the_seed(tmp_path):
    for seed, sub in ((0, tmp_path), (1, tmp_path / "seed1")):
        sub.mkdir(exist_ok=True)
        (sub / "dtd_ecalp.json").write_text(json.dumps(
            {"dataset": "dtd", "seed": seed, "accuracy": 0.5, "ece": 0.1}), encoding="utf-8")
    frame = record_rows(tmp_path, ["dtd", "pets"], [0, 1])
    assert list(frame["method"]) == ["ECALP", "ECALP"]
    assert frame["accuracy"].tolist() == [50.0, 50.0]
    (tmp_path / "seed1" / "dtd_ecalp.json").write_text(json.dumps(
        {"dataset": "dtd", "seed": 0, "accuracy": 0.5, "ece": 0.1}), encoding="utf-8")
    try:
        record_rows(tmp_path, ["dtd"], [0, 1])
    except ValueError:
        return
    raise AssertionError("a record with the wrong seed must be rejected")
