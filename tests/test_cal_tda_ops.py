import pytest
import torch

from online_method.cal_tda import margin_score, probabilistic_fusion, should_admit


def test_margin_is_the_gap_between_the_top_two_scores():
    logits = torch.tensor([[3.0, 1.0, 0.5], [2.0, 2.0, 0.0]])
    assert torch.allclose(margin_score(logits), torch.tensor([2.0, 0.0]))


def test_margin_accepts_a_single_unbatched_sample():
    assert margin_score(torch.tensor([4.0, 1.0, 0.0])) == pytest.approx(3.0)


def test_admission_requires_a_clear_winner():
    decisive = torch.tensor([5.0, 1.0, 0.0])
    ambiguous = torch.tensor([5.0, 4.9, 0.0])
    assert should_admit(decisive, threshold=1.0) is True
    assert should_admit(ambiguous, threshold=1.0) is False


def test_fusion_returns_a_probability_distribution():
    clip_logits = torch.tensor([[2.0, 1.0, 0.0]])
    cache_logits = torch.tensor([[0.0, 3.0, 0.0]])
    fused = probabilistic_fusion(clip_logits, cache_logits, weight=0.5)
    assert torch.allclose(fused.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert (fused >= 0).all()


def test_weight_zero_reduces_to_plain_clip():
    clip_logits = torch.tensor([[2.0, 1.0, 0.0]])
    cache_logits = torch.tensor([[0.0, 5.0, 0.0]])
    fused = probabilistic_fusion(clip_logits, cache_logits, weight=0.0)
    assert torch.allclose(fused, clip_logits.softmax(dim=-1), atol=1e-6)


def test_cache_can_flip_the_prediction():
    clip_logits = torch.tensor([[2.0, 1.9, 0.0]])
    cache_logits = torch.tensor([[0.0, 6.0, 0.0]])
    fused = probabilistic_fusion(clip_logits, cache_logits, weight=0.5)
    assert int(fused.argmax(dim=-1)) == 1


def test_fusion_never_inflates_confidence_beyond_its_inputs():
    # This is the whole point of the change: a convex combination cannot be more
    # confident than the most confident of its two inputs, whereas summing
    # logits can be.
    clip_logits = torch.tensor([[6.0, 0.0, 0.0]])
    cache_logits = torch.tensor([[7.0, 0.0, 0.0]])
    fused = probabilistic_fusion(clip_logits, cache_logits, weight=0.5)
    additive = (clip_logits + cache_logits).softmax(dim=-1)

    ceiling = max(
        float(clip_logits.softmax(dim=-1).max()),
        float(cache_logits.softmax(dim=-1).max()),
    )
    assert float(fused.max()) <= ceiling + 1e-6
    assert float(additive.max()) > ceiling
