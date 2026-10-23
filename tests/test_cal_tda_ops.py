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


# --- Fusion refinements: tie gate and cache sharpening ------------------------
from online_method.cal_tda import (  # noqa: E402
    CalTdaConfig, finalize_probs, is_two_way_tie, match_logit_range,
    pre_temperature_logits,
)


def test_two_way_tie_needs_two_close_leaders_holding_most_of_the_mass():
    # Row 0: a dead heat between two classes with nothing else in play -> tie.
    # Row 1: a clear winner -> not a tie, the cache has no job here.
    # Row 2: mass spread thinly over many classes -> top two are close but hold
    #        too little of the mass; CLIP is confused, not torn, so no gate.
    logits = torch.tensor([
        [5.0, 4.9, 0.0, 0.0, 0.0],
        [5.0, 1.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0],
    ])
    tie = is_two_way_tie(logits, max_gap=0.2, min_mass=0.6)
    assert tie.tolist() == [True, False, False]


def test_match_logit_range_gives_cache_the_same_spread_as_clip():
    clip_logits = torch.tensor([[30.0, 24.0, 20.0]])      # range 10
    cache_logits = torch.tensor([[2.0, 0.0, 0.5]])        # range 2
    scaled = match_logit_range(cache_logits, clip_logits)
    assert float(scaled.max() - scaled.min()) == pytest.approx(10.0)
    # Rescaling must not reorder the cache's preferences.
    assert scaled.argsort().tolist() == cache_logits.argsort().tolist()


def test_match_logit_range_leaves_a_flat_cache_alone():
    clip_logits = torch.tensor([[3.0, 1.0, 0.0]])
    flat = torch.zeros(1, 3)
    assert torch.equal(match_logit_range(flat, clip_logits), flat)


def test_gated_fusion_passes_clear_winners_through_as_plain_clip():
    config = CalTdaConfig(
        use_margin_admission=False, use_loo_temperature=False,
        use_prob_fusion=True, fusion_gate=True,
    )
    clip_logits = torch.tensor([[6.0, 1.0, 0.0]])          # decisive
    cache_logits = torch.tensor([[0.0, 3.0, 0.0]])         # cache disagrees loudly
    probs = finalize_probs(clip_logits, cache_logits, config)
    assert torch.allclose(probs, clip_logits.softmax(dim=-1), atol=1e-6)


def test_gated_fusion_lets_the_cache_break_a_tie():
    config = CalTdaConfig(
        use_margin_admission=False, use_loo_temperature=False,
        use_prob_fusion=True, fusion_gate=True,
    )
    clip_logits = torch.tensor([[5.0, 4.95, 0.0]])         # torn between 0 and 1
    cache_logits = torch.tensor([[0.0, 3.0, 0.0]])         # cache says 1
    probs = finalize_probs(clip_logits, cache_logits, config)
    assert int(probs.argmax(dim=-1)) == 1
    assert not torch.allclose(probs, clip_logits.softmax(dim=-1))


def test_sharpened_cache_breaks_a_tie_more_decisively():
    base = dict(use_margin_admission=False, use_loo_temperature=False, use_prob_fusion=True)
    plain = CalTdaConfig(**base, fusion_sharpen_cache=False)
    sharp = CalTdaConfig(**base, fusion_sharpen_cache=True)
    clip_logits = torch.tensor([[30.0, 29.9, 20.0, 20.0, 20.0]])
    cache_logits = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]])  # small, flat, favours 1
    p_plain = finalize_probs(clip_logits, cache_logits, plain)
    p_sharp = finalize_probs(clip_logits, cache_logits, sharp)
    assert int(p_sharp.argmax(dim=-1)) == 1
    assert float(p_sharp[0, 1]) > float(p_plain[0, 1])


@pytest.mark.parametrize("gate,sharpen", [(True, False), (False, True), (True, True)])
def test_refined_fusion_keeps_pre_temperature_scores_consistent(gate, sharpen):
    # Same contract as the runner test for the base method: the scores the
    # temperature search sees must be the ones finalize_probs rescales.
    config = CalTdaConfig(
        use_prob_fusion=True, use_loo_temperature=True,
        fusion_gate=gate, fusion_sharpen_cache=sharpen,
    )
    clip_logits = torch.randn(6, 4) * 3
    cache_logits = torch.rand(6, 4)
    scores = pre_temperature_logits(clip_logits, cache_logits, config)
    for temperature in (1.0, 2.5):
        assert torch.allclose(
            finalize_probs(clip_logits, cache_logits, config, temperature),
            (scores / temperature).softmax(dim=-1),
            atol=1e-6,
        )


def test_fusion_refinement_knobs_are_inert_when_fusion_is_off():
    config = CalTdaConfig(use_prob_fusion=False, fusion_gate=True, fusion_sharpen_cache=True)
    inert = config.inert_fields()
    for name in ("fusion_gate", "fusion_gate_max_gap", "fusion_gate_min_mass",
                 "fusion_sharpen_cache"):
        assert name in inert
    gate_off = CalTdaConfig(use_prob_fusion=True, fusion_gate=False)
    assert "fusion_gate_max_gap" in gate_off.inert_fields()
    assert "fusion_gate" not in gate_off.inert_fields()
