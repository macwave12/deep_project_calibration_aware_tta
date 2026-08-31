from types import SimpleNamespace

import pytest
import torch

from online_method.cal_tda import (
    CalTDA,
    CalTdaConfig,
    finalize_probs,
    pre_temperature_logits,
)
from online_method.tda import TDA

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="upstream compute_cache_logits calls .cuda() unconditionally",
)


class _FakeCLIP:
    """Minimal stand-in for the CoOp model: `eval` plus `forward_features`.

    Returns one preset image feature per call, ignoring the pixels, so two
    runners fed the same feature stream are exactly comparable.
    """

    def __init__(self, features, text_features, logit_scale=100.0):
        self.features = features
        self.text_features = text_features
        self.logit_scale = torch.tensor(logit_scale, device=features.device)
        self.index = 0

    def eval(self):
        return self

    def forward_features(self, images):
        feature = self.features[self.index : self.index + 1]
        self.index += 1
        return feature, self.text_features, self.logit_scale


def _feature_stream(n_samples=40, n_classes=5, dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, dim, generator=generator)
    text = torch.randn(n_classes, dim, generator=generator)
    features = features / features.norm(dim=-1, keepdim=True)
    text = text / text.norm(dim=-1, keepdim=True)
    return features.cuda(), text.cuda()


def test_config_round_trips_through_a_dict():
    config = CalTdaConfig(use_margin_admission=False, fusion_weight=0.25)
    data = config.as_dict()
    assert data["use_margin_admission"] is False
    assert data["fusion_weight"] == 0.25
    assert data["margin_threshold"] == 0.1  # default survives


def test_all_flags_off_reproduces_plain_additive_tda():
    config = CalTdaConfig(
        use_margin_admission=False, use_prob_fusion=False, use_loo_temperature=False,
        alpha=2.0,
    )
    clip_logits = torch.tensor([[2.0, 1.0, 0.0]])
    cache_logits = torch.tensor([[0.0, 3.0, 0.0]])

    probs = finalize_probs(clip_logits, cache_logits, config, temperature=1.0)
    expected = (clip_logits + config.alpha * cache_logits).softmax(dim=-1)
    assert torch.allclose(probs, expected, atol=1e-6)


def test_probabilistic_fusion_flag_changes_the_output():
    on = CalTdaConfig(use_prob_fusion=True, fusion_weight=0.5)
    off = CalTdaConfig(use_prob_fusion=False)
    clip_logits = torch.tensor([[2.0, 1.0, 0.0]])
    cache_logits = torch.tensor([[0.0, 3.0, 0.0]])

    assert not torch.allclose(
        finalize_probs(clip_logits, cache_logits, on, temperature=1.0),
        finalize_probs(clip_logits, cache_logits, off, temperature=1.0),
    )


def test_temperature_above_one_softens_the_final_distribution():
    config = CalTdaConfig(use_prob_fusion=False, use_loo_temperature=True)
    clip_logits = torch.tensor([[8.0, 0.0, 0.0]])
    cache_logits = torch.zeros(1, 3)

    hot = finalize_probs(clip_logits, cache_logits, config, temperature=1.0)
    cool = finalize_probs(clip_logits, cache_logits, config, temperature=4.0)
    assert float(cool.max()) < float(hot.max())


def test_finalize_always_returns_a_valid_distribution():
    config = CalTdaConfig()
    clip_logits = torch.randn(7, 5)
    cache_logits = torch.randn(7, 5)
    probs = finalize_probs(clip_logits, cache_logits, config, temperature=2.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(7), atol=1e-5)


@pytest.mark.parametrize("use_prob_fusion", [False, True])
def test_pre_temperature_logits_match_finalize(use_prob_fusion):
    # The runner collects `pre_temperature_logits` and hands them to the
    # temperature search, so they must be the exact scores `finalize_probs`
    # will later divide by that temperature. This test pins the two together.
    config = CalTdaConfig(use_prob_fusion=use_prob_fusion, use_loo_temperature=True)
    clip_logits = torch.randn(6, 4)
    cache_logits = torch.randn(6, 4)

    scores = pre_temperature_logits(clip_logits, cache_logits, config)
    for temperature in (1.0, 2.5, 7.0):
        assert torch.allclose(
            finalize_probs(clip_logits, cache_logits, config, temperature),
            (scores / temperature).softmax(dim=-1),
            atol=1e-6,
        )


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="use_margin_admision"):
        CalTdaConfig.from_dict({"use_margin_admision": False})


def _drive(runner, n_samples, args):
    dummy = torch.zeros(1, 3, 4, 4).cuda()
    outputs = []
    for _ in range(n_samples):
        runner.pre_adaptation()
        outputs.append(runner.adaptation_process(None, dummy, args)["output"])
    return outputs


@requires_cuda
def test_all_flags_off_reproduces_upstream_tda_on_a_stream():
    # The control test, at unit scale: same feature stream, same dataset
    # alpha/beta, every contribution disabled -> the same distribution upstream
    # TDA's logits imply, sample for sample.
    n_samples = 40
    features, text = _feature_stream(n_samples=n_samples)
    args = SimpleNamespace(test_sets="DTD")

    upstream = TDA(_FakeCLIP(features, text), 0)
    upstream.prepare_model_and_optimization(args)
    ours = CalTDA(
        _FakeCLIP(features, text),
        0,
        config=CalTdaConfig(
            use_margin_admission=False, use_prob_fusion=False, use_loo_temperature=False
        ),
    )
    ours.prepare_model_and_optimization(args)

    for upstream_logits, our_probs in zip(
        _drive(upstream, n_samples, args), _drive(ours, n_samples, args)
    ):
        reference = upstream_logits.float().softmax(dim=-1)
        our_probs = our_probs.float()
        # The decision must be identical, always.
        assert int(reference.argmax(dim=-1)) == int(our_probs.argmax(dim=-1))
        # The probabilities agree to upstream's own precision: TDA accumulates
        # the final logits in fp16 (`output = logits.clone()` under autocast,
        # then `+=`), whose ULP at CLIP's logit magnitudes is ~0.03, while we
        # upcast to fp32 before combining so the row sums stay inside the
        # scorer's 1e-4 tolerance. That rounding is the only difference left.
        assert torch.allclose(reference, our_probs, atol=5e-3)


@requires_cuda
def test_margin_admission_keeps_uncertain_samples_out_of_the_cache():
    n_samples = 40
    features, text = _feature_stream(n_samples=n_samples)
    args = SimpleNamespace(test_sets="DTD")

    def cache_size(config):
        runner = CalTDA(_FakeCLIP(features, text), 0, config=config)
        runner.prepare_model_and_optimization(args)
        _drive(runner, n_samples, args)
        return sum(len(items) for items in runner.pos_cache.values())

    permissive = cache_size(CalTdaConfig(use_margin_admission=False))
    strict = cache_size(
        CalTdaConfig(use_margin_admission=True, margin_threshold=1e6)
    )
    assert permissive > 0
    assert strict == 0


@requires_cuda
def test_every_returned_output_is_a_valid_distribution():
    # Asserts on the tensor the driver actually scores -- `return_dict['output']`
    # -- not on a side buffer, so this covers the real scoring path.
    n_samples = 40
    features, text = _feature_stream(n_samples=n_samples)
    args = SimpleNamespace(test_sets="DTD")

    runner = CalTDA(_FakeCLIP(features, text), 0, config=CalTdaConfig())
    runner.prepare_model_and_optimization(args)

    probs = torch.cat(_drive(runner, n_samples, args), dim=0).float().cpu()
    assert probs.shape == (n_samples, text.shape[0])
    assert torch.allclose(probs.sum(dim=-1), torch.ones(n_samples), atol=1e-5)
    assert (probs >= 0).all()


@requires_cuda
def test_temperature_summary_records_grid_saturation():
    n_samples = 40
    features, text = _feature_stream(n_samples=n_samples)
    args = SimpleNamespace(test_sets="DTD")

    runner = CalTDA(_FakeCLIP(features, text), 0, config=CalTdaConfig())
    runner.prepare_model_and_optimization(args)
    _drive(runner, n_samples, args)

    summary = runner.temperature_summary()
    assert summary['n_samples'] == n_samples
    # Only the first sample has no score history to estimate from.
    assert summary['n_estimated'] == n_samples - 1
    assert summary['min'] <= summary['mean'] <= summary['max']
    assert summary['grid_min'] == 1.0 and summary['grid_max'] == 20.0
    assert 0.0 <= summary['frac_at_grid_boundary'] <= 1.0
    assert summary['frac_at_grid_boundary'] == pytest.approx(
        summary['frac_at_grid_min'] + summary['frac_at_grid_max']
    )


@requires_cuda
def test_no_temperature_summary_when_the_contribution_is_disabled():
    features, text = _feature_stream(n_samples=5)
    args = SimpleNamespace(test_sets="DTD")
    runner = CalTDA(_FakeCLIP(features, text), 0,
                    config=CalTdaConfig(use_loo_temperature=False))
    runner.prepare_model_and_optimization(args)
    _drive(runner, 5, args)
    assert runner.temperature_summary() is None


@requires_cuda
def test_admission_summary_counts_offered_and_admitted_samples():
    n_samples = 40
    features, text = _feature_stream(n_samples=n_samples)
    args = SimpleNamespace(test_sets="DTD")

    runner = CalTDA(
        _FakeCLIP(features, text), 0,
        config=CalTdaConfig(use_margin_admission=True, margin_threshold=1e6),
    )
    runner.prepare_model_and_optimization(args)
    _drive(runner, n_samples, args)

    summary = runner.admission_summary()
    assert summary['n_offered'] == n_samples
    # threshold is unreachable, so nothing clears the margin gate.
    assert summary['n_admitted'] == 0
    assert summary['admission_rate'] == pytest.approx(0.0)


@requires_cuda
def test_admission_rate_is_one_when_margin_admission_is_off():
    # Upstream admits to the positive cache unconditionally; "off" must record
    # that faithfully rather than reading as "nothing happened".
    n_samples = 40
    features, text = _feature_stream(n_samples=n_samples)
    args = SimpleNamespace(test_sets="DTD")

    runner = CalTDA(
        _FakeCLIP(features, text), 0,
        config=CalTdaConfig(use_margin_admission=False),
    )
    runner.prepare_model_and_optimization(args)
    _drive(runner, n_samples, args)

    summary = runner.admission_summary()
    assert summary['n_offered'] == n_samples
    assert summary['n_admitted'] == n_samples
    assert summary['admission_rate'] == pytest.approx(1.0)


@requires_cuda
def test_no_admission_summary_before_any_sample_is_processed():
    features, text = _feature_stream(n_samples=1)
    runner = CalTDA(_FakeCLIP(features, text), 0, config=CalTdaConfig())
    assert runner.admission_summary() is None


@requires_cuda
def test_dataset_alpha_and_beta_override_the_method_config():
    args = SimpleNamespace(test_sets="DTD")
    runner = CalTDA(_FakeCLIP(*_feature_stream(n_samples=1)), 0,
                    config=CalTdaConfig(alpha=99.0, beta=99.0))
    runner.prepare_model_and_optimization(args)
    # DTD's TDA-tuned positive-cache values, identical to what `tda` uses.
    assert runner.config.alpha == 2.0
    assert runner.config.beta == 3.0
