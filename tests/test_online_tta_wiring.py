"""Coverage for the driver-side wiring in `online_tta.py`.

These two helpers are the seam between a method and the scorer. Nothing else in
the suite touches them, and the probability-vs-logit branch in particular is one
token away from silently corrupting every `cal_tda` ECE while leaving accuracy
(and therefore the console output) looking perfectly healthy.
"""

from types import SimpleNamespace

import pytest
import torch

from online_method.cal_tda import CalTdaConfig
from online_tta import (
    PROBABILITY_OUTPUT_ALGORITHMS,
    SCORED_ALGORITHMS,
    load_method_config,
    record_hyperparams,
    to_probabilities,
)


def test_cal_tda_output_is_not_softmaxed_a_second_time():
    probs = torch.tensor([[0.9, 0.05, 0.05]])

    passed_through = to_probabilities('cal_tda', probs)
    assert torch.allclose(passed_through, probs)
    # A second softmax would flatten 0.90 to ~0.49 and quietly halve the
    # confidence every calibration number is computed from.
    assert float(passed_through.max()) == pytest.approx(0.9)
    assert float(probs.softmax(dim=-1).max()) < 0.6


def test_logit_methods_are_softmaxed_into_distributions():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
    for algorithm in ('tda', 'clipzs'):
        out = to_probabilities(algorithm, logits)
        assert torch.allclose(out, logits.softmax(dim=-1))
        assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_only_the_three_studied_methods_are_scored():
    assert SCORED_ALGORITHMS == {'clipzs', 'tda', 'cal_tda'}
    assert PROBABILITY_OUTPUT_ALGORITHMS == {'cal_tda'}
    # The seven untouched upstream methods were never inspected for output
    # shape or scale, so they must not reach the scorer.
    for algorithm in ('dmn', 'dmn_weak', 'boostadapter', 'dpe', 'ecalp',
                      'onzeta', 'dynaprompt'):
        assert algorithm not in SCORED_ALGORITHMS


def test_config_loading_is_typed_for_cal_tda_and_plain_otherwise(tmp_path):
    config_file = tmp_path / "method.yaml"
    config_file.write_text(
        "use_margin_admission: false\nfusion_weight: 0.25\n", encoding="utf-8"
    )

    typed = load_method_config(
        SimpleNamespace(algorithm='cal_tda', config=str(config_file))
    )
    assert isinstance(typed, CalTdaConfig)
    assert typed.use_margin_admission is False
    assert typed.fusion_weight == 0.25
    assert typed.use_prob_fusion is True  # unset field keeps its default

    plain = load_method_config(
        SimpleNamespace(algorithm='tda', config=str(config_file))
    )
    assert plain == {"use_margin_admission": False, "fusion_weight": 0.25}

    assert load_method_config(SimpleNamespace(algorithm='cal_tda', config=None)) \
        == CalTdaConfig()


def test_a_typo_in_a_config_file_fails_the_run(tmp_path):
    config_file = tmp_path / "typo.yaml"
    config_file.write_text("use_prob_fussion: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="use_prob_fussion"):
        load_method_config(SimpleNamespace(algorithm='cal_tda', config=str(config_file)))


class _StubRunner:
    def __init__(self, summary=None, admission=None):
        self._summary = summary
        self._admission = admission

    def temperature_summary(self):
        return self._summary

    def admission_summary(self):
        return self._admission


def test_the_record_never_presents_an_inert_knob_as_a_live_one():
    args = SimpleNamespace(algorithm='cal_tda')
    summary = {'mean': 3.0, 'frac_at_grid_boundary': 0.0}

    fused = record_hyperparams(
        args, _StubRunner(summary), CalTdaConfig(use_prob_fusion=True)
    )
    # `alpha` scales the additive branch, which fusion never enters.
    assert 'alpha' not in fused
    assert 'alpha' in fused['inert']
    assert 'fusion_weight' in fused
    # The entropy branch is unreachable from CalTDA under any config.
    assert 'entropy_threshold' not in fused
    assert fused['temperature'] == summary

    additive = record_hyperparams(
        args, _StubRunner(summary),
        CalTdaConfig(use_prob_fusion=False, use_margin_admission=False),
    )
    assert 'alpha' in additive
    assert 'fusion_weight' not in additive
    assert 'margin_threshold' not in additive


def test_no_temperature_summary_when_the_flag_is_off():
    params = record_hyperparams(
        SimpleNamespace(algorithm='cal_tda'), _StubRunner(None),
        CalTdaConfig(use_loo_temperature=False),
    )
    assert 'temperature' not in params
    assert params['use_loo_temperature'] is False


def test_admission_summary_is_wired_into_the_record():
    admission = {'n_offered': 40, 'n_admitted': 40, 'admission_rate': 1.0}
    params = record_hyperparams(
        SimpleNamespace(algorithm='cal_tda'),
        _StubRunner(admission=admission),
        CalTdaConfig(use_margin_admission=False),
    )
    assert params['admission'] == admission


def test_no_admission_summary_when_the_runner_has_nothing_to_report():
    params = record_hyperparams(
        SimpleNamespace(algorithm='cal_tda'), _StubRunner(admission=None),
        CalTdaConfig(),
    )
    assert 'admission' not in params
