"""T03 end to end: CalTDA with the O(N) temperature search produces exactly the
outputs of the pre-T03 O(N^2) path (full history re-scanned every step)."""

import pytest
import torch
import yaml
from pathlib import Path

from online_method.cal_tda import CalTDA, CalTdaConfig
from tests.cpu_stream import FakeCLIP, cpu_only, default_args, drive, feature_stream  # noqa: F401
from utils.loo_temperature import search_temperature


class ReferenceCalTDA(CalTDA):
    """CalTDA with the temperature search as it was before T03: the score
    history is kept in full and `search_temperature` re-scans it every step."""

    def _record_scores(self, scores):
        # The parent call only keeps `_estimate_temperature`'s "anything
        # recorded yet?" check working; the search below never reads its sums.
        super()._record_scores(scores)
        self.__dict__.setdefault("_history", []).append(scores)

    def _search_temperature(self, estimate):
        return search_temperature(torch.cat(self._history, dim=0), estimate)


def _run(cls, config, features, text, args):
    runner = cls(FakeCLIP(features, text), None, config=config)
    runner.prepare_model_and_optimization(args)
    outputs = drive(runner, features.shape[0], args)
    return runner, outputs


def _method_config(name):
    return CalTdaConfig.from_dict(yaml.safe_load(Path(f"configs/method/{name}.yaml").read_text()))


@pytest.mark.parametrize("name", ["cal_tda_temp", "cal_tda", "cal_tda_gate_temp"])
def test_streaming_temperature_reproduces_the_quadratic_path(cpu_only, name):
    features, text, _ = feature_stream(n_samples=120, n_classes=10, seed=11)
    args = default_args("DTD")

    reference, ref_outputs = _run(ReferenceCalTDA, _method_config(name), features, text, args)
    ours, our_outputs = _run(CalTDA, _method_config(name), features, text, args)

    assert ours.temperatures == reference.temperatures
    for ref, out in zip(ref_outputs, our_outputs):
        assert torch.equal(ref, out)
    assert ours.temperature_summary() == reference.temperature_summary()
    # Not vacuous: the search ran and actually moved the temperature around.
    assert ours.temperature_summary()["n_estimated"] == 119
    assert len(set(ours.temperatures)) > 5
