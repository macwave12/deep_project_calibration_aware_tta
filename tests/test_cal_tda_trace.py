"""T04: the label-free trace is opt-in, read-only, and faithful.

Also the strongest equivalence check for T03: the current CalTDA (trace off)
is compared bit for bit with CalTDA exactly as it was at `BASELINE` (before
T03/T04), loaded from git, on every method config in the repo.
"""

import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from online_method.cal_tda import CalTDA, CalTdaConfig
from online_method.cal_tda_trace import STEP_FIELDS, TraceRecorder
from online_method.tda import TDA
from tests.cpu_stream import FakeCLIP, cpu_only, default_args, drive, feature_stream  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "77303a5"  # deck/refinement before T03/T04
METHOD_CONFIGS = sorted(
    str(p.relative_to(ROOT / "configs" / "method").with_suffix(""))
    for p in (ROOT / "configs" / "method").rglob("*.yaml")
)
N = 100
N_BASELINE = 60


def _config(name, **overrides):
    data = yaml.safe_load((ROOT / "configs" / "method" / f"{name}.yaml").read_text())
    return replace(CalTdaConfig.from_dict(data), **overrides)


@pytest.fixture(scope="module")
def baseline_module():
    try:
        source = subprocess.run(
            ["git", "show", f"{BASELINE}:online_method/cal_tda.py"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {BASELINE} unavailable")
    name = f"_cal_tda_baseline_{BASELINE}"
    module = types.ModuleType(name)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    exec(compile(source, f"<{BASELINE}:online_method/cal_tda.py>", "exec"), module.__dict__)
    yield module
    sys.modules.pop(name, None)


def _run(cls, config, features, text, args, recorder=None):
    runner = cls(FakeCLIP(features, text), None, config=config)
    runner.prepare_model_and_optimization(args)
    if recorder is not None:
        runner.trace = recorder
    return runner, drive(runner, features.shape[0], args)


@pytest.fixture(scope="module")
def stream():
    features, text, classes = feature_stream(n_samples=N, n_classes=10, seed=5)
    return features, text, classes


@pytest.mark.parametrize("name", METHOD_CONFIGS)
def test_trace_off_is_bit_identical_to_the_pre_change_code(cpu_only, baseline_module, stream, name):
    # Shorter stream: the baseline's temperature search is quadratic.
    features, text = stream[0][:N_BASELINE], stream[1]
    args = default_args("Pets")
    old_cfg = baseline_module.CalTdaConfig.from_dict(
        yaml.safe_load((ROOT / "configs" / "method" / f"{name}.yaml").read_text())
    )
    old, old_out = _run(baseline_module.CalTDA, old_cfg, features, text, args)
    new, new_out = _run(CalTDA, _config(name), features, text, args)
    assert new.trace is None
    assert new.temperatures == old.temperatures
    assert all(torch.equal(a, b) for a, b in zip(old_out, new_out))
    assert new.temperature_summary() == old.temperature_summary()
    assert new.admission_summary() == old.admission_summary()
    if new.config.use_loo_temperature:  # not vacuous: the search moved T
        assert len(set(new.temperatures)) > 1


@pytest.mark.parametrize("name", ["tda_equivalent", "cal_tda", "cal_tda_gate_temp", "ablation/margin_temp"])
def test_trace_on_does_not_change_outputs(cpu_only, stream, name):
    features, text, _ = stream
    args = default_args("DTD")
    off, off_out = _run(CalTDA, _config(name), features, text, args)
    on, on_out = _run(CalTDA, _config(name, trace=True), features, text, args)
    assert isinstance(on.trace, TraceRecorder)
    assert on.temperatures == off.temperatures
    assert all(torch.equal(a, b) for a, b in zip(off_out, on_out))
    assert on.config.as_record_dict() == {**off.config.as_record_dict(), "trace": True}


@pytest.mark.parametrize("name", ["tda_equivalent", "cal_tda", "cal_tda_gate_temp"])
def test_trace_rows_are_faithful(cpu_only, stream, name):
    features, text, _ = stream
    args = default_args("DTD")
    runner, outputs = _run(CalTDA, _config(name), features, text, args,
                           recorder=TraceRecorder(snapshot_every=30))
    trace = runner.trace_dict()
    steps = trace["steps"]
    assert trace["n_steps"] == N and trace["n_classes"] == 10 and trace["topk"] == 5
    assert set(steps) == set(STEP_FIELDS)
    for field, (dtype, shape) in STEP_FIELDS.items():
        assert steps[field].dtype == dtype
        assert steps[field].shape == ((N, 5) if shape else (N,))

    out = torch.cat(outputs).float()
    clip = (100.0 * features @ text.t())
    assert torch.equal(steps["out_argmax"].long(), out.argmax(-1))
    assert torch.equal(steps["out_topk_prob"], out.topk(5).values)
    assert torch.equal(steps["pred0"].long(), clip.argmax(-1))
    assert torch.equal(steps["clip_topk_index"][:, 0], steps["pred0"])
    assert torch.allclose(steps["clip_topk_logit"], clip.topk(5).values, atol=1e-4)
    assert steps["temperature"].tolist() == pytest.approx(runner.temperatures)
    # Self-vote presence needs admission; NaN â exactly where no estimate ran.
    assert not (steps["in_pos_cache"] & ~steps["admitted"]).any()
    assert not (steps["in_neg_cache"] & ~steps["neg_band"]).any()
    assert int(steps["admitted"].sum()) == runner.admission_summary()["n_admitted"]
    if runner.config.use_loo_temperature:
        assert int((~steps["a_hat"].isnan()).sum()) == runner.temperature_summary()["n_estimated"]
    else:
        assert steps["a_hat"].isnan().all()
    gate_on = runner.config.use_prob_fusion and runner.config.fusion_gate
    assert ((steps["gate_fired"] >= 0) == (gate_on & ~steps["fusion_fallback"])).all()

    # Snapshots every 30 steps plus the final state; each item's feature is the
    # stream row that inserted it, keyed by that step's pseudo-label.
    assert [s["step"] for s in trace["snapshots"]] == [29, 59, 89, 99]
    for snap in trace["snapshots"]:
        for which in ("pos", "neg"):
            cache = snap[which]
            if cache["class"].numel() == 0:
                continue
            ins = cache["inserted_step"]
            assert (ins >= 0).all() and (ins <= snap["step"]).all()
            assert torch.equal(cache["feature"], features[ins])
            assert torch.equal(cache["class"], steps["pred0"][ins].long())
    final = trace["snapshots"][-1]["pos"]
    assert final["class"].numel() == sum(len(v) for v in runner.pos_cache.values())


def test_all_flags_off_with_trace_matches_upstream_tda(cpu_only, stream):
    features, text, _ = stream
    args = SimpleNamespace(test_sets="DTD")
    upstream = TDA(FakeCLIP(features, text), None)
    upstream.prepare_model_and_optimization(args)
    ref = drive(upstream, N, args)
    ours, out = _run(CalTDA, _config("tda_equivalent", trace=True), features, text, args)
    for r, o in zip(ref, out):
        r = r.float().softmax(-1)
        assert int(r.argmax()) == int(o.argmax())
        assert torch.allclose(r, o, atol=5e-3)
    steps = ours.trace_dict()["steps"]
    # TDA admits every sample; every sample therefore votes for itself.
    assert steps["admitted"].all() and steps["gate_fired"].eq(-1).all()


def test_save_trace_round_trips(cpu_only, stream, tmp_path):
    features, text, _ = stream
    runner, _ = _run(CalTDA, _config("cal_tda_gate_temp", trace=True), features, text, default_args())
    path = runner.save_trace(tmp_path / "dtd_cal_tda.trace.pt")
    loaded = torch.load(path, weights_only=True)
    assert loaded["version"] == 1 and loaded["config"]["trace"] is True
    assert torch.equal(loaded["steps"]["out_argmax"], runner.trace_dict()["steps"]["out_argmax"])

    off, _ = _run(CalTDA, _config("cal_tda_gate_temp"), features, text, default_args())
    assert off.save_trace(tmp_path / "x.pt") is None and not (tmp_path / "x.pt").exists()


def test_trace_flag_defaults_off_and_stays_out_of_records():
    config = CalTdaConfig()
    assert config.trace is False
    assert "trace" not in config.as_record_dict()
    assert "trace" not in config.inert_fields()
    for name in METHOD_CONFIGS:
        assert _config(name).trace is False


def test_cli_trace_flag_turns_the_trace_on():
    from online_tta import load_method_config

    on = load_method_config(SimpleNamespace(algorithm="cal_tda", config=None, trace=True))
    off = load_method_config(SimpleNamespace(algorithm="cal_tda", config=None))
    assert on.trace is True and off.trace is False
