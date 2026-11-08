"""The extractor reaches its stream draw in the same RNG state as the real runner.

Runs the real `online_tta.main()` and the real `scripts/extract_features.main()`
(CPU, `--allow-cpu --force`) with `get_coop` / `build_dataset` replaced by stubs
that draw from the global RNG like CLIP's `__init__` does. A spy on
`DataLoader.__iter__` records `torch.get_rng_state()` the moment a shuffled loader
starts iterating — the runner's `for ... in enumerate(val_loader)` and the
extractor's `stream_order` — and stops the runner there. Both must be identical,
and both must have called `get_coop` / `build_dataset` with the same arguments.
"""

import types
import warnings

import pytest
import torch

from tests.replay_stubs import FakeImageDataset, StubCLIP

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")


class _StopAtLoop(Exception):
    pass


def _transform_signature(transform):
    """Transforms compared by value; the RGB-convert function by name (the runner's
    is a closure inside `main`, the extractor's a module-level function)."""
    return tuple(t.__name__ if isinstance(t, types.FunctionType) else repr(t)
                 for t in transform.transforms)


@pytest.fixture
def stubs(monkeypatch):
    import clip.custom_clip as custom_clip
    import data.datautils as datautils
    import online_tta

    calls = {"get_coop": [], "build_dataset": []}

    def fake_get_coop(arch, classnames, device, n_ctx, ctx_init, learned_cls=False):
        calls["get_coop"].append((arch, tuple(classnames), n_ctx, ctx_init))
        torch.nn.Linear(300, 300)  # like CLIP's random init, before the checkpoint loads
        torch.randn(12345)
        return StubCLIP(n_classes=len(classnames))

    def fake_build_dataset(set_id, transform, root, mode="test", n_shot=None, **kw):
        calls["build_dataset"].append((set_id, mode, n_shot, _transform_signature(transform)))
        return FakeImageDataset(n=30, n_classes=47)

    for module in (custom_clip, online_tta):
        monkeypatch.setattr(module, "get_coop", fake_get_coop)
    for module in (datautils, online_tta):
        monkeypatch.setattr(module, "build_dataset", fake_build_dataset)

    states = {}
    current = {"tag": None}
    original_iter = torch.utils.data.DataLoader.__iter__

    def spy_iter(self):
        if isinstance(self.sampler, torch.utils.data.RandomSampler):
            states.setdefault(current["tag"], torch.get_rng_state().clone())
            if current["tag"] == "online_tta":
                raise _StopAtLoop()
        return original_iter(self)

    monkeypatch.setattr(torch.utils.data.DataLoader, "__iter__", spy_iter)
    return calls, states, current


@pytest.mark.parametrize("algorithm,config,seed", [
    ("clipzs", None, 0),
    ("tda", "configs/method/tda.yaml", 1),
    ("cal_tda", "configs/method/cal_tda_gate.yaml", 2),
])
def test_extractor_reaches_the_stream_draw_in_the_runners_rng_state(stubs, tmp_path, algorithm, config, seed):
    import online_tta
    from scripts import extract_features

    calls, states, current = stubs
    argv = ["--data", str(tmp_path), "--test_sets", "DTD", "-a", "ViT-B/16", "-b", "1", "-j", "0",
            "--gpu", "0", "--ctx_init", "a_photo_of_a", "--output_dir", str(tmp_path / "ckps"),
            "--algorithm", algorithm, "--seed", str(seed), "--out-dir", str(tmp_path / "raw")]
    if config:
        argv += ["--config", config]
    current["tag"] = "online_tta"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_StopAtLoop):
            online_tta.main(argv)
    runner_calls = {k: list(v) for k, v in calls.items()}
    for v in calls.values():
        v.clear()

    current["tag"] = "extractor"
    extract_features.main(["--data-root", str(tmp_path), "--out", str(tmp_path / "features"),
                           "--datasets", "dtd", "--splits", "test", "--seeds", str(seed),
                           "--train16-views", "0", "--recorded-runs", str(tmp_path / "none"),
                           "--allow-cpu", "--force"])

    assert torch.equal(states["online_tta"], states["extractor"])
    assert runner_calls["get_coop"][0] == calls["get_coop"][0]
    # The runner builds the test split once; the extractor builds it several times
    # (preflight, length, preamble, features), always with the runner's arguments
    # and an equal transform.
    assert len(runner_calls["build_dataset"]) == 1
    assert runner_calls["build_dataset"][0][:3] == ("DTD", "test", None)
    assert all(c == runner_calls["build_dataset"][0] for c in calls["build_dataset"])


def test_the_spy_would_notice_a_different_path(stubs, tmp_path):
    """Control: an extra RNG draw before the stream changes the recorded state."""
    from utils.feature_cache import stream_order
    from utils.tools import set_random_seed

    _, states, current = stubs
    for tag, extra in (("a", False), ("b", True)):
        current["tag"] = tag
        set_random_seed(0)
        if extra:
            torch.randn(1)
        stream_order(10)
    assert not torch.equal(states["a"], states["b"])
