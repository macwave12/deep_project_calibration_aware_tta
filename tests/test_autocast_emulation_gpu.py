"""GPU only: how close is the CPU CUDA-autocast emulation to real CUDA autocast?

Run once on the GPU machine with `-s` to see the report:

    python -m pytest -s tests/test_autocast_emulation_gpu.py

Random CLIP-like features (D=512, 47 classes, 400 steps). Compares (a) the
fp16 logits `logit_scale * f @ t.t()` (max difference in fp16 ulps) and (b) a
whole TDA stream, real autocast on CUDA vs. the emulation on CPU (argmax
disagreements, max |output| difference).
"""

import json
import warnings
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from utils.feature_replay import FeatureReplayModel, build_trainer, cuda_autocast_emulation, run_stream

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning"),
]


def _features(n=400, n_classes=47, dim=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    protos = F.normalize(torch.randn(n_classes, dim, generator=g), dim=-1)
    labels = torch.randint(0, n_classes, (n,), generator=g)
    feats = F.normalize(protos[labels] + 1.5 * torch.randn(n, dim, generator=g) / dim ** 0.5, dim=-1)
    text = F.normalize(protos + 0.8 * torch.randn(n_classes, dim, generator=g) / dim ** 0.5, dim=-1)
    return feats, text, torch.tensor(100.0)


def _ulp_distance(a, b):
    """|a - b| in fp16 ulps (same-sign values; sign changes count as large)."""
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    to_ordered = lambda i: torch.where(i < 0, -(i & 0x7FFF), i)  # noqa: E731
    return (to_ordered(ia) - to_ordered(ib)).abs()


def test_emulated_logits_are_within_one_fp16_ulp_of_cuda():
    feats, text, scale = _features()
    with torch.cuda.amp.autocast():
        real = scale.cuda() * feats.cuda() @ text.cuda().t()
    with cuda_autocast_emulation():
        with torch.cuda.amp.autocast():
            emulated = scale * feats @ text.t()
    assert real.dtype == emulated.dtype == torch.float16
    ulps = _ulp_distance(real.cpu(), emulated)
    report = {"max_ulp": int(ulps.max()), "frac_nonzero_ulp": float((ulps > 0).float().mean())}
    print("\nlogits, CUDA autocast vs emulation:", json.dumps(report))
    assert report["max_ulp"] <= 1


def test_tda_stream_under_cuda_matches_the_emulation():
    feats, text, scale = _features()
    order = torch.randperm(len(feats), generator=torch.Generator().manual_seed(1))
    args = SimpleNamespace(test_sets="DTD")
    outputs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for where in ("cuda", "cpu"):
            model = FeatureReplayModel(feats.to(where), text.to(where), scale.to(where))
            trainer = build_trainer("tda", model, {})
            trainer.prepare_model_and_optimization(args)
            with cuda_autocast_emulation(where == "cpu"):
                outputs[where] = torch.cat([o.float().cpu() for o in run_stream(trainer, model, order, args)])
    real, emulated = outputs["cuda"], outputs["cpu"]
    disagree = int((real.argmax(-1) != emulated.argmax(-1)).sum())
    report = {"steps": len(order), "argmax_disagreements": disagree,
              "max_abs_output_diff": float((real - emulated).abs().max())}
    print("\nTDA stream, CUDA autocast vs emulation:", json.dumps(report))
    assert disagree <= max(1, len(order) // 100)
