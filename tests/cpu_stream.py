"""CPU harness for driving `CalTDA` / `TDA` over a synthetic feature stream.

Upstream `compute_cache_logits` calls `.cuda()` unconditionally, so on a
CPU-only machine `cpu_only` patches `Tensor.cuda` to the identity for the
duration of one test. Once the method code is device-agnostic the patch is a
harmless no-op. Every test that uses it compares two runs made under the same
patch, so the patch cannot hide a difference between them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cpu_only(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    yield


class FakeCLIP:
    """`eval` plus `forward_features`: one preset image feature per call."""

    def __init__(self, features, text_features, logit_scale=100.0):
        self.features = features
        self.text_features = text_features
        self.logit_scale = torch.tensor(logit_scale)
        self.index = 0

    def eval(self):
        return self

    def forward_features(self, images):
        feature = self.features[self.index : self.index + 1]
        self.index += 1
        return feature, self.text_features, self.logit_scale


def feature_stream(n_samples=200, n_classes=10, dim=32, seed=0, spread=0.6):
    """Clustered unit features: each sample sits near one class's text feature,
    with enough noise that CLIP is wrong on a fair share of the stream (so the
    caches carry mistakes and the temperature search has work to do)."""
    generator = torch.Generator().manual_seed(seed)
    text = torch.randn(n_classes, dim, generator=generator)
    text = text / text.norm(dim=-1, keepdim=True)
    classes = torch.randint(0, n_classes, (n_samples,), generator=generator)
    features = text[classes] + spread * torch.randn(n_samples, dim, generator=generator)
    features = features / features.norm(dim=-1, keepdim=True)
    return features, text, classes


def drive(runner, n_samples, args):
    dummy = torch.zeros(1, 3, 4, 4)
    outputs = []
    for _ in range(n_samples):
        runner.pre_adaptation()
        outputs.append(runner.adaptation_process(None, dummy, args)["output"])
    return outputs


def default_args(test_sets="DTD"):
    return SimpleNamespace(test_sets=test_sets)
