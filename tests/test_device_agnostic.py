"""Equivalence proof for the device-agnostic edits to method code (RULES §7).

The only change to `online_method/{tda,dmn_weak,onzeta}.py` is where three
tensors are *placed*: the hard-coded `.cuda()` / integer GPU id became "the
device the input already lives on". On the GPU runner those are the same device
(`model.cuda(args.gpu)` puts every input on `cuda:<gpu>`), so GPU behaviour is
unchanged. On CPU the old code could not run at all.

To prove the arithmetic is untouched, `_old_compute_cache_logits` below is a
verbatim copy of the pre-edit function. With `Tensor.cuda` patched to the
identity it runs on CPU and must agree with the new one *bit for bit*.
"""

import math
import warnings
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from online_method.tda import compute_cache_logits, update_cache


def _old_compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Verbatim pre-edit body of `online_method.tda.compute_cache_logits`."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().float() #.half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().float() # .half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits


def _filled_caches(n=60, n_classes=7, dim=16, dtype=torch.float32, seed=0):
    """Positive/negative caches filled by upstream's own `update_cache`."""
    g = torch.Generator().manual_seed(seed)
    text = F.normalize(torch.randn(n_classes, dim, generator=g), dim=-1).to(dtype)
    pos, neg, queries = {}, {}, []
    for _ in range(n):
        f = F.normalize(torch.randn(1, dim, generator=g), dim=-1).to(dtype)
        logits = 100.0 * f @ text.t()
        p = logits.float().softmax(-1).to(dtype)
        ent = -(p * torch.log(p + 1e-5)).sum(1)
        pred = int(logits.argmax())
        update_cache(pos, pred, [f, ent], 3)
        ratio = float(ent) / math.log2(n_classes)
        if 0.2 < ratio < 0.5 or len(neg) == 0:
            update_cache(neg, pred, [f, ent, p], 2, True)
        queries.append(f)
    return pos, neg, text, queries


@pytest.fixture
def cuda_is_identity(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)


def test_new_cache_logits_equal_the_old_ones_bit_for_bit(cuda_is_identity):
    # fp32 only: with fp16 features, `exp(...)` (half) @ `cache_values` (float)
    # raises a dtype error in the old and new code alike. TDA calls this outside
    # autocast, so the GPU runs that produced results/ must have had fp32
    # features — `forward_features` normalises by an autocast-fp32 `norm`.
    pos, neg, text, queries = _filled_caches()
    assert len(pos) > 1 and len(neg) > 0
    for f in queries[:20]:
        for cache, alpha, beta, thresholds in (
            (pos, 2.0, 5.0, None),
            (neg, 0.117, 1.0, (0.03, 1.0)),
        ):
            new = compute_cache_logits(f, cache, alpha, beta, text.unsqueeze(0), thresholds)
            old = _old_compute_cache_logits(f, cache, alpha, beta, text.unsqueeze(0), thresholds)
            assert new.dtype == old.dtype
            assert torch.equal(new, old)


def test_cache_logits_follow_the_input_device():
    pos, neg, text, queries = _filled_caches()
    out = compute_cache_logits(queries[0], pos, 2.0, 5.0, text.unsqueeze(0))
    assert out.device == queries[0].device
    out = compute_cache_logits(queries[0], neg, 0.117, 1.0, text.unsqueeze(0), (0.03, 1.0))
    assert out.device == queries[0].device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_on_the_gpu_new_equals_old():
    pos, neg, text, queries = _filled_caches()
    to = lambda c: {k: [[t.cuda() for t in item] for item in v] for k, v in c.items()}
    pos, neg, text = to(pos), to(neg), text.cuda()
    for f in queries[:10]:
        f = f.cuda()
        assert torch.equal(
            compute_cache_logits(f, pos, 2.0, 5.0, text.unsqueeze(0)),
            _old_compute_cache_logits(f, pos, 2.0, 5.0, text.unsqueeze(0)),
        )


class _TextOnlyModel:
    """What `dmn_weak`/`onzeta` read in `prepare_model_and_optimization`."""

    def __init__(self, n_classes=5, dim=8):
        g = torch.Generator().manual_seed(1)
        self.text = F.normalize(torch.randn(n_classes, dim, generator=g), dim=-1)
        self.feature = F.normalize(torch.randn(1, dim, generator=g), dim=-1)
        self.logit_scale = torch.tensor(math.log(100.0))
        self.dtype = torch.float32

    def eval(self):
        return self

    def get_text_features(self):
        return self.text.clone()

    def forward_features(self, images):
        return self.feature.clone(), self.text.clone(), self.logit_scale.exp()


def test_dmn_weak_and_onzeta_now_run_on_cpu():
    from online_method.dmn_weak import DMN_WEAK
    from online_method.onzeta import OnZeta

    args = SimpleNamespace(test_sets="DTD", loader_len=3)
    dummy = torch.zeros(1, 3, 1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cls in (DMN_WEAK, OnZeta):
            # The GPU runner passes an int GPU id here; it must no longer decide placement.
            runner = cls(_TextOnlyModel(), 0)
            runner.prepare_model_and_optimization(args)
            out = runner.adaptation_process(None, dummy, args)["output"]
            assert out.device.type == "cpu"
            assert torch.isfinite(out).all()
