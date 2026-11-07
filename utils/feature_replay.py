"""T02, method side: run single-view methods on CPU from cached features.

The method code is used unchanged. `FeatureReplayModel` stands in for
`clip.custom_clip.ClipTestTimeTuning`; the runner seeks it to the dataset index
of the current stream position before every step, and the method's own
`forward_features(images)` / `model(images)` call then returns that sample's
cached feature. `images` is a placeholder and carries nothing.

Ground-truth discipline (RULES §2): nothing in this module reads, receives or
names a label; the scorer glue (`utils/replay_runner.py`) loads them after the
stream has finished. `tests/test_no_label_leak.py` covers this file.

CUDA autocast emulation. On the GPU the methods' `with torch.cuda.amp.autocast()`
blocks run matmuls in fp16 (TDA's `logits` are fp16, and its output is
accumulated in fp16) and softmax/norm/exp/... in fp32. On CPU that context only
warns and disables itself, so a plain replay computes those logits in fp32.
`cuda_autocast_emulation()` swaps `torch.cuda.amp.autocast` for a context that
applies CUDA's cast policy (fp16 list -> fp16 inputs, fp32 list -> fp32 inputs,
everything else untouched) through a TorchFunctionMode. For the ops inside the
clipzs/tda/cal_tda autocast blocks (mul, matmul, softmax) that is CUDA's policy
exactly; the fp16 rounding of a matmul may still differ in the last bit between
CPU and cuBLAS. CPU's own `torch.autocast('cpu', float16)` is not used because
its softmax stays fp16.
"""

from __future__ import annotations

import contextlib

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode

from online_method.cal_tda import CalTDA
from online_method.clipzs import CLIPZS
from online_method.dmn_weak import DMN_WEAK
from online_method.ecalp import ECALP
from online_method.onzeta import OnZeta
from online_method.tda import TDA

# Single-view methods (K20): they obtain features only through the model.
REPLAYABLE_ALGORITHMS = ('clipzs', 'tda', 'cal_tda', 'dmn_weak', 'ecalp', 'onzeta')


class FeatureReplayModel:
    """Serves cached CLIP outputs through `ClipTestTimeTuning`'s interface.

    Attributes the single-view methods read: `eval()`, `forward_features`,
    `__call__` (clipzs), `get_text_features()` (dmn_weak, onzeta, outside
    autocast), `logit_scale` (the *log* value; onzeta exps it), `dtype`.
    Every call returns fresh tensors, as the real model does.
    """

    def __init__(self, image_features, text_features, logit_scale, *,
                 text_features_noautocast=None, logit_scale_raw=None):
        if image_features.ndim != 2:
            raise ValueError("image_features must be (N, D)")
        self.image_features = image_features
        self.text_features = text_features
        self._logit_scale_exp = torch.as_tensor(logit_scale)
        self._text_noautocast = text_features if text_features_noautocast is None else text_features_noautocast
        self.logit_scale = (self._logit_scale_exp.log() if logit_scale_raw is None
                            else torch.as_tensor(logit_scale_raw))
        self._position = None

    @classmethod
    def from_text_dict(cls, image_features, text: dict) -> "FeatureReplayModel":
        return cls(
            image_features, text['text_features'], text['logit_scale'],
            text_features_noautocast=text.get('text_features_noautocast'),
            logit_scale_raw=text.get('logit_scale_raw'),
        )

    @property
    def dtype(self):
        return self.image_features.dtype

    def eval(self):
        return self

    def seek(self, index: int) -> None:
        self._position = int(index)

    def forward_features(self, images):
        if self._position is None:
            raise RuntimeError("seek() the replay model to a stream position first")
        i = self._position
        return (self.image_features[i:i + 1].clone(), self.text_features.clone(),
                self._logit_scale_exp.clone())

    def __call__(self, images):
        # `ClipTestTimeTuning.inference`: logit_scale * image_features @ text_features.t()
        image_features, text_features, logit_scale = self.forward_features(images)
        return logit_scale * image_features @ text_features.t()

    def get_text_features(self):
        return self._text_noautocast.clone()


def build_trainer(algorithm: str, model, method_config):
    """The method object, constructed as `online_tta.py` constructs it."""
    if algorithm == 'cal_tda':
        return CalTDA(model, None, config=method_config)
    classes = {'clipzs': CLIPZS, 'tda': TDA, 'dmn_weak': DMN_WEAK, 'ecalp': ECALP, 'onzeta': OnZeta}
    if algorithm not in classes:
        raise ValueError(f"{algorithm!r} is not a replayable single-view method; "
                         f"choose from {REPLAYABLE_ALGORITHMS}")
    return classes[algorithm](model, None)


def run_stream(trainer, model: FeatureReplayModel, order, args) -> list:
    """`online_tta.py`'s loop, minus the labels: one `output` per stream position."""
    placeholder = torch.zeros(1, 3, 1, 1, dtype=model.dtype)
    trainer.model.eval()
    outputs = []
    for index in torch.as_tensor(order).tolist():
        model.seek(index)
        trainer.pre_adaptation()
        outputs.append(trainer.adaptation_process(None, placeholder, args)['output'])
    return outputs


# --- CUDA autocast emulation ------------------------------------------------

_T = torch.Tensor
# CUDA autocast's fp16 ("lower precision") list, restricted to ops that exist as
# Python callables and could appear in these methods.
_FP16_OPS = {
    torch.matmul, _T.matmul, _T.__matmul__, _T.__rmatmul__, torch.mm, _T.mm,
    torch.bmm, _T.bmm, torch.mv, _T.mv, torch.addmm, _T.addmm, torch.addmv, _T.addmv,
    torch.addr, _T.addr, torch.baddbmm, _T.baddbmm, torch.addbmm, _T.addbmm,
    torch.chain_matmul, F.linear,
}
# CUDA autocast's fp32 list (same restriction).
_FP32_OPS = {
    torch.softmax, _T.softmax, F.softmax, torch.log_softmax, _T.log_softmax, F.log_softmax,
    torch.exp, _T.exp, torch.expm1, _T.expm1, torch.log, _T.log, torch.log2, _T.log2,
    torch.log10, _T.log10, torch.log1p, _T.log1p, torch.reciprocal, _T.reciprocal,
    torch.rsqrt, _T.rsqrt, torch.pow, _T.pow, _T.__pow__, torch.norm, _T.norm,
    torch.sum, _T.sum, torch.prod, _T.prod, torch.cumsum, _T.cumsum, torch.cumprod, _T.cumprod,
    torch.logsumexp, _T.logsumexp, torch.cdist, torch.dist, _T.dist, torch.renorm, _T.renorm,
    F.layer_norm, F.group_norm, F.softplus, F.cosine_similarity, F.normalize,
}


def _cast(value, dtype):
    if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != dtype \
            and value.dtype in (torch.float16, torch.float32, torch.float64):
        return value.to(dtype)
    if isinstance(value, (list, tuple)):
        return type(value)(_cast(v, dtype) for v in value)
    return value


class _CudaAutocastPolicy(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _FP16_OPS:
            args, kwargs = _cast(args, torch.float16), {k: _cast(v, torch.float16) for k, v in kwargs.items()}
        elif func in _FP32_OPS:
            args = _cast(args, torch.float32)
            kwargs = {k: (_cast(v, torch.float32) if k != 'dtype' else v) for k, v in kwargs.items()}
        return func(*args, **kwargs)


@contextlib.contextmanager
def _emulated_autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
    if not enabled:
        yield
        return
    with _CudaAutocastPolicy():
        yield


@contextlib.contextmanager
def cuda_autocast_emulation(enabled: bool = True):
    """While active, the methods' `torch.cuda.amp.autocast()` applies CUDA's casts on CPU."""
    if not enabled:
        yield
        return
    original = torch.cuda.amp.autocast
    torch.cuda.amp.autocast = _emulated_autocast
    try:
        yield
    finally:
        torch.cuda.amp.autocast = original
