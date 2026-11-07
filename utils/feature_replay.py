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
applies CUDA's cast policy through a TorchFunctionMode: fp16-list ops take
fp16-rounded inputs, compute in fp32 and round the result to fp16 once (cuBLAS's
fp16 GEMM with fp32 compute type); fp32-list ops get fp32 inputs; the rest run
uncast. For the ops inside the clipzs/tda/cal_tda autocast blocks (mul, matmul,
softmax) that is CUDA's policy; cuBLAS's summation order can still move a result
by an fp16 ulp (measured by tests/test_autocast_emulation_gpu.py on a GPU). Ops
outside the known lists are counted and warned on. CPU's own
`torch.autocast('cpu', float16)` is not used because its softmax stays fp16.
"""

from __future__ import annotations

import contextlib
import warnings

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


# Ops CUDA autocast leaves alone (no cast list): they run in their inputs' dtype
# with ordinary type promotion, as here. Matched by `__name__`. A floating op in
# none of the three sets is counted as 'unlisted' and warned on once, because the
# emulation's claim to match CUDA is unchecked for it.
_NEUTRAL_NAMES = {
    'mul', '__mul__', '__rmul__', '__imul__', 'add', '__add__', '__radd__', '__iadd__',
    'sub', '__sub__', '__rsub__', '__isub__', 'div', '__truediv__', '__rtruediv__',
    'neg', '__neg__', 'clone', 't', '__getitem__', '__setitem__', 'reshape', 'view',
    'unsqueeze', 'squeeze', 'permute', 'transpose', 'contiguous', 'detach', 'to', 'type',
    'float', 'half', 'size', 'dim', 'numel', 'item', 'max', 'min', 'topk', 'argmax',
    'argmin', 'sort', 'cat', 'stack', 'where', 'zeros_like', 'ones_like', 'eq', '__eq__',
    'gt', '__gt__', 'lt', '__lt__', 'ge', '__ge__', 'le', '__le__', '__and__', '__or__',
    'clamp', 'clamp_min', 'abs', 'outer', '__len__', 'expand', 'flatten', '__format__',
    '__bool__', '__float__', '__index__', 'tolist', 'cpu', 'any', 'all', 'squeeze_',
    'add_', 'sub_', 'mul_', 'div_', '__rdiv__', '__itruediv__', 'ne', '__ne__',
}


def _round_through_fp16(value):
    """An fp16-rounded copy of a floating tensor, held in fp32 for the computation."""
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(torch.float16).to(torch.float32)
    if isinstance(value, (list, tuple)):
        return type(value)(_round_through_fp16(v) for v in value)
    return value


def _has_floating_tensor(values) -> bool:
    for v in values:
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            return True
        if isinstance(v, (list, tuple)) and _has_floating_tensor(v):
            return True
    return False


class EmulationReport(dict):
    """Ops seen inside the methods' autocast blocks, by how they were treated."""

    def note(self, kind: str, name: str) -> None:
        bucket = self.setdefault(kind, {})
        bucket[name] = bucket.get(name, 0) + 1

    def summary(self) -> dict:
        """{'fp16' | 'fp32' | 'neutral' | 'unlisted': {op name: count}}"""
        return {k: dict(sorted(v.items())) for k, v in sorted(self.items())}


class _CudaAutocastPolicy(TorchFunctionMode):
    """CUDA autocast's cast policy on CPU.

    fp16 list: inputs rounded to fp16, the op computed in fp32, the result rounded
    to fp16 once — cuBLAS's fp16 GEMM with fp32 compute type, and deterministic
    (CPU fp16 kernels may accumulate differently). fp32 list: inputs cast to fp32.
    Everything else runs uncast.
    """

    def __init__(self, report: EmulationReport, warned: set):
        super().__init__()
        self.report = report
        self.warned = warned

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, '__name__', repr(func))
        if func in _FP16_OPS:
            self.report.note('fp16', name)
            out = func(*_round_through_fp16(args),
                       **{k: _round_through_fp16(v) for k, v in kwargs.items()})
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                out = out.to(torch.float16)
            return out
        if func in _FP32_OPS:
            self.report.note('fp32', name)
            args = _cast(args, torch.float32)
            kwargs = {k: (_cast(v, torch.float32) if k != 'dtype' else v) for k, v in kwargs.items()}
            return func(*args, **kwargs)
        if name != '__get__' and _has_floating_tensor(list(args) + list(kwargs.values())):
            if name in _NEUTRAL_NAMES:
                self.report.note('neutral', name)
            else:
                self.report.note('unlisted', name)
                if name not in self.warned:
                    self.warned.add(name)
                    warnings.warn(
                        f"CUDA-autocast emulation: op {name!r} is in no known cast list and "
                        "ran uncast; CUDA autocast may treat it differently", RuntimeWarning)
        return func(*args, **kwargs)


@contextlib.contextmanager
def cuda_autocast_emulation(enabled: bool = True):
    """While active, the methods' `torch.cuda.amp.autocast()` applies CUDA's casts on CPU.

    Yields an `EmulationReport` of the ops that ran under the policy (empty when
    disabled).
    """
    report = EmulationReport()
    if not enabled:
        yield report
        return
    warned: set = set()

    @contextlib.contextmanager
    def _emulated_autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        if not enabled:
            yield
            return
        with _CudaAutocastPolicy(report, warned):
            yield

    original = torch.cuda.amp.autocast
    torch.cuda.amp.autocast = _emulated_autocast
    try:
        yield report
    finally:
        torch.cuda.amp.autocast = original
