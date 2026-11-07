"""Benchmark (and equivalence check) for T03: O(N) vs O(N^2) LOO temperature.

Runs CalTDA over one synthetic CPU feature stream twice -- once with the
pre-T03 search (`search_temperature` re-scanning the whole score history every
step), once with the streaming `RunningConfidence` -- and reports wall time and
whether the per-step temperatures and outputs are identical.

    python3 scripts/bench_temperature_search.py --n 8100 --classes 10 --config cal_tda_temp

CPU only; `Tensor.cuda` is patched to the identity when CUDA is absent, since
upstream `compute_cache_logits` still calls it unconditionally.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_method.cal_tda import CalTDA, CalTdaConfig  # noqa: E402
from tests.cpu_stream import FakeCLIP, default_args, feature_stream  # noqa: E402
from utils.loo_temperature import search_temperature  # noqa: E402


class QuadraticCalTDA(CalTDA):
    """The pre-T03 path: full history kept, re-scanned at every step."""

    def _record_scores(self, scores):
        super()._record_scores(scores)
        self.__dict__.setdefault("_history", []).append(scores)

    def _search_temperature(self, estimate):
        return search_temperature(torch.cat(self._history, dim=0), estimate)


def run(cls, config, features, text, args):
    runner = cls(FakeCLIP(features, text), None, config=config)
    runner.prepare_model_and_optimization(args)
    dummy = torch.zeros(1, 3, 4, 4)
    outputs = []
    start = time.perf_counter()
    for _ in range(features.shape[0]):
        outputs.append(runner.adaptation_process(None, dummy, args)["output"])
    return time.perf_counter() - start, runner, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8100)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--config", default="cal_tda_temp")
    parser.add_argument("--dataset", default="eurosat", help="TDA alpha/beta table key")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-quadratic", action="store_true")
    opts = parser.parse_args()

    warnings.filterwarnings("ignore")
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self

    data = yaml.safe_load((ROOT / "configs" / "method" / f"{opts.config}.yaml").read_text())
    features, text, _ = feature_stream(opts.n, opts.classes, opts.dim, opts.seed)
    args = default_args(opts.dataset)

    t_new, new, out_new = run(CalTDA, CalTdaConfig.from_dict(data), features, text, args)
    print(f"streaming : {t_new:8.2f} s  ({1e3 * t_new / opts.n:.3f} ms/sample)")
    print(f"temperature summary: {new.temperature_summary()}")
    if opts.skip_quadratic:
        return
    t_old, old, out_old = run(QuadraticCalTDA, CalTdaConfig.from_dict(data), features, text, args)
    print(f"quadratic : {t_old:8.2f} s  ({1e3 * t_old / opts.n:.3f} ms/sample)")
    print(f"speedup   : {t_old / t_new:.1f}x")
    same_t = old.temperatures == new.temperatures
    same_out = all(torch.equal(a, b) for a, b in zip(out_old, out_new))
    n_diff = sum(a != b for a, b in zip(old.temperatures, new.temperatures))
    print(f"identical temperatures: {same_t} ({n_diff} differing steps); identical outputs: {same_out}")


if __name__ == "__main__":
    main()
