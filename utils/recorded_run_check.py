"""Exact stream-order check against a recorded GPU run (analysis / scorer side).

`online_tta.py` saves `save_results.pt` at the end of every run
(`<--output_dir>/bs1/<arch>/<test_sets>/save_results.pt`; `scripts/run_one.*`
use `--output_dir online_results/ckps/<run name>`). It holds the per-step
`output` and `target` concatenated **in stream order**. So, for the seed that
run used:

* `labels[perm] == save_dic['target']` elementwise says the extracted
  permutation reproduces the run's stream, and
* for clipzs, whose output is just `logit_scale * f @ t.t()` under autocast,
  recomputing that from the extracted features must give `save_dic['output']`
  within fp16 rounding: features, text side and order are all right.

This module reads labels; it is never imported by method code (RULES §2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

# fp16 logits of magnitude < 64 have an ulp <= 2**-5; two ulps covers a
# different cuBLAS summation order (batched vs. one row at a time).
FP16_LOGIT_ATOL = 2 * 2 ** -5
CHECKED_ALGORITHMS = ("clipzs", "tda")


def find_recorded_runs(ckpt_root, algorithm: str, test_set: str) -> list[Path]:
    """`<ckpt_root>/<algorithm>/**/<test_set>/save_results.pt`, newest first."""
    root = Path(ckpt_root) / algorithm
    if not root.is_dir():
        return []
    found = [p for p in root.rglob("save_results.pt") if p.parent.name == test_set]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def check_recorded_run(save_path, labels, permutations: dict,
                       clipzs_logits: Callable[[torch.Tensor], torch.Tensor] | None = None,
                       logits_must_match: bool = True) -> dict:
    """Compare one recorded run with the extracted labels and permutations.

    `labels`: ground truth in dataset order. `permutations`: {seed: perm}.
    `clipzs_logits(perm)`: logits recomputed from the cached features in that
    stream order (only for a clipzs run). `logits_must_match=False` (features
    extracted on CPU in fp32, not under CUDA autocast) still reports the logit
    comparison but bases the verdict on the labels alone. Returns a JSON-able
    dict whose `stream_order_matches` is the verdict.
    """
    saved = torch.load(save_path, map_location="cpu", weights_only=False)
    target = torch.as_tensor(saved["target"]).reshape(-1).to(torch.int64)
    labels = torch.as_tensor(labels).to(torch.int64)
    result = {"path": str(save_path), "n_recorded": int(target.numel()),
              "matching_seed": None, "stream_order_matches": False}
    per_seed = {}
    for seed, perm in sorted(permutations.items()):
        perm = torch.as_tensor(perm).to(torch.int64)
        if perm.numel() != target.numel():
            per_seed[seed] = {"labels_match": False, "reason": "length differs"}
            continue
        mismatch = (labels[perm] != target).nonzero()
        per_seed[seed] = {"labels_match": mismatch.numel() == 0,
                          "n_label_mismatches": int(mismatch.numel()),
                          "first_mismatch": int(mismatch[0]) if mismatch.numel() else None}
        if mismatch.numel() == 0 and result["matching_seed"] is None:
            result["matching_seed"] = int(seed)
    result["per_seed"] = {str(k): v for k, v in per_seed.items()}

    if result["matching_seed"] is None:
        return result
    if clipzs_logits is None:
        result["stream_order_matches"] = True
        return result

    perm = torch.as_tensor(permutations[result["matching_seed"]]).to(torch.int64)
    recomputed = clipzs_logits(perm).float().cpu()
    recorded = torch.as_tensor(saved["output"]).float().reshape(recomputed.shape)
    diff = (recomputed - recorded).abs()
    agree = (recomputed.argmax(-1) == recorded.argmax(-1)).float().mean()
    result["clipzs_output"] = {
        "max_abs_logit_diff": float(diff.max()),
        "atol": FP16_LOGIT_ATOL,
        "argmax_agreement": float(agree),
        "within_fp16_tolerance": bool(diff.max() <= FP16_LOGIT_ATOL),
    }
    result["clipzs_output"]["informational"] = not logits_must_match
    result["stream_order_matches"] = (result["clipzs_output"]["within_fp16_tolerance"]
                                      if logits_must_match else True)
    return result


def clipzs_logits_fn(image_features, text_features, logit_scale, device="cpu",
                     autocast: Callable[[], object] | None = None, chunk: int = 4096):
    """`ClipTestTimeTuning.inference`'s last line, over the cached features in stream order."""
    def run(perm):
        out = []
        text = text_features.to(device)
        scale = torch.as_tensor(logit_scale).to(device)
        with torch.no_grad():
            for start in range(0, perm.numel(), chunk):
                f = image_features[perm[start:start + chunk]].to(device)
                if autocast is None:
                    out.append(scale * f @ text.t())
                else:
                    with autocast():
                        out.append(scale * f @ text.t())
        return torch.cat(out).cpu()
    return run


def check_dataset(store, test_set: str, ckpt_root, seeds, *, device="cpu", autocast=None,
                  logits_must_match: bool = True, log=print) -> dict:
    """Run the check for every checked algorithm with a recorded run; summarise.

    Needs only the feature, text, label and permutation files (not meta.json),
    so the extractor can run it before it marks the folder complete.
    """
    labels = store.labels("test")
    perms = {s: store.permutation("test", s) for s in seeds}
    out = {}
    for algorithm in CHECKED_ALGORITHMS:
        runs = find_recorded_runs(ckpt_root, algorithm, test_set)
        if not runs:
            out[algorithm] = None
            continue
        fn = None
        if algorithm == "clipzs":
            text = store.text()
            fn = clipzs_logits_fn(store.image_features("test"), text["text_features"],
                                  text["logit_scale"], device=device, autocast=autocast)
        out[algorithm] = check_recorded_run(runs[0], labels, perms, clipzs_logits=fn,
                                            logits_must_match=logits_must_match)
    verdicts = [r["stream_order_matches"] for r in out.values() if r is not None]
    out["stream_order_matches_recorded_run"] = all(verdicts) if verdicts else None
    if verdicts:
        log(f"[{store.dataset}] stream order matches recorded run: {all(verdicts)} "
            + ", ".join(f"{a}: seed {r['matching_seed']}" for a, r in out.items()
                        if isinstance(r, dict) and "matching_seed" in r))
    else:
        log(f"[{store.dataset}] stream order matches recorded run: n/a "
            f"(no save_results.pt for {'/'.join(CHECKED_ALGORITHMS)} under {ckpt_root})")
    return out
