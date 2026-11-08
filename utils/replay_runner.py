"""T02, scorer side: replay one (dataset, split, seed, method) and write its record.

Splits the work the way `online_tta.py` does: the method runs on the stream and
produces outputs (`utils.feature_replay`, which never sees a label); only then
are the labels loaded, put in stream order, and handed with the probabilities to
`utils.evaluation.score_and_save` — the same scorer and the same record format
(`<out_dir>/<dataset>_<name>.json`), with `hyperparams` built by the runner's own
`record_hyperparams`, plus a `replay` block saying where the features came from.

Records of `--autocast off` go to `<out_dir>/autocast_off/`, the val split to
`.../val/` and seeds other than 0 to `.../seed<k>/`, so they never overwrite the
default test / seed-0 record. Methods the GPU runner does not score (dmn_weak, ecalp, onzeta: their
outputs were never inspected, see `online_tta.SCORED_ALGORITHMS`) are run and
their raw outputs saved to `<out_dir>/<dataset>_<name>_outputs.pt`, unscored.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

from online_tta import (
    SCORED_ALGORITHMS,
    default_run_name,
    load_method_config,
    record_hyperparams,
    to_probabilities,
)
from utils.evaluation import score_and_save
from utils.feature_cache import UPSTREAM_SET_ID, FeatureStore
from utils.feature_replay import (
    FeatureReplayModel,
    build_trainer,
    cuda_autocast_emulation,
    run_stream,
)
from utils.tools import set_random_seed


def record_name(algorithm: str, run_name: str | None, protocol: str) -> str:
    """The GPU runner's naming (`online_tta.default_run_name`): `run_name` if given,
    else the algorithm, suffixed `_clean` under P2 for tda/cal_tda so a clean run
    never overwrites the published-protocol record. clipzs has no alpha/beta, so
    its P2 run *is* its P1 run and keeps the plain name."""
    return default_run_name(algorithm, run_name, protocol) or algorithm


def replay_one(features_root, dataset: str, *, algorithm: str, split: str = "test", seed: int = 0,
               config: str | None = None, protocol: str = "published", run_name: str | None = None,
               out_dir="replay_results/raw", autocast: str = "emulate", limit: int | None = None,
               log=print) -> Path:
    store = FeatureStore(features_root, dataset)
    meta = store.meta()
    order = store.permutation(split, seed)
    if limit is not None:
        order = order[:limit]
    model = FeatureReplayModel.from_text_dict(store.image_features(split), store.text())

    args = SimpleNamespace(
        algorithm=algorithm, config=config, test_sets=UPSTREAM_SET_ID[dataset],
        protocol=protocol, batch_size=1, gpu=None, loader_len=len(order), seed=seed,
        run_name=run_name, arch=meta.get("arch", "ViT-B/16"),
    )
    method_config = load_method_config(args)

    started = time.time()
    with warnings.catch_warnings():
        # Every step would otherwise warn that CUDA autocast / GradScaler is disabled.
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")
        set_random_seed(seed)  # as the runner; the replayed methods draw no random numbers
        trainer = build_trainer(algorithm, model, method_config)
        trainer.prepare_model_and_optimization(args)
        with torch.no_grad(), cuda_autocast_emulation(autocast == "emulate") as op_report:
            outputs = run_stream(trainer, model, order, args)
    seconds = time.time() - started

    name = record_name(algorithm, run_name, protocol)
    # Record file names carry no split or seed (`<dataset>_<name>.json`, as the
    # GPU runner), so val and seeds != 0 get sub-folders; test/seed 0 records sit
    # at the same relative paths as their counterparts under results/raw.
    out_dir = Path(out_dir)
    if autocast == "off":
        # Plain-fp32 replays are an A/B diagnostic, never a stand-in for the GPU record.
        out_dir = out_dir / "autocast_off"
    if split != "test":
        out_dir = out_dir / split
    if seed != 0:
        out_dir = out_dir / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if algorithm not in SCORED_ALGORITHMS:
        path = out_dir / f"{dataset}_{name}_outputs.pt"
        torch.save({"outputs": torch.cat([o.detach().float().cpu() for o in outputs]),
                    "stream_order": order, "seconds": seconds}, path)
        log(f"[{dataset}/{split}/seed{seed}] {name}: {len(order)} steps in {seconds:.1f}s "
            f"(unscored upstream method; raw outputs -> {path})")
        return path

    probs = torch.cat([to_probabilities(algorithm, o).cpu() for o in outputs], dim=0)
    # Labels enter here, after the method has finished, in stream order.
    labels = store.labels(split)[order]
    hyperparams = record_hyperparams(args, trainer, method_config)
    hyperparams["replay"] = {
        "split": split,
        "autocast": autocast,
        "autocast_ops": op_report.summary(),
        "n_steps": len(order),
        "full_stream": limit is None,
        "seconds": round(seconds, 2),
        "feature_meta": {k: meta.get(k) for k in ("format_version", "created", "git", "torch", "gpu")},
        "feature_sha256": meta.get("sha256", {}).get(f"{split}_feats.pt"),
    }
    path = score_and_save(
        probs.numpy(), labels.numpy(), dataset=dataset, method=name, backbone=args.arch,
        seed=seed, hyperparams=hyperparams, out_dir=out_dir,
    )
    log(f"[{dataset}/{split}/seed{seed}] {name}: {len(order)} steps in {seconds:.1f}s -> {path}")
    return path
