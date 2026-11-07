"""T01: export what the CPU replay needs, independent of how the model is built.

`scripts/extract_features.py` supplies the real pieces (`get_coop`, upstream
`build_dataset`, CUDA); `tests/test_feature_extraction.py` supplies a stub model
and a tiny fake dataset and runs this same code on CPU.

What "the same as the original runs" means here, piece by piece:

* Image features: `ClipTestTimeTuning.forward_features` returns
  `image_encoder(x.type(dtype))` divided by its norm, under
  `torch.cuda.amp.autocast()` and `no_grad`, one image at a time (`-b 1`).
  `image_features` below is that image half verbatim, so the text encoder does
  not run once per image. The first batch of every split is also pushed through
  the real `forward_features` and the difference is recorded in meta.json
  (expected 0.0). The returned dtype is kept as is.
* Stream order: see `utils.feature_cache.stream_order`. Each (split, seed)
  permutation is drawn after `set_random_seed(seed)` and a `preamble` that
  repeats every global-RNG consumer `online_tta.py` runs before its loop:
  building the dataset, then building the model (CLIP's `__init__` draws
  random initial weights before the checkpoint overwrites them).
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from utils.feature_cache import (
    FORMAT_VERSION,
    META_NAME,
    TEXT_NAME,
    feats_name,
    labels_name,
    perm_name,
    save_tensor,
    stream_order,
)
from utils.tools import set_random_seed


def image_features(model, images):
    """The image half of `ClipTestTimeTuning.forward_features`, verbatim."""
    feats = model.image_encoder(images.type(model.dtype))
    return feats / feats.norm(dim=-1, keepdim=True)


def _autocast(enabled: bool):
    # The methods call `torch.cuda.amp.autocast()`; use the same context.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.cuda.amp.autocast(enabled=enabled)


def extract_split(model, dataset, device, *, batch_size=1, workers=0, autocast=True):
    """Features and labels in dataset order, plus a check against `forward_features`."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.device(device).type == "cuda",
    )
    feats, labels = [], []
    check = None
    with torch.no_grad():
        for batch in loader:
            images, target = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            with _autocast(autocast):
                out = image_features(model, images)
                if check is None:
                    reference, _, _ = model.forward_features(images)
                    check = {
                        "dtype_equal": reference.dtype == out.dtype,
                        "max_abs_diff": float((reference.float() - out.float()).abs().max()),
                    }
            feats.append(out.detach().cpu())
            labels.append(torch.as_tensor(target).reshape(-1).cpu())
    return torch.cat(feats), torch.cat(labels).to(torch.int64), check


def extract_text(model, probe_images, device, *, autocast=True) -> dict:
    """Text side, in both forms the single-view methods read it."""
    with torch.no_grad():
        with _autocast(autocast):
            _, text, scale = model.forward_features(probe_images.to(device))
        noautocast = model.get_text_features()
    return {
        "text_features": text.detach().cpu(),
        "logit_scale": torch.as_tensor(scale).detach().cpu(),
        "text_features_noautocast": noautocast.detach().cpu(),
        "logit_scale_raw": torch.as_tensor(model.logit_scale).detach().cpu(),
    }


def stream_permutation(n: int, seed: int, preamble: Callable[[], object]):
    """(order, preamble's return value) for one seed, along the runner's RNG path."""
    set_random_seed(seed)
    kept = preamble()
    return stream_order(n), kept


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_dataset(
    out_root,
    dataset_key: str,
    *,
    build_split: Callable[[str], object],
    build_model: Callable[[], object],
    release_model: Callable[[object], None] = lambda model: None,
    device="cuda",
    test_seeds=(0, 1, 2),
    val_seeds=(0,),
    splits=("test", "val", "train16"),
    batch_size=1,
    workers=0,
    autocast=True,
    meta_extra: dict | None = None,
    overwrite=False,
    log=print,
) -> dict:
    """Write `<out_root>/<dataset_key>/` and return its meta. Skips a finished folder.

    `build_split(name)` returns the dataset for "test" / "val" / "train16";
    `build_model()` returns the model on `device`, built exactly as the runner
    builds it. Permutations are drawn for "test" (`test_seeds`) and "val"
    (`val_seeds`); "train16" has none (it is a support set, not a stream).
    """
    out_dir = Path(out_root) / dataset_key
    meta_path = out_dir / META_NAME
    if meta_path.exists() and not overwrite:
        log(f"[{dataset_key}] complete already ({meta_path}); skipping")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    perm_plan = [("test", s) for s in test_seeds if "test" in splits]
    perm_plan += [("val", s) for s in val_seeds if "val" in splits]

    # 1) Permutations. The first preamble's model is kept for feature extraction,
    #    so the model is built once per permutation and never once more.
    model = None
    sizes = {}
    for split, seed in perm_plan:
        def preamble(split=split):
            dataset = build_split(split)
            sizes[split] = len(dataset)
            return build_model()

        # The length is needed before the draw; reading it does not touch the RNG.
        n = sizes.get(split)
        if n is None:
            n = len(build_split(split))
        order, built = stream_permutation(n, seed, preamble)
        assert len(order) == sizes[split]
        save_tensor(out_dir / perm_name(split, seed), order)
        log(f"[{dataset_key}] stream order {split} seed {seed}: first {order[:5].tolist()}")
        if model is None:
            model = built
        else:
            release_model(built)
            del built
    if model is None:
        model = build_model()

    # 2) Features in dataset order, labels kept apart.
    if hasattr(model, "eval"):
        model.eval()
    meta_splits, checks, probe = {}, {}, None
    for split in splits:
        dataset = build_split(split)
        feats, labels, check = extract_split(
            model, dataset, device, batch_size=batch_size, workers=workers, autocast=autocast,
        )
        if probe is None:
            probe = dataset[0][0].unsqueeze(0)
        save_tensor(out_dir / feats_name(split), feats)
        torch.save(
            {"labels": labels, "files": list(getattr(dataset, "image_list", []) or [])},
            out_dir / labels_name(split),
        )
        checks[split] = check
        meta_splits[split] = {
            "n": int(feats.shape[0]),
            "dim": int(feats.shape[1]),
            "dtype": str(feats.dtype).replace("torch.", ""),
        }
        log(f"[{dataset_key}] {split}: {tuple(feats.shape)} {feats.dtype}; forward_features check {check}")

    # 3) Text side.
    text = extract_text(model, probe, device, autocast=autocast)
    torch.save(text, out_dir / TEXT_NAME)
    release_model(model)

    files = sorted(p.name for p in out_dir.iterdir() if p.suffix == ".pt")
    meta = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset_key,
        "splits": meta_splits,
        "n_classes": int(text["text_features"].shape[0]),
        "text_dtype": str(text["text_features"].dtype).replace("torch.", ""),
        "logit_scale": float(text["logit_scale"]),
        "permutations": [perm_name(s, k) for s, k in perm_plan],
        "forward_features_check": checks,
        "batch_size": batch_size,
        "autocast": autocast,
        "sha256": {name: _sha256(out_dir / name) for name in files},
        "created": _datetime.datetime.now().isoformat(timespec="seconds"),
        **(meta_extra or {}),
    }
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)  # last write: marks the folder complete
    return meta
