"""T01: export what the CPU replay needs, independent of how the model is built.

`scripts/extract_features.py` supplies the real pieces (`get_coop`, upstream
`build_dataset`, CUDA); the tests supply a stub model and tiny fake datasets and
run this same code on CPU.

What "the same as the original runs" means here, piece by piece:

* Image features: `ClipTestTimeTuning.forward_features` returns
  `image_encoder(x.type(dtype))` divided by its norm, under
  `torch.cuda.amp.autocast()` and `no_grad`, one image at a time (`-b 1`).
  `image_features` below is that image half, copied so the text encoder does not
  run once per image. Because it is a copy, the first batch of every split is
  also pushed through the loaded model's *own* `forward_features` and the two
  are compared (`image_half_matches_forward_features` in meta.json; expected
  identical). That would catch an upstream change to `forward_features` the copy
  does not follow. The returned dtype is kept as is.
* Stream order: see `utils.feature_cache.stream_order`. Each (split, seed)
  permutation is drawn after `set_random_seed(seed)` and a `preamble` that
  repeats every global-RNG consumer `online_tta.py` runs before its loop:
  building the dataset, then building the model (CLIP's `__init__` draws random
  initial weights before the checkpoint overwrites them).
* Augmented train16 views (Tip-Adapter's cache): `views` passes over the 16-shot
  set in dataset order with a random train transform, after
  `set_random_seed(aug_seed)`, saved as raw `image_encoder` outputs
  (views, N, D) — Tip-Adapter averages the views *before* normalising.

Resume: a folder whose meta.json exists is compared with the request. Missing
splits, permutations or views are added (extend); a different identity (arch,
prompt, resolution, batch size, autocast, torch/CUDA/GPU, format) redoes the
folder; an identical request is skipped.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import warnings
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from utils.feature_cache import (
    FORMAT_VERSION,
    META_NAME,
    TEXT_NAME,
    TRAIN16_AUG_NAME,
    FeatureStore,
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
    # The methods call `torch.cuda.amp.autocast()`; use the same context,
    # looked up at call time (the CPU tests swap in an emulation).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.cuda.amp.autocast(enabled=enabled)


def _loader(dataset, device, batch_size, workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.device(device).type == "cuda")


def extract_split(model, dataset, device, *, batch_size=1, workers=0, autocast=True):
    """Features and labels in dataset order, plus the `forward_features` consistency check."""
    feats, labels = [], []
    check = None
    with torch.no_grad():
        for batch in _loader(dataset, device, batch_size, workers):
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


def extract_augmented_views(model, dataset, device, *, views, seed, batch_size=1, workers=0,
                            autocast=True):
    """(views, N, D) raw `image_encoder` outputs over `views` passes in dataset order."""
    set_random_seed(seed)
    passes, labels = [], None
    with torch.no_grad():
        for _ in range(views):
            feats, pass_labels = [], []
            for batch in _loader(dataset, device, batch_size, workers):
                images = batch[0].to(device, non_blocking=True)
                with _autocast(autocast):
                    feats.append(model.image_encoder(images.type(model.dtype)).detach().cpu())
                pass_labels.append(torch.as_tensor(batch[1]).reshape(-1))
            passes.append(torch.cat(feats))
            pass_labels = torch.cat(pass_labels).to(torch.int64)
            assert labels is None or torch.equal(labels, pass_labels), "dataset order changed between views"
            labels = pass_labels
    return torch.stack(passes), labels


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


def preflight(dataset_key: str, splits, build_split: Callable[[str], object],
              missing_files: Callable[[object], list] | None = None) -> tuple[list, dict]:
    """Which splits can be extracted, before any model is built.

    Builds each split's dataset (reads its list / JSON file only) and, with
    `missing_files(dataset)`, checks the images exist. Returns (available splits,
    {skipped split: reason}).
    """
    available, skipped = [], {}
    for split in splits:
        try:
            dataset = build_split(split)
            n = len(dataset)
        except Exception as error:  # noqa: BLE001 - any failure here means "cannot extract"
            skipped[split] = f"{type(error).__name__}: {error}"
            continue
        if n == 0:
            skipped[split] = "empty split"
            continue
        missing = missing_files(dataset) if missing_files else []
        if missing:
            skipped[split] = f"{len(missing)} of {n} image files missing (first: {missing[0]})"
            continue
        available.append(split)
    return available, skipped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan(out_dir: Path, meta: dict | None, identity: dict, splits, perm_plan, aug: dict | None,
          overwrite: bool, log, key):
    """What to (re)compute: (mode, todo splits, todo perms, todo aug, need text)."""
    if meta is not None and not overwrite:
        differs = sorted(k for k in set(identity) | set(meta.get("identity", {}))
                         if identity.get(k) != meta.get("identity", {}).get(k))
        if differs:
            log(f"[{key}] existing features differ in {differs}; redoing the folder")
            meta = None
    if meta is None or overwrite:
        return "fresh", list(splits), list(perm_plan), aug, True
    done_splits = {s for s in meta.get("splits", {}) if (out_dir / feats_name(s)).exists()}
    done_perms = {p for p in meta.get("permutations", []) if (out_dir / p).exists()}
    todo_splits = [s for s in splits if s not in done_splits]
    todo_perms = [(s, k) for s, k in perm_plan if perm_name(s, k) not in done_perms]
    todo_aug = aug if aug and (meta.get("train16_aug") != aug
                               or not (out_dir / TRAIN16_AUG_NAME).exists()) else None
    need_text = not (out_dir / TEXT_NAME).exists()
    if not (todo_splits or todo_perms or todo_aug or need_text):
        return "skip", [], [], None, False
    return "extend", todo_splits, todo_perms, todo_aug, need_text


def extract_dataset(
    out_root,
    dataset_key: str,
    *,
    build_split: Callable[[str], object],
    build_model: Callable[[], object],
    free_memory: Callable[[], None] = lambda: None,
    device="cuda",
    test_seeds=(0, 1, 2),
    val_seeds=(0, 1, 2),
    splits=("test", "val", "train16"),
    train16_views: int = 0,
    aug_seed: int = 1,
    build_aug_split: Callable[[], object] | None = None,
    batch_size=1,
    workers=0,
    autocast=True,
    identity: dict | None = None,
    skipped_splits: dict | None = None,
    post_check: Callable[[Path, list], dict] | None = None,
    meta_extra: dict | None = None,
    overwrite=False,
    log=print,
) -> dict:
    """Write or update `<out_root>/<dataset_key>/` and return its meta.

    `build_split(name)` returns the dataset for "test" / "val" / "train16";
    `build_model()` the model on `device`, built exactly as the runner builds it;
    `build_aug_split()` train16 with the augmenting transform. Permutations are
    drawn for "test" (`test_seeds`) and "val" (`val_seeds`). `post_check(out_dir,
    test_seeds)` (the recorded-run order check) runs once the files are written.
    """
    out_dir = Path(out_root) / dataset_key
    meta_path = out_dir / META_NAME
    old_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    identity = {"format_version": FORMAT_VERSION, "batch_size": batch_size, "autocast": autocast,
                **(identity or {})}
    splits = list(splits)
    perm_plan = [("test", s) for s in test_seeds if "test" in splits]
    perm_plan += [("val", s) for s in val_seeds if "val" in splits]
    aug = None
    if train16_views and "train16" in splits and build_aug_split is not None:
        aug = {"views": int(train16_views), "seed": int(aug_seed), "workers": int(workers)}

    mode, todo_splits, todo_perms, todo_aug, need_text = _plan(
        out_dir, old_meta, identity, splits, perm_plan, aug, overwrite, log, dataset_key)
    if mode == "skip":
        log(f"[{dataset_key}] complete for this request ({meta_path}); skipping")
        return old_meta
    if mode == "extend":
        log(f"[{dataset_key}] extending: splits {todo_splits}, permutations "
            f"{[perm_name(s, k) for s, k in todo_perms]}, augmented views {bool(todo_aug)}")
    base_meta = old_meta if mode == "extend" else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "fresh":
        # A redo must not leave files of the old extraction behind (meta.json
        # checksums every .pt in the folder).
        if meta_path.exists():
            meta_path.unlink()
        for stale in out_dir.glob("*.pt"):
            stale.unlink()

    # 1) Permutations. The first preamble's model is kept for feature extraction,
    #    so the model is built once per permutation and not once more.
    model = None
    for split, seed in todo_perms:
        n = len(build_split(split))  # reading the length draws no random numbers

        def preamble(split=split):
            build_split(split)
            return build_model()

        order, built = stream_permutation(n, seed, preamble)
        save_tensor(out_dir / perm_name(split, seed), order)
        log(f"[{dataset_key}] stream order {split} seed {seed}: first {order[:5].tolist()}")
        if model is None:
            model = built
        else:
            del built
            free_memory()
    if model is None and (todo_splits or todo_aug or need_text):
        model = build_model()
    if model is not None and hasattr(model, "eval"):
        model.eval()

    # 2) Features in dataset order, labels kept apart.
    meta_splits = dict(base_meta.get("splits", {}))
    checks = dict(base_meta.get("image_half_matches_forward_features", {}))
    for split in todo_splits:
        dataset = build_split(split)
        feats, labels, check = extract_split(
            model, dataset, device, batch_size=batch_size, workers=workers, autocast=autocast)
        save_tensor(out_dir / feats_name(split), feats)
        torch.save({"labels": labels, "files": list(getattr(dataset, "image_list", []) or [])},
                   out_dir / labels_name(split))
        checks[split] = check
        meta_splits[split] = {"n": int(feats.shape[0]), "dim": int(feats.shape[1]),
                              "dtype": str(feats.dtype).replace("torch.", "")}
        log(f"[{dataset_key}] {split}: {tuple(feats.shape)} {feats.dtype}; "
            f"image half vs forward_features: {check}")

    # 3) Augmented train16 views (Tip-Adapter).
    aug_meta = base_meta.get("train16_aug")
    if todo_aug:
        views, aug_labels = extract_augmented_views(
            model, build_aug_split(), device, views=todo_aug["views"], seed=todo_aug["seed"],
            batch_size=batch_size, workers=workers, autocast=autocast)
        save_tensor(out_dir / TRAIN16_AUG_NAME, views)
        same = (out_dir / labels_name("train16")).exists() and torch.equal(
            FeatureStore(out_root, dataset_key).labels("train16"), aug_labels)
        aug_meta = {**todo_aug, "shape": list(views.shape),
                    "dtype": str(views.dtype).replace("torch.", ""), "normalized": False,
                    "labels_equal_train16": bool(same)}
        log(f"[{dataset_key}] train16 augmented: {tuple(views.shape)} {views.dtype}")

    # 4) Text side.
    if need_text:
        probe_split = next(s for s in splits if (out_dir / feats_name(s)).exists())
        probe = build_split(probe_split)[0][0].unsqueeze(0)
        torch.save(extract_text(model, probe, device, autocast=autocast), out_dir / TEXT_NAME)
    text = torch.load(out_dir / TEXT_NAME, map_location="cpu", weights_only=True)
    if model is not None:
        del model
        free_memory()

    # 5) Exact order check against a recorded GPU run, when one is available.
    recorded = base_meta.get("recorded_run_check")
    test_touched = any(s == "test" for s, _ in todo_perms) or "test" in todo_splits
    if post_check is not None and "test" in splits and (test_touched or recorded is None):
        recorded = post_check(out_dir, [k for s, k in perm_plan if s == "test"])

    permutations = sorted(set(base_meta.get("permutations", []))
                          | {perm_name(s, k) for s, k in perm_plan})
    files = sorted(p.name for p in out_dir.iterdir() if p.suffix == ".pt")
    history = list(base_meta.get("history", []))
    history.append({"at": _datetime.datetime.now().isoformat(timespec="seconds"), "mode": mode,
                    "splits": todo_splits, "permutations": [perm_name(s, k) for s, k in todo_perms],
                    "train16_aug": bool(todo_aug)})
    meta = {
        **base_meta,
        **(meta_extra or {}),
        "format_version": FORMAT_VERSION,
        "dataset": dataset_key,
        "identity": identity,
        "splits": meta_splits,
        "skipped_splits": {**base_meta.get("skipped_splits", {}), **(skipped_splits or {})},
        "n_classes": int(text["text_features"].shape[0]),
        "text_dtype": str(text["text_features"].dtype).replace("torch.", ""),
        "logit_scale": float(text["logit_scale"]),
        "permutations": permutations,
        "train16_aug": aug_meta,
        "image_half_matches_forward_features": checks,
        "recorded_run_check": recorded,
        "batch_size": batch_size,
        "autocast": autocast,
        "sha256": {name: _sha256(out_dir / name) for name in files},
        "created": base_meta.get("created") or history[-1]["at"],
        "updated": history[-1]["at"],
        "history": history,
    }
    for split in meta["splits"]:
        meta["skipped_splits"].pop(split, None)
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)  # last write: marks the folder complete
    return meta
