"""T01: the project's only GPU job. Export CLIP features for the CPU replay.

Run from the repository root (see README, "Feature cache and CPU replay"):

    python scripts/extract_features.py --data-root <DATA_ROOT> --out features

Builds the model and the data exactly as `online_tta.py` does for
`clipzs`/`tda`/`cal_tda` with `scripts/run_one.*` (ViT-B/16, `--ctx_init
a_photo_of_a`, `-b 1 -j 0`, Resize(224, BICUBIC) -> CenterCrop(224) -> RGB ->
ToTensor -> CLIP normalize, fp16 autocast) and writes `utils/feature_cache.py`'s
layout. Before any GPU work it checks the torch version against
requirements.txt and that every requested split's list file and images exist
(unavailable splits are skipped and recorded). After extraction it checks the
stream order against the recorded runs' `save_results.pt`, when present.
Re-running the same command resumes: finished pieces are kept, missing ones added.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASETS = ("dtd", "flower102", "pets", "eurosat", "aircraft")
DEV_DATASETS = ("imagenetv2", "caltech101", "food101", "ucf101")  # T14 development sets
FRESH_DATASETS = ("imagenet", "imagenet_a", "imagenet_r", "imagenet_sketch",  # T15 fresh evaluation
                  "stl10", "cifar10", "cifar100", "voc2007", "gtsrb", "cars")
WDS_SET_IDS = {"STL10": "stl10", "CIFAR10": "cifar10", "CIFAR100": "cifar100", "VOC2007": "voc2007",
               "GTSRB": "gtsrb", "StanfordCarsWDS": "cars"}  # set id -> key in data/wds_classnames.json
ALL_DATASETS = DATASETS + DEV_DATASETS + FRESH_DATASETS
TEST_ONLY_SET_IDS = ("I", "A", "K", "R", "V") + tuple(WDS_SET_IDS)  # ImageFolder sets ignore `mode`
NORMALIZE_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORMALIZE_STD = [0.26862954, 0.26130258, 0.27577711]


def _convert_image_to_rgb(image):
    # Module level (not a closure as in online_tta.py) so workers > 0 can pickle it.
    return image.convert("RGB")


def build_transform(resolution: int):
    """`online_tta.py`'s `data_transform` for the single-view methods, verbatim."""
    import torchvision.transforms as transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose([
        transforms.Resize(size=resolution, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=(resolution, resolution)),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def build_tip_adapter_transform(resolution: int):
    """Tip-Adapter's official cache-building train transform (main.py, `train_tranform`)."""
    import torchvision.transforms as transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose([
        transforms.RandomResizedCrop(size=resolution, scale=(0.5, 1), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def class_names(set_id: str) -> list:
    """Class names in label order. ImageNet variants use `imagenet_classes`: ImageNet-A / -R keep
    their 200 classes in ImageNet index order (= ImageFolder's sorted-wnid order, as online_tta.py
    masks them); ImageNetV2 is relabelled to ImageNet indices (`imagenetv2_numeric_labels`)."""
    if set_id in WDS_SET_IDS:
        import json
        names = json.loads((REPO / "data" / "wds_classnames.json").read_text(encoding="utf-8"))
        return list(names[WDS_SET_IDS[set_id]])
    if set_id in ("I", "V", "K", "A", "R"):
        from data.imagnet_prompts import imagenet_classes
        if set_id == "A":
            from data.imagenet_variants import imagenet_a_mask
            return [imagenet_classes[i] for i in imagenet_a_mask]
        if set_id == "R":
            from data.imagenet_variants import imagenet_r_mask
            return [c for c, m in zip(imagenet_classes, imagenet_r_mask) if m]
        return list(imagenet_classes)
    from data import cls_to_names
    return getattr(cls_to_names, f"{set_id.lower()}_classes")


def pinned_torch_version() -> str | None:
    """The `torch==` pin in requirements.txt (the version the recorded runs used)."""
    try:
        text = (REPO / "requirements.txt").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^\s*torch==(\S+)", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def expected_torch_version(pinned: str | None, use_cuda: bool) -> str | None:
    """The pin on a GPU machine; its CPU build (`<version>+cpu`) on a CPU-only one."""
    if not pinned:
        return None
    return pinned if use_cuda else pinned.split("+")[0] + "+cpu"


def missing_files(dataset) -> list:
    """Image files an upstream dataset lists but the disk lacks."""
    files = getattr(dataset, "image_list", None)
    if not files:
        return []
    if hasattr(dataset, "image_path"):
        base = dataset.image_path
    elif hasattr(dataset, "path"):  # Aircraft
        base = os.path.join(dataset.path, "images")
    else:
        return []
    return [f for f in files if not os.path.isfile(os.path.join(base, f))]


def _git_commit() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT"),
                   help="dataset root (same as run_one's DATA_ROOT); defaults to $DATA_ROOT")
    p.add_argument("--out", default="features", help="output folder (default: features/)")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=ALL_DATASETS,
                   help="default: the five project datasets; T14 dev sets: " + " ".join(DEV_DATASETS)
                        + "; T15 fresh sets: " + " ".join(FRESH_DATASETS))
    p.add_argument("--splits", nargs="+", default=["test", "val", "train16"], choices=["test", "val", "train16"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="stream seeds for the test split")
    p.add_argument("--val-seeds", nargs="+", type=int, default=[0, 1, 2], help="stream seeds for the val split")
    p.add_argument("--train16-views", type=int, default=10,
                   help="augmented train16 views for Tip-Adapter (0 = none); default 10")
    p.add_argument("--aug-seed", type=int, default=1, help="seed of the augmented-view pass")
    p.add_argument("--arch", default="ViT-B/16")
    p.add_argument("--ctx-init", default="a_photo_of_a")
    p.add_argument("--n-ctx", type=int, default=4, help="online_tta's default; ignored when --ctx-init is set")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1,
                   help="1 (default) matches the runs' -b 1 kernel shapes; larger is faster but may "
                        "differ from them in the last bits")
    p.add_argument("--workers", type=int, default=0, help="dataloader workers (runs used 0)")
    p.add_argument("--recorded-runs", default="online_results/ckps",
                   help="where online_tta.py left save_results.pt (run_one's --output_dir root)")
    p.add_argument("--overwrite", action="store_true", help="redo datasets that already have meta.json")
    p.add_argument("--force", action="store_true",
                   help="run even if torch differs from the requirements.txt pin")
    p.add_argument("--allow-cpu", action="store_true",
                   help="extract on CPU (no GPU): fp32 features without CUDA autocast, so they differ "
                        "from the GPU runs in the last bits; expects the CPU torch build (2.11.0+cpu)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.data_root:
        sys.exit("Set DATA_ROOT or pass --data-root")
    args.data_root = args.data_root.rstrip("/\\") or args.data_root
    # data/fewshot_datasets.py reads DATA_ROOT at import time for the split JSONs.
    os.environ["DATA_ROOT"] = args.data_root

    import torch
    import torch.backends.cudnn as cudnn

    pinned = pinned_torch_version()
    use_cuda = torch.cuda.is_available()
    expected = expected_torch_version(pinned, use_cuda)
    print("=" * 72)
    print(f"torch {torch.__version__}   (requirements.txt pins {pinned})   "
          f"CUDA {torch.version.cuda}   available: {torch.cuda.is_available()}")
    print("=" * 72, flush=True)
    print(f"mode: {'GPU (CUDA autocast, as the recorded runs)' if use_cuda else 'CPU (fp32, no autocast)'}; "
          f"expected torch {expected}", flush=True)
    if expected and torch.__version__ != expected and not args.force:
        sys.exit(f"torch {torch.__version__} != expected {expected}: the stream order and features are only "
                 "known to match the recorded runs under the same torch. Install the pin, or pass "
                 "--force to proceed anyway (the version is recorded in meta.json).")

    from clip.custom_clip import get_coop
    from data.datautils import build_dataset
    from utils.feature_cache import UPSTREAM_SET_ID, FeatureStore
    from utils.feature_extraction import extract_dataset, preflight
    from utils.recorded_run_check import check_dataset

    if not use_cuda and not args.allow_cpu:
        sys.exit("No CUDA device. Pass --allow-cpu to extract on CPU (slower; fp32 features, see the handoff).")
    device = f"cuda:{args.gpu}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True  # as online_tta.py
    transform = build_transform(args.resolution)
    aug_transform = build_tip_adapter_transform(args.resolution)

    def split_builder(set_id):
        def build_split(split):
            if split == "train16":
                return build_dataset(set_id, transform, args.data_root, mode="train", n_shot=16)
            return build_dataset(set_id, transform, args.data_root, mode=split)
        return build_split

    # Preflight: every list file and image, before any model is built.
    plan, skipped = {}, {}
    for key in args.datasets:
        splits = args.splits
        if UPSTREAM_SET_ID[key] in TEST_ONLY_SET_IDS and splits != ["test"]:
            # ImageFolder sets ignore `mode`: a "val" or "train16" split would be the test set.
            print(f"[{key}] test split only (an ImageFolder set has no val/train split)")
            splits = ["test"]
        available, skipped[key] = preflight(key, splits, split_builder(UPSTREAM_SET_ID[key]),
                                            missing_files)
        if "test" not in available:
            print(f"[{key}] SKIPPED: test split unavailable ({skipped[key].get('test')})")
            continue
        plan[key] = available
        for split, reason in skipped[key].items():
            print(f"[{key}] split {split} skipped: {reason}")
    print("preflight: " + ", ".join(f"{k}: {'/'.join(v)}" for k, v in plan.items()), flush=True)
    if not plan:
        sys.exit("nothing to extract")

    gpu_name = torch.cuda.get_device_name(args.gpu) if use_cuda else None
    identity = {
        "arch": args.arch, "ctx_init": args.ctx_init, "n_ctx": args.n_ctx,
        "resolution": args.resolution, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": gpu_name, "device": "cuda" if use_cuda else "cpu",
    }
    env = {
        "arch": args.arch,
        "ctx_init": args.ctx_init,
        "prompt_template": args.ctx_init.replace("_", " ") + " {class}.",
        "resolution": args.resolution,
        "transform": repr(transform),
        "train16_aug_transform": repr(aug_transform),
        "torch": torch.__version__,
        "torch_pinned": pinned,
        "torch_expected": expected,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if use_cuda else None,
        "gpu": gpu_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_commit(),
        "extractor_args": {k: v for k, v in vars(args).items() if k != "data_root"},
    }
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def free_memory():
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    verdicts = {}
    for key, splits in plan.items():
        set_id = UPSTREAM_SET_ID[key]
        classnames = class_names(set_id)

        def build_model(classnames=classnames):
            # online_tta.py: get_coop(arch, classnames, args.gpu, n_ctx, ctx_init), then .cuda(gpu).
            model = get_coop(args.arch, classnames, args.gpu if use_cuda else "cpu", args.n_ctx, args.ctx_init)
            if use_cuda:
                model = model.cuda(args.gpu)
            return model.eval()

        def build_aug_split(set_id=set_id):
            return build_dataset(set_id, aug_transform, args.data_root, mode="train", n_shot=16)

        def post_check(out_dir, seeds, key=key, set_id=set_id):
            autocast = (lambda: torch.cuda.amp.autocast()) if use_cuda else None
            return check_dataset(FeatureStore(out_root, key), set_id, args.recorded_runs, seeds,
                                 device=device, autocast=autocast,
                                 # CPU features are fp32: compare logits, but judge order by labels.
                                 logits_must_match=use_cuda)

        t0 = time.time()
        meta = extract_dataset(
            out_root, key,
            build_split=split_builder(set_id), build_model=build_model, free_memory=free_memory,
            device=device, test_seeds=args.seeds, val_seeds=args.val_seeds, splits=splits,
            train16_views=args.train16_views, aug_seed=args.aug_seed, build_aug_split=build_aug_split,
            batch_size=args.batch_size, workers=args.workers, autocast=use_cuda,
            identity=identity, skipped_splits=skipped[key], post_check=post_check,
            meta_extra={**env, "upstream_set_id": set_id, "classnames": list(classnames)},
            overwrite=args.overwrite,
        )
        check = (meta.get("recorded_run_check") or {}).get("stream_order_matches_recorded_run")
        verdicts[key] = check
        print(f"[{key}] done in {time.time() - t0:.0f}s; stream order matches recorded run: "
              f"{'n/a (no save_results.pt)' if check is None else check}", flush=True)

    total = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    print(f"all done in {time.time() - started:.0f}s; {out_root} holds {total / 1e6:.1f} MB")
    print("stream order matches recorded run: " + ", ".join(f"{k}={v}" for k, v in verdicts.items()))
    if any(v is False for v in verdicts.values()):
        print("WARNING: a stream order does NOT match its recorded run; report this before replaying.")
    return verdicts


if __name__ == "__main__":
    main()
