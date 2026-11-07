"""T01: the project's only GPU job. Export CLIP features for the CPU replay.

Run from the repository root (see README, "Feature cache"):

    python scripts/extract_features.py --data-root <DATA_ROOT> --out features

Builds the model and the data exactly as `online_tta.py` does for
`clipzs`/`tda`/`cal_tda` with `scripts/run_one.*` (ViT-B/16, `--ctx_init
a_photo_of_a`, `-b 1 -j 0`, Resize(224, BICUBIC) -> CenterCrop(224) -> RGB ->
ToTensor -> CLIP normalize, fp16 autocast) and writes `utils/feature_cache.py`'s
layout. A dataset whose `meta.json` exists is skipped, so an interrupted run is
resumed by running the same command again.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASETS = ("dtd", "flower102", "pets", "eurosat", "aircraft")
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
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--splits", nargs="+", default=["test", "val", "train16"], choices=["test", "val", "train16"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="stream seeds for the test split")
    p.add_argument("--val-seeds", nargs="+", type=int, default=[0], help="stream seeds for the val split")
    p.add_argument("--arch", default="ViT-B/16")
    p.add_argument("--ctx-init", default="a_photo_of_a")
    p.add_argument("--n-ctx", type=int, default=4, help="online_tta's default; ignored when --ctx-init is set")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1,
                   help="1 (default) matches the runs' -b 1 kernel shapes; larger is faster but may "
                        "differ from them in the last bits")
    p.add_argument("--workers", type=int, default=0, help="dataloader workers (runs used 0)")
    p.add_argument("--overwrite", action="store_true", help="redo datasets that already have meta.json")
    p.add_argument("--allow-cpu", action="store_true",
                   help="debug only: CPU features are fp32 without autocast and do NOT match the GPU runs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.data_root:
        sys.exit("Set DATA_ROOT or pass --data-root")
    # data/fewshot_datasets.py reads DATA_ROOT at import time for the split JSONs.
    os.environ["DATA_ROOT"] = args.data_root

    import torch
    import torch.backends.cudnn as cudnn

    from clip.custom_clip import get_coop
    from data import cls_to_names
    from data.datautils import build_dataset
    from utils.feature_cache import UPSTREAM_SET_ID
    from utils.feature_extraction import extract_dataset

    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu:
        sys.exit("No CUDA device. This job must run on the GPU (pass --allow-cpu only to debug).")
    device = f"cuda:{args.gpu}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True  # as online_tta.py
    transform = build_transform(args.resolution)

    env = {
        "arch": args.arch,
        "ctx_init": args.ctx_init,
        "prompt_template": args.ctx_init.replace("_", " ") + " {class}.",
        "resolution": args.resolution,
        "transform": repr(transform),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if use_cuda else None,
        "gpu": torch.cuda.get_device_name(args.gpu) if use_cuda else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_commit(),
        "extractor_args": {k: v for k, v in vars(args).items() if k != "data_root"},
    }
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for key in args.datasets:
        set_id = UPSTREAM_SET_ID[key]
        classnames = getattr(cls_to_names, f"{set_id.lower()}_classes")

        def build_split(split, set_id=set_id):
            if split == "train16":
                return build_dataset(set_id, transform, args.data_root, mode="train", n_shot=16)
            return build_dataset(set_id, transform, args.data_root, mode=split)

        def build_model(classnames=classnames):
            # online_tta.py: get_coop(arch, classnames, args.gpu, n_ctx, ctx_init), then .cuda(gpu).
            model = get_coop(args.arch, classnames, args.gpu if use_cuda else "cpu", args.n_ctx, args.ctx_init)
            if use_cuda:
                model = model.cuda(args.gpu)
            return model.eval()

        def release_model(model):
            del model
            if use_cuda:
                torch.cuda.empty_cache()

        t0 = time.time()
        extract_dataset(
            out_root, key,
            build_split=build_split, build_model=build_model, release_model=release_model,
            device=device, test_seeds=args.seeds, val_seeds=args.val_seeds, splits=args.splits,
            batch_size=args.batch_size, workers=args.workers, autocast=use_cuda,
            meta_extra={**env, "upstream_set_id": set_id, "classnames": list(classnames)},
            overwrite=args.overwrite,
        )
        print(f"[{key}] done in {time.time() - t0:.0f}s", flush=True)

    total = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    print(f"all done in {time.time() - started:.0f}s; {out_root} holds {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
