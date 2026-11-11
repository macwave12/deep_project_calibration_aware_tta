"""T15 fresh evaluation sets from LAION CLIP_benchmark's webdataset mirrors on Hugging Face.

Downloads `clip-benchmark/wds_<name>/test/<i>.tar` (public, no token) and writes an
ImageFolder layout that `data/datautils.py` reads: `<data-root>/<folder>/<image_dir>/<class>/<key>.<ext>`.
A sample is `<key>.cls` (the class index as text) plus one image (`.webp`, `.jpg` or `.png`).
Class folders are the zero-padded index (`0000`, ...), which ImageFolder sorts in index order;
ImageNet-R uses its sorted wnids instead, matching `imagenet_r_mask`. Resumable: one marker per
finished shard. Class names: `data/wds_classnames.json` (ImageNet-R: `imagenet_classes`).

    python scripts/fetch_wds.py --data-root /content/data_root --datasets stl10 cifar10 imagenet_r
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/clip-benchmark/wds_{name}/resolve/main/test/{file}"
IMAGE_EXTS = (".webp", ".jpg", ".jpeg", ".png")

# our key: (webdataset name, folder under data root, image dir, number of test shards)
WDS = {
    "stl10": ("stl10", "stl10", "images", 2),
    "cifar10": ("vtab-cifar10", "cifar10", "images", 4),
    "cifar100": ("vtab-cifar100", "cifar100", "images", 4),
    "voc2007": ("voc2007", "voc2007", "images", 6),
    "gtsrb": ("gtsrb", "gtsrb", "images", 3),
    "cars": ("cars", "stanford-cars-wds", "images", 5),
    "imagenet_r": ("imagenet-r", "imagenet-rendition", "imagenet-r", 4),
}


def class_folder(key: str, cls: int) -> str:
    if key == "imagenet_r":
        from data.imagenet_variants import imagenet_r_wnids
        return sorted(imagenet_r_wnids)[cls]
    return f"{cls:04d}"


def convert_tar(fileobj, out_dir: Path, key: str) -> int:
    """Stream one webdataset tar into class folders; returns the number of images written."""
    pending: dict[str, dict] = {}
    written = 0
    with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            stem, dot, ext = member.name.rpartition(".")
            if not dot:
                continue
            ext = "." + ext.lower()
            data = tf.extractfile(member).read()
            entry = pending.setdefault(stem, {})
            if ext == ".cls":
                entry["cls"] = int(data.decode().strip())
            elif ext in IMAGE_EXTS:
                entry["img"], entry["ext"] = data, ext
            if "cls" in entry and "img" in entry:
                d = out_dir / class_folder(key, entry["cls"])
                d.mkdir(parents=True, exist_ok=True)
                (d / (Path(stem).name + entry["ext"])).write_bytes(entry["img"])
                written += 1
                del pending[stem]
    return written


def fetch(key: str, data_root: Path) -> int:
    name, folder, image_dir, n_shards = WDS[key]
    out_dir = data_root / folder / image_dir
    state = data_root / folder / "_wds_state"
    state.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(n_shards):
        done = state / f"{i}.done"
        if done.exists():
            total += int(done.read_text())
            continue
        url = HF.format(name=name, file=f"{i}.tar")
        print(f"[{key}] shard {i + 1}/{n_shards}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            n = convert_tar(resp, out_dir, key)
        done.write_text(str(n))
        total += n
    print(f"[{key}] {total} test images in {out_dir}", flush=True)
    return total


def main(argv=None):
    import sys
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, choices=list(WDS))
    a = ap.parse_args(argv)
    for key in a.datasets:
        fetch(key, Path(a.data_root))


if __name__ == "__main__":
    main()
