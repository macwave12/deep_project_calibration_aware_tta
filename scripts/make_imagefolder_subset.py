"""Reduce an ImageFolder evaluation set to a fixed random subset (CPU extraction budget).

Images not selected are moved (not deleted) to `<dataset folder>/_excluded/`, keeping their
class folders, so moving them back undoes the subset. Selection:
`random.Random(SEED).sample(sorted(all class-folder files), SIZE)`. A marker
`<dataset folder>/_subset.json` records it; rerunning with the same size is a no-op.

    python scripts/make_imagefolder_subset.py --data-root /path --dataset imagenet_r --size 10000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEED = 0


def class_files(image_root: Path) -> list[str]:
    return sorted(p.relative_to(image_root).as_posix()
                  for p in image_root.glob("*/*") if p.is_file())


def make_subset(ds_dir: Path, image_dir: str, size: int, seed: int = SEED) -> dict:
    root = ds_dir / image_dir
    marker = ds_dir / "_subset.json"
    if marker.exists():
        info = json.loads(marker.read_text(encoding="utf-8"))
        if info["size"] == size and info["seed"] == seed:
            return info
        raise ValueError(f"{ds_dir} already holds a different subset: {info}")
    files = class_files(root)
    keep = set(random.Random(seed).sample(files, size))
    excluded = ds_dir / "_excluded"
    for rel in files:
        if rel not in keep:
            dst = excluded / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / rel), str(dst))
    info = {"seed": seed, "size": size, "total": len(files), "image_dir": image_dir}
    marker.write_text(json.dumps(info), encoding="utf-8")
    return info


def main(argv=None):
    from scripts.download_datasets import ALL_SOURCES
    choices = [k for k, s in ALL_SOURCES.items() if s.kind == "imagefolder"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True, choices=choices)
    ap.add_argument("--size", type=int, required=True)
    a = ap.parse_args(argv)
    spec = ALL_SOURCES[a.dataset]
    info = make_subset(Path(a.data_root) / spec.folder, spec.image_dir, a.size)
    print(f"[{a.dataset}] {info['total']} -> {info['size']} images (seed {info['seed']})")


if __name__ == "__main__":
    main()
