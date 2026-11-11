"""Stream orders for seeds the GPU extraction did not export (seed policy, DECISIONS 2026-11-10).

Seeds 0/1/2 are the exact orders of the original runs (extracted with the features). Seeds
>= 3 are CPU-generated: `torch.Generator().manual_seed(GENERATOR_OFFSET + seed).randperm(N)`.
Existing files are never overwritten, so an extracted order always wins.

    python scripts/make_permutations.py --seeds 3 4 5 6 7 8 9 [--split test] [--datasets ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.feature_cache import ALL_DATASETS, DATASETS, labels_name, perm_name  # noqa: E402

GENERATOR_OFFSET = 10_000
EXTRACTED_SEEDS = (0, 1, 2)


def cpu_permutation(n: int, seed: int) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(GENERATOR_OFFSET + seed))


def write_permutations(features_root, dataset: str, split: str, seeds) -> list[Path]:
    d = Path(features_root) / dataset
    n = len(torch.load(d / labels_name(split), map_location="cpu", weights_only=False)["labels"])
    written = []
    for seed in seeds:
        if seed in EXTRACTED_SEEDS:
            raise ValueError(f"seed {seed} is an extracted order; refusing to generate it")
        path = d / perm_name(split, seed)
        if path.exists():
            continue
        torch.save(cpu_permutation(n, seed), path)
        written.append(path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default="features")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=ALL_DATASETS)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args(argv)
    for ds in a.datasets:
        paths = write_permutations(a.features, ds, a.split, a.seeds)
        print(f"[{ds}/{a.split}] wrote {len(paths)} permutation(s)")


if __name__ == "__main__":
    main()
