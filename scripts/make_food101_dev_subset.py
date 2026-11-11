"""T14 dev set: Food101's test split reduced to a fixed random subset (CPU extraction budget).

Keeps the full CoOp split as `split_zhou_Food101.full.json` and writes
`split_zhou_Food101.json` with train/val unchanged and `test` = `random.Random(SEED).sample(
full_test, SIZE)`, sorted by path. Idempotent: always rebuilt from the `.full.json` copy.

    python scripts/make_food101_dev_subset.py --data-root /path/to/data_root
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

SEED = 0
SIZE = 10_000


def subset(splits: dict, seed: int = SEED, size: int = SIZE) -> dict:
    test = random.Random(seed).sample(splits["test"], size)
    test.sort(key=lambda e: e[0])
    return {"train": splits["train"], "val": splits["val"], "test": test}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    a = ap.parse_args(argv)
    d = Path(a.data_root) / "food-101"
    full, out = d / "split_zhou_Food101.full.json", d / "split_zhou_Food101.json"
    if not full.exists():
        shutil.copyfile(out, full)
    splits = json.loads(full.read_text(encoding="utf-8"))
    out.write_text(json.dumps(subset(splits)), encoding="utf-8")
    print(f"{out}: test {len(splits['test'])} -> {SIZE} (seed {SEED})")


if __name__ == "__main__":
    main()
