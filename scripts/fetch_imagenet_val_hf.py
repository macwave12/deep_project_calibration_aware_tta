"""ImageNet-1k validation (50k) from the gated Hugging Face dataset `ILSVRC/imagenet-1k`.

Needs a Hugging Face token whose account has accepted the dataset's terms (env HF_TOKEN or
--token). Downloads the 14 `data/validation-*.parquet` shards (6.7 GB) and writes the layout
`data/datautils.py` reads for set id 'I': `<data-root>/imagenet/images/val/<label:04d>/<n>.JPEG`.
Zero-padded label folders sort in ImageNet index order, which is `imagenet_classes`' order.
Resumable: finished shards are skipped (one marker per shard). Needs pyarrow (present on Colab).

    HF_TOKEN=... python scripts/fetch_imagenet_val_hf.py --data-root /content/data_root
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

REPO_URL = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data"
N_SHARDS = 14
EXPECTED = 50000


def shard_name(i: int) -> str:
    return f"validation-{i:05d}-of-{N_SHARDS:05d}.parquet"


def label_dir(val_root: Path, label: int) -> Path:
    return val_root / f"{int(label):04d}"


def write_rows(rows, val_root: Path, start: int) -> int:
    """rows: iterable of (image bytes, label). Returns the next running index."""
    i = start
    for data, label in rows:
        d = label_dir(val_root, label)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{i:05d}.JPEG").write_bytes(data)
        i += 1
    return i


def download(url: str, dest: Path, token: str) -> None:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN"))
    a = ap.parse_args(argv)
    if not a.token:
        raise SystemExit("set HF_TOKEN: a token whose account accepted ILSVRC/imagenet-1k's terms")
    import pyarrow.parquet as pq

    root = Path(a.data_root) / "imagenet"
    val_root, shards, state = root / "images" / "val", root / "_shards", root / "_state"
    for d in (val_root, shards, state):
        d.mkdir(parents=True, exist_ok=True)
    index = 0
    for i in range(N_SHARDS):
        name = shard_name(i)
        done = state / f"{name}.done"
        if done.exists():
            index = int(done.read_text())
            continue
        path = shards / name
        if not path.exists():
            print(f"downloading {name}", flush=True)
            download(f"{REPO_URL}/{name}", path, a.token)
        table = pq.read_table(path, columns=["image", "label"]).to_pydict()
        rows = ((img["bytes"], lab) for img, lab in zip(table["image"], table["label"]))
        index = write_rows(rows, val_root, index)
        done.write_text(str(index))
        path.unlink()
        print(f"{name}: {index} images so far", flush=True)
    n = sum(1 for _ in val_root.glob("*/*.JPEG"))
    print(f"ImageNet val: {n} images in {sum(1 for _ in val_root.iterdir())} classes")
    if n != EXPECTED:
        raise SystemExit(f"expected {EXPECTED} images, found {n}")


if __name__ == "__main__":
    main()
