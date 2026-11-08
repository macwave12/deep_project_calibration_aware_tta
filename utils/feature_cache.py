"""On-disk format of the cached CLIP features (T01 writes it, T02 reads it).

Layout, one folder per dataset (our lowercase keys)::

    features/<dataset>/
        test_feats.pt  val_feats.pt  train16_feats.pt   (N, D) image features, dataset order
        train16_aug_feats.pt                            (views, N, D) raw image_encoder outputs of
                                                        augmented train16 views (Tip-Adapter)
        text_feats.pt                                   dict, see `TEXT_KEYS`
        perm_test_seed0.pt  perm_test_seed1.pt  ...     (N,) int64 dataset indices, stream order
        perm_val_seed0.pt  perm_val_seed1.pt  ...
        labels_test.pt  labels_val.pt  labels_train16.pt   SCORER / ANALYSIS ONLY
        meta.json                                       written last: its presence marks completion

Ground-truth discipline (RULES §2): labels and image file names (which spell out
the class for several datasets) live only in `labels_<split>.pt`. The method
side of the replay never calls `FeatureStore.labels`; `tests/test_no_label_leak.py`
checks that structurally.

Stream order: `online_tta.py` iterates `DataLoader(dataset, batch_size=1,
shuffle=True, num_workers=0)` after `set_random_seed(seed)` and a fixed set of
global-RNG consumers (building the dataset and the CLIP model). The order is drawn
from the global torch generator when iteration *starts*: one draw for the
iterator's `_base_seed`, then one for `RandomSampler`'s private generator seed,
then `randperm`. `stream_order` replays exactly that by iterating a DataLoader of
the same configuration over indices, so it needs no knowledge of torch internals
— only that it is called with the global RNG in the same state.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

FORMAT_VERSION = 1
SPLITS = ("test", "val", "train16")
DATASETS = ("dtd", "flower102", "pets", "eurosat", "aircraft")

# Our lowercase dataset key -> upstream `--test_sets` value (mixed case, and
# `eurosat` stays lowercase). Same mapping as scripts/run_one.* and tests/helpers.py.
UPSTREAM_SET_ID = {
    "dtd": "DTD",
    "flower102": "Flower102",
    "pets": "Pets",
    "eurosat": "eurosat",
    "aircraft": "Aircraft",
}

# text_feats.pt is a dict with these keys:
#   text_features            (C, D) as `model.forward_features` returns it under autocast
#   logit_scale              0-dim, `forward_features`' third output (already exp'd)
#   text_features_noautocast (C, D) `model.get_text_features()` outside autocast
#                            (what dmn_weak / onzeta read in `prepare_model_and_optimization`)
#   logit_scale_raw          0-dim, `model.logit_scale` (log value; onzeta exps it itself)
TEXT_KEYS = ("text_features", "logit_scale", "text_features_noautocast", "logit_scale_raw")


def feats_name(split: str) -> str:
    return f"{split}_feats.pt"


def perm_name(split: str, seed: int) -> str:
    return f"perm_{split}_seed{seed}.pt"


def labels_name(split: str) -> str:
    return f"labels_{split}.pt"


TEXT_NAME = "text_feats.pt"
TRAIN16_AUG_NAME = "train16_aug_feats.pt"
META_NAME = "meta.json"


class _StreamIndex(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return index


def stream_order(n: int) -> torch.Tensor:
    """Dataset indices in the order `online_tta.py`'s loader would yield them.

    Must be called with the global torch RNG in the state it has where
    `online_tta.py` reaches `for i, (images, target) in enumerate(val_loader)`.
    The loader configuration matches the runner's (`-b 1 -j 0`, shuffle=True);
    `pin_memory` draws no random numbers and is left at its default.
    Consumes the global RNG exactly as that loop does.
    """
    loader = DataLoader(_StreamIndex(n), batch_size=1, shuffle=True, num_workers=0)
    order = torch.cat([batch.reshape(-1) for batch in loader]).to(torch.int64)
    assert order.numel() == n
    return order


class FeatureStore:
    """Read access to one dataset's cached features."""

    def __init__(self, root, dataset: str):
        self.root = Path(root)
        self.dataset = dataset
        self.dir = self.root / dataset

    def meta(self) -> dict:
        path = self.dir / META_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: extraction for {self.dataset!r} did not finish "
                "(meta.json is written last)"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def image_features(self, split: str) -> torch.Tensor:
        feats = torch.load(self.dir / feats_name(split), map_location="cpu", weights_only=True)
        if not isinstance(feats, torch.Tensor) or feats.ndim != 2:
            raise ValueError(f"{feats_name(split)} must hold one (N, D) tensor")
        return feats

    def text(self) -> dict:
        data = torch.load(self.dir / TEXT_NAME, map_location="cpu", weights_only=True)
        missing = [k for k in TEXT_KEYS if k not in data]
        if missing:
            raise ValueError(f"{TEXT_NAME} lacks {missing}")
        return data

    def permutation(self, split: str, seed: int) -> torch.Tensor:
        path = self.dir / perm_name(split, seed)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; extract it with --seeds including {seed}"
            )
        return torch.load(path, map_location="cpu", weights_only=True).to(torch.int64)

    def train16_augmented(self) -> torch.Tensor:
        """(views, N, D) raw (unnormalised) features of augmented train16 views."""
        return torch.load(self.dir / TRAIN16_AUG_NAME, map_location="cpu", weights_only=True)

    def labels(self, split: str) -> torch.Tensor:
        """Ground truth in dataset order. Scorer / analysis code only (RULES §2)."""
        data = torch.load(self.dir / labels_name(split), map_location="cpu", weights_only=False)
        return data["labels"].to(torch.int64)


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Write via a temp file so an interrupted run never leaves a truncated file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp)
    tmp.replace(path)
