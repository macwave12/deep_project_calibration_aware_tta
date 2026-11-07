"""T01 acceptance, on CPU with a stub model and a tiny fake dataset."""

import json
import warnings

import pytest
import torch
from torch.utils.data import DataLoader

from tests.replay_stubs import FakeImageDataset, StubCLIP, consume_rng_like_model_init
from utils.feature_cache import TEXT_KEYS, FeatureStore, labels_name, stream_order
from utils.feature_extraction import extract_dataset, stream_permutation
from utils.tools import set_random_seed


def _runner_stream(seed, dataset_factory, preamble):
    """Dataset indices in the order `online_tta.py`'s loop sees them.

    Same steps as the runner: seed, build dataset, build model, then
    `DataLoader(..., batch_size=1, shuffle=True, num_workers=0, pin_memory=True)`
    and iterate. Indices are recovered from the yielded images.
    """
    set_random_seed(seed)
    dataset = dataset_factory()
    preamble()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pin_memory without CUDA
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        order = []
        for images, _ in loader:
            match = (dataset.images == images[0]).flatten(1).all(dim=1).nonzero()
            assert match.numel() == 1
            order.append(int(match))
    return torch.tensor(order)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stream_order_is_the_shuffled_loader_order_after_the_same_preamble(seed):
    def preamble():
        FakeImageDataset()
        return consume_rng_like_model_init()

    expected = _runner_stream(seed, FakeImageDataset, consume_rng_like_model_init)
    order, model = stream_permutation(len(FakeImageDataset()), seed, preamble)
    assert torch.equal(order, expected)
    assert isinstance(model, StubCLIP)
    assert sorted(order.tolist()) == list(range(len(order)))


def test_the_preamble_matters():
    # Control: skipping the model-init RNG draws gives a different stream, so
    # the extractor must repeat them (and the replay must not re-derive orders).
    with_init = stream_permutation(48, 0, consume_rng_like_model_init)[0]
    without = stream_permutation(48, 0, lambda: None)[0]
    assert not torch.equal(with_init, without)


def test_seeds_give_different_streams():
    orders = [stream_permutation(48, s, consume_rng_like_model_init)[0] for s in (0, 1, 2)]
    assert not torch.equal(orders[0], orders[1])
    assert not torch.equal(orders[1], orders[2])


def test_stream_order_is_a_pure_function_of_rng_state():
    torch.manual_seed(5)
    a = stream_order(30)
    torch.manual_seed(5)
    b = stream_order(30)
    assert torch.equal(a, b)


SPLIT_SIZES = {"test": 48, "val": 20, "train16": 10}


def _build_split(name):
    return FakeImageDataset(n=SPLIT_SIZES[name], seed={"test": 7, "val": 8, "train16": 9}[name])


def _extract(tmp_path, **kw):
    logs = []
    meta = extract_dataset(
        tmp_path, "dtd", build_split=_build_split, build_model=consume_rng_like_model_init,
        device="cpu", autocast=False, log=logs.append, meta_extra={"arch": "stub"}, **kw,
    )
    return meta, logs


def test_extraction_writes_the_documented_layout(tmp_path):
    meta, _ = _extract(tmp_path)
    folder = tmp_path / "dtd"
    expected = {
        "test_feats.pt", "val_feats.pt", "train16_feats.pt", "text_feats.pt",
        "perm_test_seed0.pt", "perm_test_seed1.pt", "perm_test_seed2.pt", "perm_val_seed0.pt",
        "labels_test.pt", "labels_val.pt", "labels_train16.pt", "meta.json",
    }
    assert {p.name for p in folder.iterdir()} == expected

    assert meta["splits"]["test"] == {"n": 48, "dim": 16, "dtype": "float32"}
    assert meta["n_classes"] == 5
    assert meta["arch"] == "stub"
    assert set(meta["sha256"]) == expected - {"meta.json"}
    assert json.loads((folder / "meta.json").read_text())["dataset"] == "dtd"
    for split in SPLIT_SIZES:
        assert meta["forward_features_check"][split] == {"dtype_equal": True, "max_abs_diff": 0.0}


def test_features_are_in_dataset_order_and_unchanged(tmp_path):
    _extract(tmp_path)
    store = FeatureStore(tmp_path, "dtd")
    model = StubCLIP()
    for split in SPLIT_SIZES:
        dataset = _build_split(split)
        feats = store.image_features(split)
        reference, _, _ = model.forward_features(dataset.images)
        assert feats.dtype == reference.dtype  # kept as returned, never cast
        for i in range(len(dataset)):
            single, _, _ = model.forward_features(dataset.images[i : i + 1])
            assert torch.equal(feats[i : i + 1], single)


def test_permutations_match_the_runner_stream(tmp_path):
    _extract(tmp_path)
    store = FeatureStore(tmp_path, "dtd")
    for split, seed in (("test", 0), ("test", 1), ("test", 2), ("val", 0)):
        expected = _runner_stream(seed, lambda: _build_split(split), consume_rng_like_model_init)
        assert torch.equal(store.permutation(split, seed), expected)


def test_labels_live_only_in_the_labels_file(tmp_path):
    _extract(tmp_path)
    folder = tmp_path / "dtd"
    store = FeatureStore(tmp_path, "dtd")
    for split in SPLIT_SIZES:
        feats = torch.load(folder / f"{split}_feats.pt", weights_only=True)
        assert isinstance(feats, torch.Tensor) and feats.ndim == 2  # nothing else inside
        assert torch.equal(store.labels(split), _build_split(split).labels)
        labels_file = torch.load(folder / labels_name(split), weights_only=False)
        assert labels_file["files"] == _build_split(split).image_list
    text = torch.load(folder / "text_feats.pt", weights_only=True)
    assert set(text) == set(TEXT_KEYS)
    meta_text = (folder / "meta.json").read_text()
    assert "class0/img_" not in meta_text  # file names spell out classes


def test_text_side_is_what_the_methods_read(tmp_path):
    _extract(tmp_path)
    text = FeatureStore(tmp_path, "dtd").text()
    model = StubCLIP()
    assert torch.equal(text["text_features"], model.get_text_features())
    assert torch.equal(text["text_features_noautocast"], model.get_text_features())
    assert torch.equal(text["logit_scale"], model.logit_scale.exp())
    assert torch.equal(text["logit_scale_raw"], model.logit_scale)


def test_a_finished_dataset_is_skipped_unless_overwrite(tmp_path):
    _extract(tmp_path)
    stamp = (tmp_path / "dtd" / "test_feats.pt").stat().st_mtime_ns
    _, logs = _extract(tmp_path)
    assert any("skipping" in line for line in logs)
    assert (tmp_path / "dtd" / "test_feats.pt").stat().st_mtime_ns == stamp
    _, logs = _extract(tmp_path, overwrite=True)
    assert not any("skipping" in line for line in logs)


def test_an_unfinished_folder_is_refused_by_the_reader(tmp_path):
    _extract(tmp_path)
    (tmp_path / "dtd" / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="did not finish"):
        FeatureStore(tmp_path, "dtd").meta()
