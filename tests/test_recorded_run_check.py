"""The exact stream-order check against `online_tta.py`'s save_results.pt."""

import warnings
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from tests.replay_stubs import FakeImageDataset, consume_rng_like_model_init
from utils.feature_cache import FeatureStore
from utils.feature_extraction import extract_dataset
from utils.feature_replay import build_trainer
from utils.recorded_run_check import check_dataset, find_recorded_runs
from utils.tools import set_random_seed

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
SIZES = {"test": 48, "val": 12, "train16": 10}


def _build_split(name):
    return FakeImageDataset(n=SIZES[name], seed={"test": 7, "val": 8, "train16": 9}[name])


def _record_run(ckpts, algorithm, seed):
    """`online_tta.main()`'s loop + its `save_results.pt`, on CPU with the stub."""
    set_random_seed(seed)
    dataset = _build_split("test")
    model = consume_rng_like_model_init()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        trainer = build_trainer(algorithm, model, {})
        args = SimpleNamespace(test_sets="DTD")
        trainer.prepare_model_and_optimization(args)
        outputs, targets = [], []
        for images, target in loader:
            outputs.append(trainer.adaptation_process(None, images, args)["output"])
            targets.append(target)
    folder = ckpts / algorithm / "bs1" / "ViT-B" / "16" / "DTD"
    folder.mkdir(parents=True)
    path = folder / "save_results.pt"
    torch.save({"output": torch.cat(outputs), "target": torch.cat(targets)}, path)
    return path


@pytest.fixture
def store(tmp_path):
    extract_dataset(tmp_path / "features", "dtd", build_split=_build_split,
                    build_model=consume_rng_like_model_init, device="cpu", autocast=False,
                    log=lambda *_: None, splits=("test",))
    return FeatureStore(tmp_path / "features", "dtd")


def test_a_matching_recorded_run_is_confirmed_with_its_seed(store, tmp_path):
    ckpts = tmp_path / "ckpts"
    _record_run(ckpts, "clipzs", seed=0)
    _record_run(ckpts, "tda", seed=1)
    assert len(find_recorded_runs(ckpts, "clipzs", "DTD")) == 1
    result = check_dataset(store, "DTD", ckpts, [0, 1, 2], log=lambda *_: None)
    assert result["stream_order_matches_recorded_run"] is True
    assert result["clipzs"]["matching_seed"] == 0
    # Batched vs one-row-at-a-time matmul: summation order only.
    assert result["clipzs"]["clipzs_output"]["max_abs_logit_diff"] < 1e-4
    assert result["clipzs"]["clipzs_output"]["argmax_agreement"] == 1.0
    assert result["tda"]["matching_seed"] == 1


def test_a_different_order_is_caught(store, tmp_path):
    ckpts = tmp_path / "ckpts"
    path = _record_run(ckpts, "tda", seed=0)
    saved = torch.load(path, weights_only=False)
    target = saved["target"]
    i, j = 0, int((target != target[0]).nonzero()[0])  # swap two steps with different labels
    target[[i, j]] = target[[j, i]]
    torch.save(saved, path)
    result = check_dataset(store, "DTD", ckpts, [0, 1, 2], log=lambda *_: None)
    assert result["stream_order_matches_recorded_run"] is False
    assert result["tda"]["per_seed"]["0"]["n_label_mismatches"] == 2


def test_clipzs_outputs_must_match_too(store, tmp_path):
    ckpts = tmp_path / "ckpts"
    path = _record_run(ckpts, "clipzs", seed=0)
    saved = torch.load(path, weights_only=False)
    saved["output"][5, 0] += 0.5  # beyond fp16 rounding
    torch.save(saved, path)
    result = check_dataset(store, "DTD", ckpts, [0], log=lambda *_: None)
    assert result["clipzs"]["matching_seed"] == 0  # labels alone still agree
    assert result["clipzs"]["clipzs_output"]["within_fp16_tolerance"] is False
    assert result["stream_order_matches_recorded_run"] is False


def test_no_recorded_run_gives_no_verdict(store, tmp_path):
    result = check_dataset(store, "DTD", tmp_path / "nothing", [0], log=lambda *_: None)
    assert result["stream_order_matches_recorded_run"] is None


def test_the_extractor_records_the_check_in_meta(tmp_path):
    ckpts = tmp_path / "ckpts"
    _record_run(ckpts, "clipzs", seed=2)
    root = tmp_path / "features"

    def post_check(out_dir, seeds):
        return check_dataset(FeatureStore(root, "dtd"), "DTD", ckpts, seeds, log=lambda *_: None)

    meta = extract_dataset(root, "dtd", build_split=_build_split, build_model=consume_rng_like_model_init,
                           device="cpu", autocast=False, log=lambda *_: None, splits=("test",),
                           post_check=post_check)
    assert meta["recorded_run_check"]["stream_order_matches_recorded_run"] is True
    assert meta["recorded_run_check"]["clipzs"]["matching_seed"] == 2


def test_the_cpu_script_exit_status(store, tmp_path):
    from analysis.check_stream_order import main

    ckpts = tmp_path / "ckpts"
    features = str(tmp_path / "features")
    assert main(["--features", features, "--ckpts", str(ckpts)]) == 1  # nothing to check
    _record_run(ckpts, "clipzs", seed=1)
    assert main(["--features", features, "--ckpts", str(ckpts), "--autocast", "off"]) == 0
