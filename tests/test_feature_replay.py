"""T02 acceptance on synthetic data: replay == running the method directly, bit for bit."""

import json
import warnings
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from online_method.cal_tda import CalTDA, CalTdaConfig
from online_method.clipzs import CLIPZS
from online_method.tda import TDA
from online_tta import to_probabilities
from tests.replay_stubs import FakeImageDataset, StubCLIP, consume_rng_like_model_init
from utils.evaluation import score_and_save
from utils.feature_extraction import extract_dataset
from utils.feature_replay import (
    FeatureReplayModel,
    build_trainer,
    cuda_autocast_emulation,
    run_stream,
)
from utils.replay_runner import record_name, replay_one
from utils.tools import set_random_seed

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")


class _SequentialCLIP:
    """Direct path: hands out features one after the other, like test_cal_tda_runner._FakeCLIP."""

    def __init__(self, features, text, logit_scale):
        self.features, self.text, self.scale, self.i = features, text, logit_scale, 0

    def eval(self):
        return self

    def forward_features(self, images):
        f = self.features[self.i:self.i + 1].clone()
        self.i += 1
        return f, self.text.clone(), self.scale.clone()

    def __call__(self, images):
        f, t, s = self.forward_features(images)
        return s * f @ t.t()


def _random_store(n=24, n_classes=5, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    protos = F.normalize(torch.randn(n_classes, dim, generator=g), dim=-1)
    labels = torch.randint(0, n_classes, (n,), generator=g)
    feats = F.normalize(protos[labels] + 0.6 * torch.randn(n, dim, generator=g), dim=-1)
    text = F.normalize(protos + 0.3 * torch.randn(n_classes, dim, generator=g), dim=-1)
    order = torch.randperm(n, generator=g)
    return feats, text, torch.tensor(100.0), order


def _direct(algorithm, config, feats, text, scale, order, args):
    model = _SequentialCLIP(feats[order], text, scale)
    trainer = {"clipzs": lambda: CLIPZS(model, 0), "tda": lambda: TDA(model, 0),
               "cal_tda": lambda: CalTDA(model, 0, config=config)}[algorithm]()
    trainer.prepare_model_and_optimization(args)
    dummy = torch.zeros(1, 3, 4, 4)
    return [trainer.adaptation_process(None, dummy, args)["output"] for _ in range(len(order))]


CASES = [
    ("clipzs", None),
    ("tda", None),
    ("cal_tda", CalTdaConfig(use_margin_admission=False, use_prob_fusion=True, fusion_gate=True,
                             use_loo_temperature=False)),
    ("cal_tda", CalTdaConfig(use_margin_admission=True, margin_threshold=1.0, use_loo_temperature=False)),
]
# The LOO temperature is O(N^2) on CPU (K18) and runs outside every autocast
# block, so it is checked once, without emulation, to keep the suite fast.
LOO_CASE = ("cal_tda", CalTdaConfig(use_margin_admission=False, use_prob_fusion=False,
                                    use_loo_temperature=True))
PARAMS = [(a, c, e) for a, c in CASES for e in (False, True)] + [(*LOO_CASE, False)]


@pytest.mark.parametrize("algorithm,config,emulate", PARAMS,
                         ids=[f"{a}{i}-{'cuda-autocast' if e else 'fp32'}" for i, (a, _, e) in enumerate(PARAMS)])
def test_replay_is_bit_identical_to_running_the_method_directly(algorithm, config, emulate):
    feats, text, scale, order = _random_store()
    args = SimpleNamespace(test_sets="DTD", protocol="published")
    with cuda_autocast_emulation(emulate):
        direct = _direct(algorithm, config and CalTdaConfig(**config.as_dict()), feats, text, scale, order, args)
        model = FeatureReplayModel(feats, text, scale)
        trainer = build_trainer(algorithm, model, config and CalTdaConfig(**config.as_dict()))
        trainer.prepare_model_and_optimization(args)
        replayed = run_stream(trainer, model, order, args)
    assert len(direct) == len(replayed) == len(order)
    for a, b in zip(direct, replayed):
        assert a.dtype == b.dtype
        assert torch.equal(a, b)


def test_emulated_autocast_gives_tda_cuda_dtypes():
    # Under CUDA autocast TDA's logits are fp16 and its output accumulates in fp16
    # (see test_cal_tda_runner's tolerance note); without emulation both are fp32.
    feats, text, scale, order = _random_store(n=6)
    args = SimpleNamespace(test_sets="DTD")
    for emulate, dtype in ((True, torch.float16), (False, torch.float32)):
        model = FeatureReplayModel(feats, text, scale)
        trainer = build_trainer("tda", model, {})
        trainer.prepare_model_and_optimization(args)
        with cuda_autocast_emulation(emulate):
            outs = run_stream(trainer, model, order, args)
        assert {o.dtype for o in outs} == {dtype}
    assert torch.cuda.amp.autocast.__name__ != "_emulated_autocast"  # restored


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_emulation_matches_real_cuda_autocast_dtypes():
    s, f, t = torch.tensor(100.0), torch.randn(1, 64), torch.randn(10, 64)
    with torch.cuda.amp.autocast():
        real = s.cuda() * f.cuda() @ t.cuda().t()
        real_sm = real.softmax(-1)
    with cuda_autocast_emulation():
        with torch.cuda.amp.autocast():
            emu = s * f @ t.t()
            emu_sm = emu.softmax(-1)
    assert (real.dtype, real_sm.dtype) == (emu.dtype, emu_sm.dtype)


def test_the_model_refuses_to_serve_before_seek():
    feats, text, scale, _ = _random_store(n=3)
    with pytest.raises(RuntimeError, match="seek"):
        FeatureReplayModel(feats, text, scale).forward_features(None)


# --- end to end: T01 extraction -> T02 replay -> record ---------------------

def _build_split(name):
    return FakeImageDataset(n={"test": 40, "val": 16, "train16": 10}[name],
                            seed={"test": 7, "val": 8, "train16": 9}[name])


def _extract(root, emulate):
    # Extraction runs under the same autocast as the stream it is compared with:
    # on the GPU both are real CUDA autocast; here both are the emulation.
    with cuda_autocast_emulation(emulate):
        extract_dataset(root, "dtd", build_split=_build_split, build_model=consume_rng_like_model_init,
                        device="cpu", autocast=emulate, log=lambda *_: None,
                        meta_extra={"arch": "ViT-B/16"})
    return root


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    return _extract(tmp_path_factory.mktemp("features"), emulate=False)


@pytest.fixture(scope="module")
def autocast_store_root(tmp_path_factory):
    return _extract(tmp_path_factory.mktemp("features_autocast"), emulate=True)


def _runner_loop(algorithm, config, seed, emulate, out_dir, protocol="published"):
    """online_tta.py's main(), on CPU, with a StubCLIP that sees real image tensors."""
    args = SimpleNamespace(test_sets="DTD", protocol=protocol, batch_size=1, loader_len=40,
                           algorithm=algorithm)
    set_random_seed(seed)
    dataset = _build_split("test")
    model = consume_rng_like_model_init()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    trainer = build_trainer(algorithm, model, config)
    trainer.prepare_model_and_optimization(args)
    probs, labels = [], []
    with cuda_autocast_emulation(emulate):
        for images, target in loader:
            out = trainer.adaptation_process(None, images, args)["output"]
            probs.append(to_probabilities(algorithm, out))
            labels.append(target.reshape(-1))
    probs, labels = torch.cat(probs), torch.cat(labels)
    path = score_and_save(probs.numpy(), labels.numpy(), dataset="dtd", method="reference",
                          backbone="ViT-B/16", seed=seed, hyperparams={}, out_dir=out_dir)
    return probs, json.loads(path.read_text())


@pytest.mark.parametrize("algorithm,config_file,seed,emulate", [
    ("tda", "configs/method/tda.yaml", 0, False),
    ("tda", "configs/method/tda.yaml", 1, True),
    ("clipzs", None, 2, True),
    ("cal_tda", "configs/method/cal_tda_gate.yaml", 0, True),
])
def test_replay_reproduces_the_runner_on_the_extracted_features(
        store_root, autocast_store_root, tmp_path, algorithm, config_file, seed, emulate):
    from online_tta import load_method_config

    root = autocast_store_root if emulate else store_root

    config = load_method_config(SimpleNamespace(algorithm=algorithm, config=config_file))
    ref_probs, ref = _runner_loop(algorithm, config, seed, emulate, tmp_path / "ref")
    path = replay_one(root, "dtd", algorithm=algorithm, seed=seed, config=config_file,
                      out_dir=tmp_path / "replay", autocast="emulate" if emulate else "off",
                      log=lambda *_: None)
    record = json.loads(path.read_text())
    assert record["accuracy"] == ref["accuracy"]
    assert record["ece"] == ref["ece"]
    assert record["reliability"] == ref["reliability"]
    assert record["n_samples"] == 40 and record["seed"] == seed
    assert record["hyperparams"]["replay"]["full_stream"] is True
    if algorithm != "clipzs":
        assert record["hyperparams"]["protocol"] == "published"


def test_the_stream_never_reads_labels(store_root, tmp_path):
    # Copy the store without its labels: the method side must still run to the
    # end; only the scorer, after the stream, needs (and misses) them.
    import shutil

    root = tmp_path / "nolabels"
    shutil.copytree(store_root, root)
    for p in (root / "dtd").glob("labels_*.pt"):
        p.unlink()
    from utils.feature_cache import FeatureStore

    store = FeatureStore(root, "dtd")
    model = FeatureReplayModel.from_text_dict(store.image_features("test"), store.text())
    trainer = build_trainer("tda", model, {})
    args = SimpleNamespace(test_sets="DTD")
    trainer.prepare_model_and_optimization(args)
    outs = run_stream(trainer, model, store.permutation("test", 0), args)
    assert len(outs) == 40
    with pytest.raises(FileNotFoundError):
        replay_one(root, "dtd", algorithm="tda", out_dir=tmp_path / "o", log=lambda *_: None)


def test_clean_protocol_is_recorded_and_named_apart(store_root, tmp_path):
    path = replay_one(store_root, "dtd", algorithm="tda", protocol="clean",
                      out_dir=tmp_path, log=lambda *_: None)
    assert path.name == "dtd_tda_clean.json"
    params = json.loads(path.read_text())["hyperparams"]
    assert (params["protocol"], params["alpha"], params["beta"]) == ("clean", 2.0, 5.0)
    assert record_name("tda", None, "published") == "tda"
    assert record_name("tda", "mine", "clean") == "mine"


def test_val_split_and_other_seeds_get_their_own_folders(store_root, tmp_path):
    val = replay_one(store_root, "dtd", algorithm="clipzs", split="val", out_dir=tmp_path,
                     log=lambda *_: None)
    seed1 = replay_one(store_root, "dtd", algorithm="clipzs", seed=1, out_dir=tmp_path,
                       log=lambda *_: None)
    assert val.relative_to(tmp_path).as_posix() == "val/dtd_clipzs.json"
    assert seed1.relative_to(tmp_path).as_posix() == "seed1/dtd_clipzs.json"
    assert json.loads(val.read_text())["n_samples"] == 16


@pytest.mark.parametrize("algorithm", ["dmn_weak", "ecalp", "onzeta"])
def test_unscored_upstream_methods_run_on_cpu_and_keep_raw_outputs(store_root, tmp_path, algorithm):
    path = replay_one(store_root, "dtd", algorithm=algorithm, out_dir=tmp_path, limit=12,
                      log=lambda *_: None)
    saved = torch.load(path, weights_only=False)
    assert path.name == f"dtd_{algorithm}_outputs.pt"
    assert saved["outputs"].shape[0] == 12
    assert torch.isfinite(saved["outputs"]).all()


def test_suites_name_existing_configs():
    from pathlib import Path

    from scripts.replay import parse_args, suite_runs

    for suite in ("headline", "ablation", "ablation_gate"):
        for _, config, _, _ in suite_runs(suite):
            assert config is None or Path(config).exists(), config
    names = [n or a for a, _, n, _ in suite_runs("headline")]
    assert names == ["clipzs", "tda", "cal_tda_gate_temp", "cal_tda_gate", "cal_tda_temp"]
    with pytest.raises(SystemExit):
        parse_args(["--method", "tda", "--suite", "headline"])


def test_compare_flags_records_outside_the_tolerance(tmp_path):
    from analysis.compare_replay import compare, main

    def write(path, acc, ece, n=100):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dataset": "dtd", "method": path.stem.split("_", 1)[1],
                                    "accuracy": acc, "ece": ece, "n_samples": n, "seed": 0}))

    write(tmp_path / "ref/dtd_tda.json", 0.4704, 0.1741)
    write(tmp_path / "ref/ablation/dtd_none.json", 0.4704, 0.1741)
    write(tmp_path / "rep/dtd_tda.json", 0.4720, 0.1765)          # +0.16 / +0.24: ok
    write(tmp_path / "rep/ablation/dtd_none.json", 0.4750, 0.1741)  # +0.46 acc: fails
    write(tmp_path / "rep/seed1/dtd_tda.json", 0.48, 0.17)          # no reference: informational
    rows = {r["record"]: r["status"] for r in compare(tmp_path / "rep", tmp_path / "ref")}
    assert rows == {"dtd_tda.json": "ok", "ablation/dtd_none.json": "OUT OF TOLERANCE",
                    "seed1/dtd_tda.json": "no reference"}
    assert main(["--replay", str(tmp_path / "rep"), "--reference", str(tmp_path / "ref")]) == 1
    (tmp_path / "rep/ablation/dtd_none.json").unlink()
    assert main(["--replay", str(tmp_path / "rep"), "--reference", str(tmp_path / "ref")]) == 0
