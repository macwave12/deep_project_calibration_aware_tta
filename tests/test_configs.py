# tests/test_configs.py
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from online_method.cal_tda import CalTdaConfig
from online_method.tda import TDA

DATASETS = ["dtd", "flower102", "pets", "eurosat", "aircraft"]
METHOD_CONFIGS = ["tda", "cal_tda", "tda_equivalent"]

# Upstream's `--test_sets` keys (mixed-case) for each of our lowercase filenames.
# This mapping lives here (and in the runner scripts) because upstream's own
# `TDA.pos_params` dict is keyed by these exact strings.
DATASET_TO_UPSTREAM_KEY = {
    "dtd": "DTD",
    "flower102": "Flower102",
    "pets": "Pets",
    "eurosat": "eurosat",
    "aircraft": "Aircraft",
}


class _DummyModel:
    """Stands in for the CLIP model so `TDA` can be constructed without a GPU.

    `prepare_model_and_optimization` only ever calls `.eval()` on `model`.
    """

    def eval(self):
        pass


def _upstream_tda_instance():
    """A `TDA` instance built without touching CUDA or loading a real model."""
    return TDA(model=_DummyModel(), device=None)


@pytest.mark.parametrize("name", DATASETS)
def test_every_dataset_has_a_config(name):
    assert Path(f"configs/{name}.yaml").exists()


@pytest.mark.parametrize("name", METHOD_CONFIGS)
def test_method_configs_construct_a_valid_config_object(name):
    data = yaml.safe_load(Path(f"configs/method/{name}.yaml").read_text())
    config = CalTdaConfig(**data)  # raises TypeError on any unknown/renamed key
    assert isinstance(config.as_dict(), dict)


def test_tda_equivalent_config_disables_every_contribution():
    data = yaml.safe_load(Path("configs/method/tda_equivalent.yaml").read_text())
    config = CalTdaConfig(**data)
    assert config.use_margin_admission is False
    assert config.use_prob_fusion is False
    assert config.use_loo_temperature is False


def test_cal_tda_config_enables_every_contribution():
    data = yaml.safe_load(Path("configs/method/cal_tda.yaml").read_text())
    config = CalTdaConfig(**data)
    assert config.use_margin_admission is True
    assert config.use_prob_fusion is True
    assert config.use_loo_temperature is True


@pytest.mark.parametrize("name,upstream_key", sorted(DATASET_TO_UPSTREAM_KEY.items()))
def test_dataset_config_matches_upstream_tda_hyperparameters(name, upstream_key):
    """Our per-dataset YAML must mirror upstream's `TDA.pos_params`/neg values exactly.

    The runner reads alpha/beta from `TDA.pos_params` at runtime
    (`online_method/tda.py:69-70`), not from these YAML files — the files are
    verified documentation, not the source of truth. Nothing else would catch
    it if they drifted apart, which is what this test is for. Values are read
    off the real upstream code path (not re-typed as literals here), so a
    future change to `tda.py`'s table would fail this test rather than pass
    silently.
    """
    data = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())

    upstream = _upstream_tda_instance()
    upstream.prepare_model_and_optimization(SimpleNamespace(test_sets=upstream_key))

    assert data["positive"]["alpha"] == upstream.pos_alpha
    assert data["positive"]["beta"] == upstream.pos_beta
    assert data["negative"]["alpha"] == upstream.neg_alpha
    assert data["negative"]["beta"] == upstream.neg_beta
