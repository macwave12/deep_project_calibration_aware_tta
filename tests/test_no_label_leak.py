"""The project's central methodological guarantee, enforced automatically.

Test-time adaptation is only meaningful if the method never sees a label.
Labels enter exactly one module: utils/evaluation.py, at the very end of a run.
"""

import inspect
import re
from pathlib import Path

import pytest

from online_method.cal_tda import CalTDA

METHOD_FILES = [
    Path("online_method/cal_tda.py"),
    Path("utils/loo_temperature.py"),
]

# `pseudo_labels` are assigned by the method itself and are explicitly allowed.
# We search for patterns that indicate label leaks:
# - Variable names in forbidden list (gt_label, ground_truth, target_label, true_label)
# - Parameter pattern for 'target' in function signatures
FORBIDDEN_NAME = re.compile(r"\b(gt_label|ground_truth|target_label|true_label)\b")
FORBIDDEN_TARGET_PARAM = re.compile(r"def \w+\([^)]*\btarget\b[^)]*\)")


@pytest.mark.parametrize("path", METHOD_FILES, ids=lambda p: str(p))
def test_method_code_never_names_a_ground_truth_label(path):
    source = path.read_text(encoding="utf-8")
    assert not FORBIDDEN_NAME.search(source), f"{path} uses forbidden label variable names"
    assert not FORBIDDEN_TARGET_PARAM.search(source), f"{path} accepts 'target' parameter (upstream label name)"


def test_only_the_scorer_computes_correctness():
    """`labels ==` style comparisons belong in calibration/evaluation, not methods."""
    source = Path("online_method/cal_tda.py").read_text(encoding="utf-8")
    assert "compute_accuracy" not in source
    assert "compute_ece" not in source


def test_the_scorer_is_the_documented_single_entry_point():
    source = Path("utils/evaluation.py").read_text(encoding="utf-8")
    assert "def score_and_save" in source


def test_caltda_entry_points_never_receive_labels():
    """Structural check: CalTDA's public methods never accept a label parameter.

    A regex can be evaded by renaming; this check defeats that by inspecting the
    actual signatures of the methods that touch samples.
    """
    forbidden_param_names = {"target", "labels", "label", "gt", "ground_truth", "true_label"}

    entry_points = [
        CalTDA.__init__,
        CalTDA.prepare_model_and_optimization,
        CalTDA.adaptation_process,
    ]

    for method in entry_points:
        sig = inspect.signature(method)
        param_names = set(sig.parameters.keys())
        leak_params = param_names & forbidden_param_names
        assert not leak_params, (
            f"{method.__qualname__} accepts forbidden parameter(s): {leak_params}. "
            "The method must never receive ground truth."
        )
