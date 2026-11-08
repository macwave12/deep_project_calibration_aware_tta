"""T04 joiner: cached features map back to dataset indices; labels join there."""

import numpy as np
import pytest
import torch

from analysis.trace_join import (
    AMBIGUOUS,
    UNMATCHED,
    FeatureIndex,
    join_snapshots,
    join_steps,
    match_indices,
    snapshot_summary,
)
from online_method.cal_tda import CalTDA, CalTdaConfig
from online_method.cal_tda_trace import TraceRecorder
from tests.cpu_stream import FakeCLIP, cpu_only, default_args, drive, feature_stream  # noqa: F401


def test_exact_match_unique_missing_and_duplicate_rows():
    data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float16)
    index = FeatureIndex(data)
    query = torch.tensor([[0.0, 1.0], [0.5, 0.5], [0.25, 0.75], [1.0, 0.0]], dtype=torch.float16)
    assert index.candidates(query) == [[1], [2, 3], [], [0]]
    # Not "close": exact. One ULP off is no match.
    nudged = data[:1].clone()
    nudged[0, 0] = 1.0 + 2.0 ** -10  # next fp16 value above 1.0
    assert nudged[0, 0] != data[0, 0]
    assert index.candidates(nudged) == [[]]
    resolved = match_indices(index.candidates(query))
    assert resolved.tolist() == [1, AMBIGUOUS, UNMATCHED, 0]
    # The permutation resolves the duplicate: step 7 was dataset index 3.
    perm = np.arange(10)
    perm[7] = 3
    assert match_indices(index.candidates(query), np.array([1, 7, -1, 0]), perm).tolist() == [1, 3, UNMATCHED, 0]


def test_dtype_mismatch_is_an_error_not_a_silent_miss():
    index = FeatureIndex(torch.zeros(3, 4, dtype=torch.float16))
    with pytest.raises(ValueError, match="dtype"):
        index.candidates(torch.zeros(1, 4))


def test_joiner_on_a_shuffled_synthetic_stream(cpu_only):
    # Dataset in its own order, streamed through a permutation, as the real
    # loader does (shuffle=True). Labels never go near the method.
    n, c = 120, 10
    features, text, labels = feature_stream(n_samples=n, n_classes=c, seed=21, spread=0.9)
    features = features.half()  # the GPU path's dtype; matching must be exact in it
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3))
    runner = CalTDA(FakeCLIP(features[perm].float(), text), None,
                    config=CalTdaConfig(use_margin_admission=False, use_prob_fusion=False))
    runner.prepare_model_and_optimization(default_args("DTD"))
    runner.trace = TraceRecorder(snapshot_every=40)
    drive(runner, n, default_args("DTD"))
    trace = runner.trace_dict()

    # Cached features are fp32 here; the dataset file is what the model
    # returned, so the analysis compares in that dtype.
    for snap in trace["snapshots"]:
        for which in ("pos", "neg"):
            snap[which]["feature"] = snap[which]["feature"].half()

    joined = join_snapshots(trace, features, labels, perm)
    assert len(joined) > 0
    assert (joined["dataset_index"] >= 0).all()
    assert joined["perm_consistent"].all()
    # Ground truth for the join, computed independently from the stream order.
    expected = perm[torch.as_tensor(joined["inserted_step"].to_numpy())].numpy()
    assert (joined["dataset_index"].to_numpy() == expected).all()
    assert (joined["label"].to_numpy() == labels.numpy()[expected]).all()

    summary = snapshot_summary(joined, c)
    assert summary["step"].unique().tolist() == [39, 79, 119]
    final_pos = summary[(summary["step"] == 119) & (summary["cache"] == "pos")].iloc[0]
    items = [(k, it[0]) for k, v in runner.pos_cache.items() for it in v]
    assert final_pos["n_items"] == len(items)
    # Purity from the joiner equals purity from a direct lookup.
    row_of = {features[perm][i].float().numpy().tobytes(): int(perm[i]) for i in range(n)}
    direct = np.mean([labels[row_of[f.reshape(-1).half().float().numpy().tobytes()]] == k for k, f in items])
    assert final_pos["purity"] == pytest.approx(direct)
    assert 0 < final_pos["purity"] < 1  # the synthetic cache does hold mistakes
    assert final_pos["coverage_correct"] <= final_pos["classes_present"] <= 1

    steps = join_steps(trace, labels, perm)
    assert len(steps) == n
    assert (steps["dataset_index"].to_numpy() == perm.numpy()).all()
    assert (steps["label"].to_numpy() == labels[perm].numpy()).all()
    assert steps["correct"].mean() == pytest.approx(
        float((trace["steps"]["out_argmax"].long() == labels[perm]).float().mean())
    )
