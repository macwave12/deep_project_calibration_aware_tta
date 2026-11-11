import pytest
import torch

from scripts.make_permutations import cpu_permutation, write_permutations


def _store(tmp_path, n=50):
    d = tmp_path / "dtd"
    d.mkdir()
    torch.save({"labels": torch.zeros(n, dtype=torch.int64)}, d / "labels_test.pt")
    return tmp_path


def test_permutations_are_reproducible_and_valid(tmp_path):
    root = _store(tmp_path)
    paths = write_permutations(root, "dtd", "test", [3, 4])
    p3 = torch.load(paths[0], weights_only=True)
    assert torch.equal(p3, cpu_permutation(50, 3))
    assert torch.equal(p3.sort().values, torch.arange(50))
    assert not torch.equal(p3, torch.load(paths[1], weights_only=True))


def test_existing_orders_are_kept_and_extracted_seeds_refused(tmp_path):
    root = _store(tmp_path)
    keep = torch.arange(50).flip(0)
    torch.save(keep, root / "dtd" / "perm_test_seed5.pt")
    assert write_permutations(root, "dtd", "test", [5]) == []
    assert torch.equal(torch.load(root / "dtd" / "perm_test_seed5.pt", weights_only=True), keep)
    with pytest.raises(ValueError, match="extracted"):
        write_permutations(root, "dtd", "test", [1])
