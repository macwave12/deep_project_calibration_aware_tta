import os
import pytest

# The one place the plan's expected counts are encoded.
EXPECTED_TEST_COUNTS = {
    "dtd": 1692,
    "flower102": 2463,
    "pets": 3669,
    "eurosat": 8100,
    "aircraft": 3333,
}

DATA_ROOT = os.environ.get("DATA_ROOT")

pytestmark = pytest.mark.skipif(
    not DATA_ROOT, reason="set DATA_ROOT to the dataset root to run split checks"
)


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_TEST_COUNTS.items()))
def test_test_split_has_expected_number_of_samples(name, expected):
    from tests.helpers import build_test_dataset  # thin wrapper over upstream loaders

    dataset = build_test_dataset(name, DATA_ROOT)
    assert len(dataset) == expected


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_TEST_COUNTS.items()))
def test_production_entry_point_resolves_the_same_split(name, expected):
    """build_dataset is what online_tta.py actually calls; it must resolve too."""
    from data.datautils import build_dataset
    from tests.helpers import UPSTREAM_SET_ID

    dataset = build_dataset(
        UPSTREAM_SET_ID[name], None, DATA_ROOT, mode="test"
    )
    assert len(dataset) == expected
