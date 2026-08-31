import os

from data.datautils import ID_to_DIRNAME

# build_dataset (data/datautils.py) has a case-sensitive membership check against
# upstream's `fewshot_datasets` list, whose entries are mixed-case. This maps our
# lowercase keys to the exact value build_dataset expects.
UPSTREAM_SET_ID = {
    "dtd": "DTD",
    "flower102": "Flower102",
    "pets": "Pets",
    "eurosat": "eurosat",
    "aircraft": "Aircraft",
}


def build_test_dataset(name, data_root):
    """Return the test-split dataset object for `name`, using upstream loaders.

    `name` is our lowercase key: dtd | flower102 | pets | eurosat | aircraft.

    Resolves the per-dataset directory through `ID_to_DIRNAME` -- the same dict
    `build_dataset` in data/datautils.py uses -- rather than a second, hand-maintained
    mapping, so this helper and the production entry point can't silently disagree
    about the on-disk layout.
    """
    from data.fewshot_datasets import build_fewshot_dataset

    root = os.path.join(data_root, ID_to_DIRNAME[name])
    return build_fewshot_dataset(name, root, transform=None, mode="test")
