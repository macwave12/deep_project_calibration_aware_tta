"""T15 fresh evaluation sets: specs, class names, subsets, ImageNet-val layout."""

import pytest

import scripts.download_datasets as dd
from scripts.extract_features import ALL_DATASETS, FRESH_DATASETS, class_names
from scripts.extract_features import parse_args as extract_args
from scripts.fetch_imagenet_val_hf import label_dir, write_rows
from scripts.make_imagefolder_subset import class_files, make_subset
from utils.feature_cache import FRESH_DATASETS as CACHE_FRESH, UPSTREAM_SET_ID


def test_fresh_sets_are_opt_in_and_known_everywhere():
    assert set(FRESH_DATASETS) == set(CACHE_FRESH)
    assert all(d in UPSTREAM_SET_ID for d in FRESH_DATASETS)
    assert set(dd.FRESH_SOURCES) == set(FRESH_DATASETS) - {"imagenet"}
    assert dd.parse_args(["--data-root", "x"]).datasets == list(dd.SOURCES)
    assert not set(extract_args([]).datasets) & set(FRESH_DATASETS)
    assert set(FRESH_DATASETS) <= set(ALL_DATASETS)


def test_fresh_folders_match_the_loaders():
    from data.datautils import ID_to_DIRNAME
    for key, spec in dd.FRESH_SOURCES.items():
        assert ID_to_DIRNAME[UPSTREAM_SET_ID[key]] == f"{spec.folder}/{spec.image_dir}"
    assert ID_to_DIRNAME["I"] == "imagenet/images"  # + "/val", as fetch_imagenet_val_hf writes


def test_class_names_for_imagenet_variants():
    from data.imagnet_prompts import imagenet_classes
    from data.imagenet_variants import all_wnids, imagenet_r_wnids
    a, r, k, i = class_names("A"), class_names("R"), class_names("K"), class_names("I")
    assert len(a) == len(r) == 200 and len(k) == len(i) == 1000
    # ImageFolder sorts wnid folders, which is ImageNet index order: R's names follow it.
    assert all_wnids == sorted(all_wnids)
    assert r == [imagenet_classes[j] for j, w in enumerate(all_wnids) if w in imagenet_r_wnids]


def test_imagefolder_count_ignores_top_level_files(tmp_path):
    spec = dd.FRESH_SOURCES["imagenet_a"]
    root = tmp_path / spec.folder / spec.image_dir
    (root / "n01498041").mkdir(parents=True)
    (root / "n01498041" / "a.jpg").write_bytes(b"x")
    (root / "README.txt").write_text("not an image")
    entries = dd._split_entries(spec, str(tmp_path / spec.folder))["test"]
    assert [e.replace("\\", "/") for e in entries] == ["imagenet-a/n01498041/a.jpg"]


def test_imagefolder_subset_is_reproducible_and_reversible(tmp_path):
    def build(base):
        for c in ("n1", "n2", "n3"):
            (base / "imgs" / c).mkdir(parents=True)
            for j in range(10):
                (base / "imgs" / c / f"{j}.jpg").write_bytes(b"x")
    a, b = tmp_path / "a", tmp_path / "b"
    build(a)
    build(b)
    info = make_subset(a, "imgs", 7)
    make_subset(b, "imgs", 7)
    assert info["total"] == 30 and class_files(a / "imgs") == class_files(b / "imgs")
    assert len(class_files(a / "imgs")) == 7 and len(class_files(a / "_excluded")) == 23
    assert make_subset(a, "imgs", 7) == info  # idempotent
    with pytest.raises(ValueError):
        make_subset(a, "imgs", 5)


def test_imagenet_val_rows_land_in_sorted_label_folders(tmp_path):
    nxt = write_rows([(b"a", 0), (b"b", 999), (b"c", 10)], tmp_path, start=0)
    assert nxt == 3
    assert (label_dir(tmp_path, 999) / "00001.JPEG").read_bytes() == b"b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0000", "0010", "0999"]
