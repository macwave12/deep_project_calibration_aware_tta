"""T14 development datasets: opt-in specs, ImageNetV2 labels, nested archives."""

import io
import os
import tarfile
import zipfile

import pytest
from PIL import Image

import scripts.download_datasets as dd
from data.datautils import imagenetv2_numeric_labels
from scripts.extract_features import ALL_DATASETS, DEV_DATASETS, class_names
from scripts.extract_features import parse_args as extract_args
from utils.feature_cache import DEV_DATASETS as CACHE_DEV, UPSTREAM_SET_ID


def test_dev_sets_are_opt_in_everywhere():
    assert set(dd.DEV_SOURCES) == set(DEV_DATASETS) == set(CACHE_DEV)
    assert not set(dd.DEV_SOURCES) & set(dd.SOURCES)
    a = dd.parse_args(["--data-root", "x"])
    assert a.datasets == list(dd.SOURCES)
    assert dd.parse_args(["--data-root", "x", "--datasets", "ucf101"]).datasets == ["ucf101"]
    assert extract_args([]).datasets == [d for d in ALL_DATASETS if d not in DEV_DATASETS]
    assert all(d in UPSTREAM_SET_ID for d in DEV_DATASETS)


def test_dev_folders_match_the_loaders():
    from data.datautils import ID_to_DIRNAME
    for key, spec in dd.DEV_SOURCES.items():
        set_id = UPSTREAM_SET_ID[key]
        dirname = ID_to_DIRNAME["V" if set_id == "V" else set_id.lower()]
        assert dirname.split("/")[0] == spec.folder


def test_class_names_for_dev_sets():
    assert len(class_names("V")) == 1000
    assert class_names("Caltech101")[:2] == ["face", "leopard"]
    assert len(class_names("UCF101")) == 101 and len(class_names("Food101")) == 101


def test_imagenetv2_labels_follow_folder_numbers(tmp_path):
    from torchvision import datasets
    for name in ["0", "1", "2", "10", "100"]:
        (tmp_path / name).mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / name / "a.jpeg")
    ds = datasets.ImageFolder(str(tmp_path))
    assert ds.classes == ["0", "1", "10", "100", "2"]  # the string sort that breaks labels
    imagenetv2_numeric_labels(ds)
    got = {os.path.basename(os.path.dirname(p)): t for p, t in ds.samples}
    assert got == {"0": 0, "1": 1, "2": 2, "10": 10, "100": 100}
    assert ds.targets == [t for _, t in ds.samples]


def test_nested_archive_is_unpacked_before_moves(tmp_path):
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w:gz") as tf:
        data = b"img"
        info = tarfile.TarInfo("101_ObjectCategories/ant/image_0001.jpg")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    outer = tmp_path / "caltech-101.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("caltech-101/101_ObjectCategories.tar.gz", inner.getvalue())
    arch = dd.DEV_SOURCES["caltech101"].archives[0]
    ds_dir = tmp_path / "caltech-101"
    dd.layout_archive(arch, str(outer), str(ds_dir), str(tmp_path / "_staging"))
    assert (ds_dir / "101_ObjectCategories" / "ant" / "image_0001.jpg").read_bytes() == b"img"


def test_imagefolder_kind_counts_test_images(tmp_path):
    spec = dd.DEV_SOURCES["imagenetv2"]
    root = tmp_path / spec.folder / spec.image_dir
    for c in range(3):
        (root / str(c)).mkdir(parents=True)
        for i in range(2):
            (root / str(c) / f"{i}.jpeg").write_bytes(b"x")
    entries = dd._split_entries(spec, str(tmp_path / spec.folder))
    assert len(entries["test"]) == 6 and "val" not in entries


def test_food101_subset_is_deterministic_and_keeps_train_val():
    from scripts.make_food101_dev_subset import subset
    splits = {"train": [["t", 0, "a"]], "val": [["v", 0, "a"]],
              "test": [[f"x/{i}.jpg", i % 3, "c"] for i in range(50)]}
    a, b = subset(splits, seed=0, size=10), subset(splits, seed=0, size=10)
    assert a == b and len(a["test"]) == 10 and a["train"] == splits["train"]
    assert [e[0] for e in a["test"]] == sorted(e[0] for e in a["test"])
    assert subset(splits, seed=1, size=10) != a
