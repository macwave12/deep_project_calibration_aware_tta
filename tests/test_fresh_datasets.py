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
    from scripts.fetch_wds import WDS
    assert set(dd.FRESH_SOURCES) <= set(FRESH_DATASETS) and set(WDS) <= set(FRESH_DATASETS)
    # every fresh set has a source: downloader, CLIP_benchmark mirror, or the HF ImageNet script
    assert set(FRESH_DATASETS) == set(dd.FRESH_SOURCES) | set(WDS) | {"imagenet"}
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


# --- webdataset-based fresh sets (scripts/fetch_wds.py) ----------------------------------

def _wds_tar(samples):
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in samples:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def test_wds_tar_becomes_index_ordered_class_folders(tmp_path):
    from scripts.fetch_wds import convert_tar
    tar = _wds_tar([("s0000000.cls", b"2"), ("s0000000.webp", b"A"),
                    ("s0000001.png", b"B"), ("s0000001.cls", b"10"), ("s0000002.cls", b"0")])
    n = convert_tar(tar, tmp_path, "stl10")
    assert n == 2  # s0000002 has no image
    assert (tmp_path / "0002" / "s0000000.webp").read_bytes() == b"A"
    assert (tmp_path / "0010" / "s0000001.png").read_bytes() == b"B"


def test_wds_imagenet_r_uses_sorted_wnid_folders(tmp_path):
    from data.imagenet_variants import imagenet_r_wnids
    from scripts.fetch_wds import convert_tar
    convert_tar(_wds_tar([("a.cls", b"0"), ("a.jpg", b"x"), ("b.cls", b"199"), ("b.jpg", b"y")]),
                tmp_path, "imagenet_r")
    wnids = sorted(imagenet_r_wnids)
    assert (tmp_path / wnids[0] / "a.jpg").exists() and (tmp_path / wnids[199] / "b.jpg").exists()


def test_wds_sets_have_class_names_and_loader_paths():
    from data.datautils import ID_to_DIRNAME, WDS_SET_IDS
    from scripts.fetch_wds import WDS
    expected = {"STL10": 10, "CIFAR10": 10, "CIFAR100": 100, "VOC2007": 20, "GTSRB": 43,
                "StanfordCarsWDS": 196}
    for set_id, n in expected.items():
        assert len(class_names(set_id)) == n and set_id in WDS_SET_IDS
        key = next(k for k, v in UPSTREAM_SET_ID.items() if v == set_id)
        _, folder, image_dir, _ = WDS[key]
        assert ID_to_DIRNAME[set_id] == f"{folder}/{image_dir}"
    assert "_" not in " ".join(class_names("CIFAR100"))


def test_wds_loader_labels_follow_class_index(tmp_path):
    from PIL import Image
    from data.datautils import build_dataset
    root = tmp_path / "gtsrb" / "images"
    for c in (0, 2, 10):
        (root / f"{c:04d}").mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(root / f"{c:04d}" / "x.png")
    ds = build_dataset("GTSRB", None, str(tmp_path))
    # Labels are the folder numbers, even with classes missing from the split.
    assert sorted({t for _, t in ds.samples}) == [0, 2, 10] and ds.targets == [t for _, t in ds.samples]


def test_tda_clean_protocol_runs_on_datasets_without_published_values():
    from types import SimpleNamespace
    from online_method.tda import TDA
    from tests.cpu_stream import FakeCLIP, feature_stream
    feats, text, _ = feature_stream(n_samples=4, n_classes=3)
    clean = TDA(FakeCLIP(feats, text), None)
    clean.prepare_model_and_optimization(SimpleNamespace(test_sets="STL10", protocol="clean"))
    assert (clean.pos_alpha, clean.pos_beta) == (2.0, 5.0)
    published = TDA(FakeCLIP(feats, text), None)
    with pytest.raises(KeyError):
        published.prepare_model_and_optimization(SimpleNamespace(test_sets="STL10", protocol="published"))
