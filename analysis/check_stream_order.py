"""CPU check: do the extracted stream orders reproduce the recorded GPU runs exactly?

For every dataset in the feature folder, pairs `perm_test_seed<k>.pt` with the
newest `online_results/ckps/{clipzs,tda}/**/<TestSet>/save_results.pt`
(`online_tta.py` writes it, per-step `output` and `target` in stream order):

* `labels_test[perm] == target` elementwise (clipzs and tda), and
* clipzs logits recomputed from the cached features (with CUDA-autocast
  emulation) equal the recorded `output` within fp16 rounding.

Same code as the extractor's in-job check (`utils/recorded_run_check.py`).
Exit status 1 if any dataset does not match, or nothing could be checked.

    python analysis/check_stream_order.py [--features features] [--ckpts online_results/ckps]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", default="features")
    p.add_argument("--ckpts", default="online_results/ckps")
    p.add_argument("--autocast", default="emulate", choices=["emulate", "off"])
    a = p.parse_args(argv)

    import torch

    from utils.feature_cache import UPSTREAM_SET_ID, FeatureStore
    from utils.feature_replay import cuda_autocast_emulation
    from utils.recorded_run_check import check_dataset

    verdicts = {}
    for folder in sorted(Path(a.features).iterdir()):
        if not (folder / "meta.json").exists() or folder.name not in UPSTREAM_SET_ID:
            continue
        store = FeatureStore(a.features, folder.name)
        seeds = sorted(int(n[len("perm_test_seed"):-3]) for n in store.meta().get("permutations", [])
                       if n.startswith("perm_test_seed"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
            with cuda_autocast_emulation(a.autocast == "emulate"):
                autocast = (lambda: torch.cuda.amp.autocast()) if a.autocast == "emulate" else None
                result = check_dataset(store, UPSTREAM_SET_ID[folder.name], a.ckpts, seeds, autocast=autocast)
        verdicts[folder.name] = result["stream_order_matches_recorded_run"]
        for algorithm in ("clipzs", "tda"):
            r = result.get(algorithm)
            if r:
                print(f"    {algorithm}: {json.dumps({k: v for k, v in r.items() if k != 'per_seed'})}")
    checked = {k: v for k, v in verdicts.items() if v is not None}
    print("\nstream order matches recorded run: " + ", ".join(f"{k}={v}" for k, v in verdicts.items()))
    return 0 if checked and all(checked.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
