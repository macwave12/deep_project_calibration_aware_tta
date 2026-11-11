#!/usr/bin/env bash
# T16 fresh-set evaluation: CLIP, TDA (full trace) and the agreement gate, P2 only, seeds 0/1/2.
# ImageNet-Sketch (50,889 x 1000 classes) keeps no full trace (about 1.8 GB per seed); its
# label-free confidence signals come from a separate 500-sample full-trace run, as for ImageNet-val.
#   bash scripts/replay_fresh.sh stl10 cifar10 ...   -> replay_results/t16_fresh/
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
OUT="${OUT:-replay_results/t16_fresh}"
for ds in "$@"; do
  trace=full
  if [ "$ds" = imagenet_sketch ] || [ "$ds" = imagenet ]; then
    trace=off
    "$PY" scripts/replay.py --dataset "$ds" --method cal_tda --config configs/method/tda_equivalent.yaml \
      --run-name tda_eq_clean_head500 --protocol clean --seed 0 1 2 --trace full --limit 500 --out-dir "$OUT"
  fi
  "$PY" scripts/replay.py --dataset "$ds" --method clipzs --seed 0 1 2 --out-dir "$OUT"
  "$PY" scripts/replay.py --dataset "$ds" --method cal_tda --config configs/method/tda_equivalent.yaml \
    --run-name tda_eq_clean --protocol clean --seed 0 1 2 --trace "$trace" --out-dir "$OUT"
  "$PY" scripts/replay.py --dataset "$ds" --method cal_tda --config configs/method/cal_tda_agree.yaml \
    --run-name cal_tda_agree_clean --protocol clean --seed 0 1 2 --out-dir "$OUT"
done
