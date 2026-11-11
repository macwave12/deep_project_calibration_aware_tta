#!/usr/bin/env bash
# T14 dev-set confirmation: TDA (full trace), agreement gate and CLIP, P2 and P1, seeds 0/1/2.
#   bash scripts/replay_dev.sh caltech101 ucf101 ...   -> replay_results/t14/
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
for ds in "$@"; do
  for proto in clean published; do
    s=""; [ "$proto" = clean ] && s="_clean"
    "$PY" scripts/replay.py --dataset "$ds" --method cal_tda --config configs/method/tda_equivalent.yaml \
      --run-name "tda_eq$s" --protocol "$proto" --seed 0 1 2 --trace full --out-dir replay_results/t14
    "$PY" scripts/replay.py --dataset "$ds" --method cal_tda --config configs/method/cal_tda_agree.yaml \
      --run-name "cal_tda_agree$s" --protocol "$proto" --seed 0 1 2 --out-dir replay_results/t14
  done
  "$PY" scripts/replay.py --dataset "$ds" --method clipzs --seed 0 1 2 --out-dir replay_results/t14
done
