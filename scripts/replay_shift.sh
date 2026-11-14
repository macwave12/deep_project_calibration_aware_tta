#!/usr/bin/env bash
# T17 shifted-set evaluation (sets far from CLIP's web photos): CLIP, TDA (full trace) and the agreement
# gate, P2 only, seeds 0/1/2. Features from scripts/fetch_wds.py + scripts/extract_features.py.
#   bash scripts/replay_shift.sh fer2013 pcam ...   -> replay_results/t17_shift/
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
OUT="${OUT:-replay_results/t17_shift}"
for ds in "$@"; do
  "$PY" scripts/replay.py --dataset "$ds" --method clipzs --seed 0 1 2 --out-dir "$OUT"
  "$PY" scripts/replay.py --dataset "$ds" --method cal_tda --config configs/method/tda_equivalent.yaml \
    --run-name tda_eq_clean --protocol clean --seed 0 1 2 --trace full --out-dir "$OUT"
  "$PY" scripts/replay.py --dataset "$ds" --method cal_tda --config configs/method/cal_tda_agree.yaml \
    --run-name cal_tda_agree_clean --protocol clean --seed 0 1 2 --out-dir "$OUT"
done
