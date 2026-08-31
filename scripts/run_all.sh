#!/usr/bin/env bash
# Full sweep: 5 datasets x 3 methods. Regenerates every number in the report.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(dtd flower102 pets eurosat aircraft)

for dataset in "${DATASETS[@]}"; do
    "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method clipzs
    "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method tda --config configs/method/tda.yaml
    "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method cal_tda --config configs/method/cal_tda.yaml
done

python analysis/aggregate.py
python analysis/make_figures.py

# Keep the report's copies in sync with what was just regenerated, so the
# report never drifts from the results (Windows symlinks need admin rights).
mkdir -p report/figures
cp analysis/figures/*.png report/figures/

echo "done: results/summary.csv, analysis/figures/ and report/figures/ are up to date"
