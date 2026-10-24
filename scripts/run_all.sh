#!/usr/bin/env bash
# Full sweep: 5 datasets x 5 methods. Regenerates every number in the report.
# The three cal_tda_* runs are one algorithm under three configs; --run-name
# names the record and the `method` column in results/summary.csv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(dtd flower102 pets eurosat aircraft)

for dataset in "${DATASETS[@]}"; do
    "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method clipzs
    "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method tda --config configs/method/tda.yaml
    "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method cal_tda --config configs/method/cal_tda_gate_temp.yaml --run-name cal_tda_gate_temp
    "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method cal_tda --config configs/method/cal_tda_gate.yaml --run-name cal_tda_gate
    "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method cal_tda --config configs/method/cal_tda_temp.yaml --run-name cal_tda_temp
done

python analysis/aggregate.py
python analysis/make_figures.py

# Keep the report's copies in sync with what was just regenerated, so the
# report never drifts from the results (Windows symlinks need admin rights).
mkdir -p report/figures
cp analysis/figures/*.png report/figures/

echo "done: results/summary.csv, analysis/figures/ and report/figures/ are up to date"
