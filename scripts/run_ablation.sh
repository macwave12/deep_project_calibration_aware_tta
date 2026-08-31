#!/usr/bin/env bash
# All eight on/off combinations of the three contributions, on every dataset.
#
# Each variant is a YAML config under configs/method/ablation/, not a boolean
# CLI flag: argparse treats `--flag False` as the non-empty string "False",
# which is truthy, so every "off" variant would silently run as full cal_tda.
# The configs flow through CalTdaConfig.from_dict, which rejects unknown keys.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(dtd flower102 pets eurosat aircraft)
VARIANTS=(none margin fusion temp margin_fusion margin_temp fusion_temp all)

mkdir -p results/raw/ablation

for dataset in "${DATASETS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        config="configs/method/ablation/${variant}.yaml"
        if [[ ! -f "$config" ]]; then
            echo "missing ablation config: $config" >&2
            exit 1
        fi
        "$SCRIPT_DIR/run_one.sh" --dataset "$dataset" --method cal_tda --config "$config" \
            --run-name "$variant" --out-dir results/raw/ablation
    done
done

python analysis/make_ablation_table.py

# The headline scatter and every per-dataset reliability diagram now also plot the
# ablation's `temp` variant (results/ablation.csv, results/raw/ablation/), so figures must
# be regenerated here too, not just after scripts/run_all.sh, or they silently go stale
# relative to a rerun ablation sweep.
python analysis/make_figures.py

mkdir -p report/figures
cp analysis/figures/*.png report/figures/

echo "done: results/ablation.csv, report/tables/ablation.tex, analysis/figures/ and report/figures/ are up to date"
