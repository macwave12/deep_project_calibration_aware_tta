#!/usr/bin/env bash
# Run a single dataset+method and write results/raw/<dataset>_<method>.json.
# Flag names are upstream's: --data, --test_sets, -a, --algorithm.
#
# `-j 0` is mandatory: Windows' `spawn` start method cannot pickle the
# `_convert_image_to_rgb` closure defined inside online_tta.py, so any run
# with dataloader workers > 0 crashes there. Kept here too (not just in the
# PowerShell script) so this Colab fallback behaves identically.
set -euo pipefail

DATASET=""
METHOD=""
CONFIG=""
DATA_ROOT="${DATA_ROOT:-}"
ARCH="ViT-B/16"
SEED=0
RUN_NAME=""
OUT_DIR=""

usage() {
    echo "usage: run_one.sh --dataset <key> --method <clipzs|tda|cal_tda> [--config path] [--data-root path] [--arch name] [--seed n] [--run-name name] [--out-dir path]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ -z "$DATASET" || -z "$METHOD" ]] && usage
if [[ -z "$DATA_ROOT" ]]; then
    echo "Set DATA_ROOT or pass --data-root" >&2
    exit 1
fi

# Our dataset keys (configs, filenames, CSV, --dataset here) are lowercase.
# Upstream's --test_sets values are mixed-case and internally inconsistent
# ("eurosat" stays lowercase on both sides) — this mapping is the only place
# in the shell layer that difference is handled. Getting a case wrong here
# produces a loud KeyError in upstream's tuned alpha/beta table.
case "$DATASET" in
    dtd) TEST_SET="DTD" ;;
    flower102) TEST_SET="Flower102" ;;
    pets) TEST_SET="Pets" ;;
    eurosat) TEST_SET="eurosat" ;;
    aircraft) TEST_SET="Aircraft" ;;
    *)
        echo "unknown dataset key '$DATASET'; known keys: dtd flower102 pets eurosat aircraft" >&2
        exit 1
        ;;
esac

if [[ -z "$OUT_DIR" ]]; then
    # `tda_equivalent.yaml` is the flags-off control that proves `cal_tda`
    # collapses to upstream TDA; it is a verification artifact, not a
    # headline method result, so its record must not land beside the three
    # studied methods where `analysis/aggregate.py`'s non-recursive glob over
    # `results/raw/*.json` would fold it into `results/summary.csv`.
    if [[ "$CONFIG" == "configs/method/tda_equivalent.yaml" ]]; then
        OUT_DIR="results/raw/verification"
    else
        OUT_DIR="results/raw"
    fi
fi

CKPT_TAG="${RUN_NAME:-$METHOD}"

ARGS=(online_tta.py --data "$DATA_ROOT" --test_sets "$TEST_SET" -a "$ARCH" -b 1 -j 0 --gpu 0 \
      --ctx_init a_photo_of_a -p 50 --output_dir "online_results/ckps/$CKPT_TAG" \
      --algorithm "$METHOD" --seed "$SEED" --out-dir "$OUT_DIR")
[[ -n "$CONFIG" ]] && ARGS+=(--config "$CONFIG")
[[ -n "$RUN_NAME" ]] && ARGS+=(--run-name "$RUN_NAME")

echo "running: $DATASET ($TEST_SET) / $METHOD ($ARCH)"
python "${ARGS[@]}"
