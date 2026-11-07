#!/usr/bin/env bash
# T01: extract CLIP features for the CPU replay (the project's only GPU job).
# Run from the repository root:  DATA_ROOT=/path/to/data_root bash scripts/extract_features.sh
# Resumable: a dataset with features/<dataset>/meta.json is skipped on re-run.
# Extra arguments are passed through, e.g.  bash scripts/extract_features.sh --datasets dtd pets
set -euo pipefail

if [[ -z "${DATA_ROOT:-}" ]]; then
    echo "Set DATA_ROOT to the dataset root" >&2
    exit 1
fi
OUT="${OUT:-features}"

echo "extracting features to $OUT (data: $DATA_ROOT)"
python scripts/extract_features.py --data-root "$DATA_ROOT" --out "$OUT" "$@"
