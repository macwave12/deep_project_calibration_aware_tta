#!/usr/bin/env bash
# T01: extract CLIP features for the CPU replay (the project's only GPU job).
#   DATA_ROOT=/path/to/data_root bash scripts/extract_features.sh
# Works from any directory (runs from the repository root). Resumable: re-run the
# same command and finished pieces are kept. Output: features/ (or $OUT), log:
# $OUT/extract_features.log. Extra arguments go to the Python script, e.g.
#   bash scripts/extract_features.sh --datasets dtd pets
# Direct fallback:  python3 scripts/extract_features.py --data-root /path/to/data_root --out features
set -euo pipefail

if [[ -z "${DATA_ROOT:-}" ]]; then
    echo "Set DATA_ROOT to the dataset root" >&2
    exit 1
fi
DATA_ROOT="${DATA_ROOT%/}"
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "DATA_ROOT not found: $DATA_ROOT" >&2
    exit 1
fi
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"  # absolute: we change directory below
export DATA_ROOT PYTHONUNBUFFERED=1
OUT="${OUT:-features}"
PYTHON="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON" ]]; then
    echo "no python or python3 on PATH" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
LOG="$OUT/extract_features.log"
echo "extracting features to $OUT (data: $DATA_ROOT); log: $LOG"
"$PYTHON" scripts/extract_features.py --data-root "$DATA_ROOT" --out "$OUT" "$@" | tee -a "$LOG"
