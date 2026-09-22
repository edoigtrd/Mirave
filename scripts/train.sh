#!/usr/bin/env bash
# Generic trainer wrapper — the xl/xxl scripts in this directory are thin
# presets over this. Not meant to be run directly unless you want to pick
# every flag yourself.
#
# Usage: scripts/train.sh <base_model_id> <output_dir> [extra train.py args...]
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <base_model_id> <output_dir> [extra train.py args...]" >&2
    exit 1
fi

BASE_MODEL="$1"
OUTPUT_DIR="$2"
shift 2

cd "$(dirname "$0")/.."
uv run train.py --base-model "$BASE_MODEL" --output-dir "$OUTPUT_DIR" "$@"
