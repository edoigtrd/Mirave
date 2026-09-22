#!/usr/bin/env bash
# Usage: scripts/push_xl.sh [--private]
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/push.sh checkpoints/xlmr-xl-pointer/best Edoigtrd/Mirave-4.2B-xlm-roberta-xl "${1:-}"
