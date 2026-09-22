#!/usr/bin/env bash
# Usage: scripts/push_xxl.sh [--private]
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/push.sh checkpoints/xlmr-xxl-pointer/best Edoigtrd/Mirave-10.7B-xlm-roberta-xxl "${1:-}"
