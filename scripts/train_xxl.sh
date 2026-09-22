#!/usr/bin/env bash
# Trains the xlm-roberta-xxl (10.7B backbone) variant. Same caveats as
# train_xl.sh, more so: this needs a serious multi-GPU or very-large-single-
# GPU cloud instance. Smaller batch size + more accumulation than xl as a
# starting point, but expect to tune this against whatever you actually
# rent before trusting the numbers.
#
# Usage: scripts/train_xxl.sh [extra train.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/train.sh facebook/xlm-roberta-xxl checkpoints/xlmr-xxl-pointer \
    --batch-size 2 \
    --grad-accum 16 \
    --eval-batch-size 2 \
    --gradient-checkpointing \
    --eval-every 250 \
    "$@"
