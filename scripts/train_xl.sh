#!/usr/bin/env bash
# Trains the xlm-roberta-xl (3.5B backbone) variant. This will NOT fit on a
# 16GB local card — the embedding table alone (a full-rank trainable
# ~250k x hidden_size matrix, see MODEL_CARD.md's "Why LoRA + a
# fully-trainable embedding table") is far larger than -large's, on top of
# a 3.5B frozen backbone. Meant for a cloud GPU with real headroom
# (A100/H100 80GB class). --gradient-checkpointing trades compute for
# activation memory, which matters much more here than on -large.
#
# The batch size / grad-accum / eval-every below are starting points, not
# tuned for whatever instance you actually rent — watch the first few
# hundred steps' memory usage and adjust.
#
# Usage: scripts/train_xl.sh [extra train.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/train.sh facebook/xlm-roberta-xl checkpoints/xlmr-xl-pointer \
    --batch-size 8 \
    --grad-accum 4 \
    --eval-batch-size 8 \
    --gradient-checkpointing \
    --eval-every 250 \
    "$@"
