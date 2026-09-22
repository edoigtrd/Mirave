#!/usr/bin/env bash
# Generic push wrapper — the xl/xxl scripts in this directory are thin
# presets over this. Evaluates the checkpoint on validation/test/ood (so
# you have real numbers before publishing, not just training-time ones),
# then uploads the checkpoint (adapter + tokenizer + head) to a HF repo.
#
# Does NOT write or push a model card: eval numbers only exist after this
# script runs, and the card's prose (limitations, comparisons, etc.) needs
# a human read of those numbers, not a template — write/push README.md to
# the repo as a separate, deliberate step once you've looked at the output
# below (see how MODEL_CARD.md was done for -large).
#
# Usage: scripts/push.sh <checkpoint_dir> <hf_repo_id> [--private]
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <checkpoint_dir> <hf_repo_id> [--private]" >&2
    exit 1
fi

CHECKPOINT_DIR="$1"
REPO_ID="$2"
PRIVATE=false
if [ "${3:-}" = "--private" ]; then
    PRIVATE=true
fi

cd "$(dirname "$0")/.."

EVAL_DIR="$CHECKPOINT_DIR/eval"
mkdir -p "$EVAL_DIR"

echo "=== Evaluating $CHECKPOINT_DIR before publishing ==="
for split in validation test ood; do
    echo "--- $split ---"
    uv run evaluate.py --checkpoint "$CHECKPOINT_DIR" --split "$split" --output "$EVAL_DIR/$split.yaml"
done
echo "Full results written to $EVAL_DIR/*.yaml — pull the numbers for the model card from there."

echo "=== Uploading weights to $REPO_ID ==="
PRIVATE="$PRIVATE" CHECKPOINT_DIR="$CHECKPOINT_DIR" REPO_ID="$REPO_ID" uv run python - <<'PYEOF'
import os
from huggingface_hub import HfApi

checkpoint_dir = os.environ["CHECKPOINT_DIR"]
repo_id = os.environ["REPO_ID"]
private = os.environ["PRIVATE"] == "true"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    folder_path=checkpoint_dir,
    path_in_repo=".",
    ignore_patterns=["adapter/README.md"],
    commit_message="Upload checkpoint",
)
print(f"Pushed weights to https://huggingface.co/{repo_id}")
PYEOF

echo
echo "Weights are up. Now write/push a model card by hand using the eval numbers above."
