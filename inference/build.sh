#!/usr/bin/env bash
# Builds the inference image with a build context scoped to this directory
# (not the repo root — see the Dockerfile's header comment for why).
#
# model.py and dataset.py are the single source of truth at the repo root;
# this script copies them in fresh before every build so there's no second
# copy to hand-maintain. Don't edit inference/model.py or inference/dataset.py
# directly — edit the root ones and rerun this script.
set -euo pipefail
cd "$(dirname "$0")"

cp ../model.py ../dataset.py .
docker build -t "${1:-edoigtrd/mirave-inference:latest}" .
