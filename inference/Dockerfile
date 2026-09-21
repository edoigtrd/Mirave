# Build context is THIS directory, not the repo root — run inference/build.sh
# rather than `docker build` directly, since model.py/dataset.py need to be
# copied in first (Docker can't COPY from outside the build context, and the
# repo root has ~10GB of unrelated stuff — .venv/, checkpoints/, logs/ — that
# has no business being sent to the daemon as build context):
#
#   ./inference/build.sh                 # copies the two files in, then builds
#   docker run --gpus all -p 8000:8000 \
#       -e HF_TOKEN=<token with read access to the private checkpoint repo> \
#       edoigtrd/mirave-inference:latest
#
# No GPU available? Drop --gpus all and set -e MIRAVE_DEVICE=cpu — the same
# image runs on CPU, just slower.

FROM python:3.14-slim

WORKDIR /app

# PyTorch's cu130 wheels bundle their own CUDA runtime libraries (cublas,
# cudnn, ...) as pip dependencies, so a plain slim base is enough here —
# the host only needs the NVIDIA driver + nvidia-container-toolkit for
# `--gpus all` to work, no CUDA base image required.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

RUN pip install --no-cache-dir \
    transformers \
    peft \
    safetensors \
    huggingface_hub \
    fastapi \
    "uvicorn[standard]"

# model.py / dataset.py are copied here by build.sh right before this runs —
# see that script rather than editing these two files in place.
COPY model.py dataset.py server.py ./

ENV MIRAVE_CHECKPOINT=Edoigtrd/Mirave-0.6B-xlm-roberta-large
ENV MIRAVE_DEVICE=cuda
ENV MIRAVE_API_KEY=dev-key

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
