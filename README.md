# Mirave

An open reproduction attempt of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One Model": instead of generating text, the model
takes a `state` + `question` + a list of `options` and returns a calibrated
probability distribution over those options, in a single forward pass.

- **Model weights**: [huggingface.co/Edoigtrd/Mirave-0.6B-xlm-roberta-large](https://huggingface.co/Edoigtrd/Mirave-0.6B-xlm-roberta-large)
  (LoRA adapter on `FacebookAI/xlm-roberta-large`) — see [`MODEL_CARD.md`](MODEL_CARD.md)
  for the architecture, training procedure, and evaluation results.
- **Training data**: [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev)

This repo is the code: data loading, training, inference, evaluation, and a
mock HTTP server that speaks the same wire format as the real Jev API.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/). A local MongoDB
instance holds the training data (`data.py` populates it from the HF
dataset above).

```bash
uv sync
```

Create a `.env` with MongoDB credentials:

```
MONGODB_USERNAME=root
MONGODB_PASSWORD=...
MONGODB_HOST=localhost
MONGODB_PORT=27017
```

Then load the dataset into Mongo (idempotent — drops and reloads every
collection on each run):

```bash
uv run data.py
```

## Repo layout

| File | Purpose |
|---|---|
| `data.py` | Downloads Open-Jev from HF and loads it into MongoDB (`train`/`validation`/`test`/`ood`/`calibration` collections). |
| `model.py` | The `PointerHead` architecture, LoRA setup, checkpoint save/load. |
| `dataset.py` | Prompt construction (`build_example`) and the MongoDB-backed `torch.utils.data.Dataset`. |
| `train.py` | Fine-tunes the pointer head with LoRA; logs to file + console, checkpoints on validation improvement, handles Ctrl+C gracefully. |
| `infer.py` | CLI for running a trained checkpoint on new examples (single example or JSONL batch). |
| `evaluate.py` | Full evaluation of a checkpoint on a split: cross-entropy, KL vs. target, top-1 accuracy, expected calibration error — broken down by `kind`. |
| `inference/` | A FastAPI server mocking the real Jev HTTP API (`POST /v1/systemone`), backed by this model instead. Includes a `Dockerfile` + `build.sh`. |

## Training

```bash
uv run train.py --output-dir checkpoints/xlmr-large-pointer
```

Trains on the `train` split only, model-selects against `validation`.
`test`/`ood`/`calibration` are never touched during training. See
`train.py --help` for hyperparameters (LoRA rank/alpha, learning rates,
batch size, gradient clipping, ...).

## Evaluation

```bash
uv run evaluate.py --checkpoint checkpoints/xlmr-large-pointer/best --split test
```

## Inference

CLI, single example or a JSONL file of examples:

```bash
uv run infer.py --checkpoint checkpoints/xlmr-large-pointer/best \
    --kind choice --state "..." --question "..." \
    --options "option a" "option b" "option c"
```

### Mock HTTP server

`inference/server.py` implements the real Jev API's documented contract
(`POST /v1/systemone`, the `choice`/`score`/`noul` question types, the same
response shape) against this project's own model, so client code written
for the real API — including the real `typesafe-sdk` Python package — works
unmodified against it. See [`exemple.py`](exemple.py) for a working example
pointed at a local instance.

```bash
uv run uvicorn inference.server:app --host 0.0.0.0 --port 8000
```

Or via Docker (build context is `inference/`, not the repo root — run the
build script rather than `docker build` directly, since it also copies in
the two shared modules the server depends on):

```bash
./inference/build.sh
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=<token with read access to the model repo> \
    edoigtrd/mirave-inference:latest
```

Env vars: `MIRAVE_CHECKPOINT` (local path or HF repo id, defaults to the
published model above), `MIRAVE_DEVICE` (`cuda`/`cpu`), `MIRAVE_API_KEY`
(bearer token the server expects).

## License

MIT.
