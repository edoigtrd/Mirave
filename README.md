# Mirave

An open reproduction attempt of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One Model": instead of generating text, the model
takes a `state` + `question` + a list of `options` and returns a calibrated
probability distribution over those options, in a single forward pass.

Two sizes are available, sharing the same head design, prompt format, and
training data:

- **-large**: [huggingface.co/Edoigtrd/Mirave-0.6B-xlm-roberta-large](https://huggingface.co/Edoigtrd/Mirave-0.6B-xlm-roberta-large)
  (LoRA adapter on `FacebookAI/xlm-roberta-large`, 560M backbone) — see
  [`MODEL_CARD_large.md`](MODEL_CARD_large.md) for architecture, training
  procedure, and evaluation.
- **-xl**: [huggingface.co/Edoigtrd/Mirave-4.2B-xlm-roberta-xl](https://huggingface.co/Edoigtrd/Mirave-4.2B-xlm-roberta-xl)
  (LoRA adapter on `facebook/xlm-roberta-xl`, 3.5B backbone) — see
  [`MODEL_CARD_XL.md`](MODEL_CARD_XL.md). Meaningfully more accurate
  across every question kind (see its card's comparison table), at the
  cost of a much larger backbone to download and run.
- **Training data**: [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev)

This repo is the code: data loading, training, inference, evaluation, and a
mock HTTP server that speaks the same wire format as the real Jev API.

## Try it

Two ways to run the model — neither needs MongoDB or any of the training
pipeline below, that's only for retraining.

### Docker (fastest)

```bash
docker run --gpus all -p 8000:8000 edoigtrd/mirave-inference:latest
```

Starts an HTTP server at `http://localhost:8000` compatible with the real
Jev API's documented contract (`POST /v1/systemone`):

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Authorization: Bearer dev-key" -H "Content-Type: application/json" \
  -d '{
    "state": "I was charged twice and need the duplicate refunded today.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {"billing": "Payment issues", "technical": "Bugs", "sales": "Pricing questions"}
      }
    }
  }'
```

Or use the official `typesafe-sdk` Python client pointed at it instead of
the real API — see [`exemple.py`](exemple.py) for a working example.

Defaults to the **-large** checkpoint. For **-xl**, add
`-e MIRAVE_CHECKPOINT=Edoigtrd/Mirave-4.2B-xlm-roberta-xl`.

No GPU? Drop `--gpus all` and add `-e MIRAVE_DEVICE=cpu` — same image, just
slower (much slower for -xl). Full env var reference in
[`MODEL_CARD_large.md`](MODEL_CARD_large.md#docker-image) /
[`MODEL_CARD_XL.md`](MODEL_CARD_XL.md#docker-image).

### Local (Python, no Docker)

```bash
uv sync
uv run infer.py --checkpoint Edoigtrd/Mirave-0.6B-xlm-roberta-large \
    --kind choice --state "..." --question "..." \
    --options "option a" "option b" "option c"
```

`--checkpoint` downloads straight from
[Edoigtrd/Mirave-0.6B-xlm-roberta-large](https://huggingface.co/Edoigtrd/Mirave-0.6B-xlm-roberta-large)
on the Hub — swap in `Edoigtrd/Mirave-4.2B-xlm-roberta-xl` for the -xl
checkpoint instead. Point it at a local directory instead once you've
trained your own checkpoint with `train.py` below.

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

For the -xl backbone, use the preset wrapper instead — it sets the batch
size, gradient accumulation, and `--gradient-checkpointing` this much
bigger backbone needs, and won't fit a single 16GB consumer GPU the way
-large does (the published -xl checkpoint was trained on a rented H100 —
[RunPod](https://runpod.io?ref=5jrts9za) is one option for a GPU with
enough headroom):

```bash
scripts/train_xl.sh
```

## Evaluation

```bash
uv run evaluate.py --checkpoint checkpoints/xlmr-large-pointer/best --split test
```

## Inference

`infer.py` also takes a JSONL file for batch inference instead of a single
`--state`/`--question`/`--options` example (see [Try it](#try-it) above for
the single-example form):

```bash
uv run infer.py --checkpoint checkpoints/xlmr-large-pointer/best --jsonl examples.jsonl
```

Each line: `{"state": ..., "question": ..., "kind": ..., "options": [...]}`.

### Mock HTTP server, from source

The [Try it](#try-it) section above runs the prebuilt image. To build it
yourself instead — e.g. after retraining, or to change the server code —
build context is `inference/`, not the repo root, so build via the script
rather than `docker build` directly (it also copies in the two shared
modules the server imports):

```bash
./inference/build.sh
```

Or run it straight from Python, no Docker at all:

```bash
uv run uvicorn inference.server:app --host 0.0.0.0 --port 8000
```

Env vars: `MIRAVE_CHECKPOINT` (local path or HF repo id, defaults to the
published model above), `MIRAVE_DEVICE` (`cuda`/`cpu`), `MIRAVE_API_KEY`
(bearer token the server expects).

## License

MIT.
