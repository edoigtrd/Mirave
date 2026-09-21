"""Mock of TypeSafe AI's Jev HTTP API (https://docs.typesafe.ai/), backed by
this project's own model instead of the real (closed) one.

Matches the real API's documented contract as closely as possible:

    POST /v1/systemone
    Authorization: Bearer <API_KEY>
    Content-Type: application/json

    {
      "state": "...",
      "model": "jev-latest",
      "questions": {
        "<name>": {"type": "choice", "instructions": "...", "criteria": {"key": "description", ...}},
        "<name>": {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]},
        "<name>": {"type": "noul",   "instructions": "...", "criteria": {"yes": "...", "no": "..."}}
      }
    }

Response shape (per-type answer objects) and field names follow the same
documentation. Real Jev evaluates every question in parallel in a single
call; this mock does the same thing for real — all questions in a request
become one batch and go through the model in a single forward pass, so
adding questions barely changes response time here either.

The one field the docs don't specify precisely enough to fake (score's
`score` value can be non-integer, "between levels") is computed as the
distribution's expectation over level indices — the natural reading of
"between levels" for a model that outputs a probability per level.

    uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

try:
    from dataset import _as_text, build_example, collate
    from model import load_checkpoint
except ModuleNotFoundError:
    # Local dev layout: model.py/dataset.py live one directory up from here
    # (in the Docker image they're copied as siblings instead — see build.sh).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dataset import _as_text, build_example, collate
    from model import load_checkpoint

CHECKPOINT = os.environ.get("MIRAVE_CHECKPOINT", "Edoigtrd/Mirave-0.6B-xlm-roberta-large")
DEVICE = os.environ.get("MIRAVE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = int(os.environ.get("MIRAVE_MAX_LENGTH", "512"))
API_KEY = os.environ.get("MIRAVE_API_KEY", "dev-key")
MODEL_NAME = "mirave-0.6b-xlm-roberta-large"

# noul options are fixed and match training-time convention exactly:
# index 0 = "no", index 1 = "yes".
NOUL_OPTIONS = ["no", "yes"]


def resolve_checkpoint(spec: str) -> str:
    """A local directory, or a HF Hub repo id to download (private repos
    need HF_TOKEN set — this is how the Docker image gets weights without
    baking them into the image)."""
    if Path(spec).exists():
        return spec
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=spec, token=os.environ.get("HF_TOKEN"))


print(f"loading checkpoint '{CHECKPOINT}' onto {DEVICE}...", flush=True)
_model, _tokenizer = load_checkpoint(resolve_checkpoint(CHECKPOINT), device=DEVICE)
print("ready.", flush=True)


# The real API accepts `state`/`instructions`/criteria values as plain text
# *or* arbitrary JSON (object/array) — see typesafe_sdk._core.json_types
# .JSONContent. `Any` here is deliberately permissive rather than modeling
# that recursive union field-for-field; _as_text() below does the actual
# text coercion for whatever comes in.


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Any | None = None
    criteria: dict[str, Any | None]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Any | None = None
    criteria: list[Any]


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Any | None = None
    criteria: dict[str, Any | None] | None = None


Question = Annotated[Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: Any
    model: str = "jev-latest"  # accepted for API compatibility, not used to pick a model
    questions: dict[str, Question]


app = FastAPI(title="Jev-compatible mock API", description=__doc__)


def check_auth(authorization: str | None) -> None:
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest, authorization: Annotated[str | None, Header()] = None) -> dict:
    check_auth(authorization)
    if not req.questions:
        raise HTTPException(status_code=422, detail="questions must be non-empty")

    names = list(req.questions.keys())
    examples = []
    per_question: list[dict] = []

    for name in names:
        q = req.questions[name]
        instructions = _as_text(q.instructions) if q.instructions is not None else ""

        if q.type == "choice":
            keys = list(q.criteria.keys())
            options = [
                f"{key}: {_as_text(desc)}" if desc is not None else key
                for key, desc in q.criteria.items()
            ]
            per_question.append({"type": "choice", "keys": keys})
        elif q.type == "score":
            options = [_as_text(level) for level in q.criteria]
            per_question.append({"type": "score", "legend": options})
        else:  # noul
            options = NOUL_OPTIONS
            if q.criteria:
                instructions += " (" + "; ".join(
                    f"{k}: {_as_text(v)}" for k, v in q.criteria.items() if v is not None
                ) + ")"
            per_question.append({"type": "noul"})

        examples.append(
            build_example(
                _tokenizer,
                state=req.state,
                question=instructions,
                kind=q.type,
                options=options,
                max_length=MAX_LENGTH,
            )
        )

    batch = collate(examples, pad_token_id=_tokenizer.pad_token_id)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    start = time.perf_counter()
    with torch.no_grad():
        logits = _model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            opt_positions=batch["opt_positions"],
            opt_mask=batch["opt_mask"],
        ).logits
        probs = torch.softmax(logits.masked_fill(~batch["opt_mask"], float("-inf")), dim=-1)
    latency_ms = (time.perf_counter() - start) * 1000

    answers: dict[str, dict] = {}
    for i, name in enumerate(names):
        meta = per_question[i]
        n_opts = int(batch["opt_mask"][i].sum().item())
        p = probs[i, :n_opts].tolist()
        best = max(range(n_opts), key=lambda j: p[j])

        if meta["type"] == "choice":
            keys = meta["keys"]
            answers[name] = {
                "type": "choice",
                "choice": keys[best],
                "confidence": p[best],
                "probabilities": dict(zip(keys, p)),
            }
        elif meta["type"] == "score":
            expected_score = sum(level * prob for level, prob in enumerate(p))
            answers[name] = {
                "type": "score",
                "score": expected_score,
                "confidence": p[best],
                "legend": {str(level): text for level, text in enumerate(meta["legend"])},
                "probabilities": {str(level): prob for level, prob in enumerate(p)},
            }
        else:  # noul
            answers[name] = {"type": "noul", "noul": p[1]}

    return {
        "model": MODEL_NAME,
        "answers": answers,
        "usage": {
            "input_tokens": int(batch["attention_mask"].sum().item()),
            "output_tokens": 0,
        },
        # not part of the real Jev response schema — kept for local
        # benchmarking against Jev's own published latency numbers.
        "latency_ms": round(latency_ms, 1),
    }
