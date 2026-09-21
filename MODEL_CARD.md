---
license: mit
base_model: FacebookAI/xlm-roberta-large
datasets:
  - ZefanCai/Open-Jev
library_name: peft
tags:
  - pointer-network
  - decision-head
  - calibration
  - lora
  - xlm-roberta
  - routing
language:
  - multilingual
  - en
pipeline_tag: other
---

# Mirave XLM-RoBERTa-Large — Typed Decision Pointer Head

**This model is an open reproduction attempt of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One Model."** Jev's own weights and training
methodology aren't public; this project reproduces its externally
observable behavior and design goal — unstructured state in, typed
probabilistic decisions out, in one forward pass — from scratch, on a
different (open, bidirectionally-pretrained) base model and with a
supervised training recipe. See [Acknowledgements](#acknowledgements) for
exactly what's reproduced and what necessarily differs.

Mirave is a **typed decision head**: given a piece of context (`state`), a
`question`, and a list of `options`, it returns a calibrated probability
distribution over those options in a **single forward pass** — no text
generation, no fixed-size output layer, and no upper bound on how many
options a single call can score.

It is a LoRA adapter + a small pointer head (~1M parameters) on top of the
frozen [`FacebookAI/xlm-roberta-large`](https://huggingface.co/FacebookAI/xlm-roberta-large)
encoder. This card documents the **-large** (560M backbone) release; a
larger `xlm-roberta-xl` (3.5B) variant, trained on more compute, is planned
as a separate model sharing the same head design and prompt format.

Training/loading code, the dataset pipeline, and a Jev-API-compatible mock
inference server live in the project repository:
[github.com/edoigtrd/Mirave](https://github.com/edoigtrd/Mirave).

> **Status: trained.** These weights are the `best` checkpoint by
> validation loss (step 10,500 of a 14,835-step, 3-epoch run) — **not**
> the checkpoint from the end of training. Loss diverged late in the third
> epoch (val_loss jumped from 0.44 to a flat 0.96 and never recovered,
> most likely an unclipped exploding gradient — gradient clipping has
> since been added to the training script). The published weights predate
> that divergence and are unaffected by it; see
> [Training Procedure](#training-procedure) for the full trajectory.

## Model Details

| | |
|---|---|
| **Base model** | `FacebookAI/xlm-roberta-large` (560M params, 24 layers, hidden size 1024) |
| **Adaptation** | LoRA (rank 16, alpha 32, dropout 0.05) on `query`/`key`/`value`/`dense` in every attention and feed-forward block, plus a fully fine-tuned token embedding table |
| **Added head** | `PointerHead` — two `Linear(1024, 1024)` projections (query, key), ~2.1M parameters |
| **New vocabulary** | 8 special tokens: `<kind>`, `</kind>`, `<state>`, `</state>`, `<question>`, `</question>`, `<opt>`, `</opt>` |
| **License** | MIT (inherited from the base model) |
| **Language** | Multilingual capability inherited from XLM-R's 100-language pretraining; see [Training Data](#training-data) for what the fine-tuning set actually covers |

## How It Works

Most "pick one of N options" setups either train a fixed-size classifier
head (capped at N, retrained if N changes) or ask a generative model to
output the option verbatim (slow, and its probability isn't a clean
distribution over exactly the candidate set). This model does neither.

The base model, `xlm-roberta-large`, is bidirectional by pretraining
(masked language modeling, not causal), which matters here: every option
gets to see the full context and every other option through the same
self-attention, with no ordering asymmetry between "the first option" and
"the last option." A causal decoder-only model doesn't have that property
without surgery.

Input is serialized as a flat sequence:

```
<s> <kind>choice</kind> <state>...</state> <question>...</question> <opt>option A</opt> <opt>option B</opt> ... </s>
```

`<s>` is XLM-R's native first token (its embedding is already meaningful
from pretraining, unlike a freshly-initialized custom token). A single
`<opt>`/`</opt>` pair is reused for every option — there's no per-position
`<opt1>`, `<opt2>`, ... token, so the model can't learn a shortcut that
correlates an option's rank with its odds of being correct, and the option
count isn't baked into the vocabulary or the weights.

On the forward pass:

1. The hidden state at `<s>` (position 0) is projected to a **query** vector.
2. The hidden state at each option's `</opt>` is projected to a **key** vector.
3. Each option's logit is the dot product of the query with its key.
4. A softmax over the valid (non-padding) logits gives the output distribution.

Because the number of logits is just however many `</opt>` tokens were in
the prompt, the same weights score 2 options or 20 without retraining or
padding the output layer.

### Why LoRA + a fully-trainable embedding table

Full fine-tuning of a well-pretrained bidirectional encoder risks
catastrophic forgetting of a latent space that's expensive to rebuild;
freezing everything except the head leaves the 8 new special tokens stuck
at their random initialization, since nothing else in the model can teach
the network what they mean. Training LoRA adapters on the attention/MLP
projections *and* leaving the embedding table itself trainable threads
that needle: the pretrained representations move only through low-rank
updates, while the embedding table — the only place genuinely new
information (the special tokens) needs to land — gets full-rank updates.

In practice the embedding table (256M of the 263M trainable parameters)
also needs a much lower learning rate than the LoRA adapters: it's
full-parameter fine-tuning of already-well-trained weights, not a small
adapter, and using the same learning rate for both produced visible
training instability early on. The optimizer therefore uses two parameter
groups with independent learning rates.

## How to Get Started

There's no `trust_remote_code` `AutoModel.from_pretrained(...)` one-liner:
the pointer head isn't an architecture `transformers` knows natively, and
adding a custom `modeling_*.py` to the Hub repo was deliberately avoided in
favor of keeping code and weights separate — this repo hosts the weights,
loading code lives in
[github.com/edoigtrd/Mirave](https://github.com/edoigtrd/Mirave) (`model.py`
+ `dataset.py`).

```python
from model import load_checkpoint  # from github.com/edoigtrd/Mirave

model, tokenizer = load_checkpoint("path/to/this/checkpoint", device="cuda")

from dataset import build_example
example = build_example(
    tokenizer,
    state="Conversation with a customer about a delayed order.",
    question="What does the customer want?",
    kind="choice",
    options=["refund", "replacement", "status update"],
)
# batch, run through the model, softmax the logits — see infer.py for the
# full batched CLI version.
```

### Docker image

A prebuilt image serves this model behind an HTTP API compatible with the
real Jev API's documented contract (`POST /v1/systemone` — see
[docs.typesafe.ai](https://docs.typesafe.ai/)): same endpoint, same
`choice`/`score`/`noul` question types, same response shape, so client code
written for the real API (including the official `typesafe-sdk` Python
package) works against it unmodified.

```bash
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=<token with read access to this repo> \
    edoigtrd/mirave-inference:latest
```

[hub.docker.com/r/edoigtrd/mirave-inference](https://hub.docker.com/r/edoigtrd/mirave-inference)

No GPU? Drop `--gpus all` and add `-e MIRAVE_DEVICE=cpu` — same image, just
slower. Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MIRAVE_CHECKPOINT` | this repo (`Edoigtrd/Mirave-0.6B-xlm-roberta-large`) | Local checkpoint path, or a HF repo id to download at startup. |
| `MIRAVE_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `MIRAVE_API_KEY` | `dev-key` | Bearer token the mock server expects on `Authorization: Bearer <...>`. |
| `HF_TOKEN` | — | Needed only when `MIRAVE_CHECKPOINT` points at a private HF repo (this one is private). |

Source and build instructions: `inference/` in
[github.com/edoigtrd/Mirave](https://github.com/edoigtrd/Mirave).

## Training Data

Fine-tuned on [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev)
(config `release-v2-redistributable`, CC0-1.0 for the generated content), a
synthetic "typed decision" dataset of ~79K training examples pairing a
`state` + `question` + `options` with a reference probability distribution
(`target`) over those options. Three label `kind`s are mixed in training:

- **`choice`** — distribution over a genuine option set (e.g. routing a
  support ticket to a category from a small fixed list).
- **`noul`** — binary yes/no decisions.
- **`score`** — ordinal/rubric-style ratings expressed as a distribution
  over discrete levels. Note the head has no built-in ordinal inductive
  bias (options are compared pairwise via dot product, not ranked), so
  this is the kind where its calibration is weakest — see
  [Limitations](#limitations-and-bias).

Source task families span customer-support routing, game/agent control
(Snake, tic-tac-toe, a T-Rex-runner clone, ViZDoom, a tile platformer),
workflow control (invoice processing, security incidents, agent trace
observability), geometric-probability reasoning, and general reasoning
questions — all synthetic controls, not real user data or official
benchmark ground truth.

**Language coverage**: language metadata is only populated on a minority
of rows (about 4.2K of 79K); of those, the labels are `en` or `en+zh`
(code-switched). The remaining rows don't carry a language tag at all
(mostly the game/workflow-control families, whose templates are English).
Treat this model's non-English behavior as inherited from XLM-R's
pretraining, not as something the fine-tuning data specifically taught or
verified.

**What the model never sees**: the reference `target` distribution itself,
and any provenance/control metadata, are used only as training labels —
never serialized into the input prompt. Only `state`, `question`, `kind`,
and `options` are.

## Training Procedure

- **Optimizer**: AdamW, two parameter groups — LoRA adapters + head at a
  base learning rate, the embedding table at 10x lower — weight decay 0.01.
- **Schedule**: linear warmup (6% of steps) then linear decay.
- **Precision**: bf16 autocast via 🤗 Accelerate.
- **Batching**: variable option counts per example, so batches are built
  with a custom collator that pads sequences and options separately and
  masks padding options to `-inf` before the softmax; per-example loss is
  soft cross-entropy against the full `target` distribution (equivalent to
  KL divergence up to the target's own entropy).
- **Model selection**: checkpointed against the `validation` split every
  500 steps; the held-out `test`/`ood` splits are only touched once,
  after training, to produce the numbers below — never for model
  selection or hyperparameter tuning.
- **Hardware**: single consumer GPU (16GB), 3 epochs (14,835 steps) over
  the training split.
- **Gradient clipping**: **not** enabled for this run (max grad norm
  unclamped). Validation loss tracked cleanly downward from 0.81 to 0.44
  over the first ~10,500 steps, then jumped to 0.96 within 500 steps and
  stayed there — flat to 4 decimal places across every subsequent
  evaluation, consistent with a single exploding-gradient step pushing a
  weight out of its stable range rather than ordinary overfitting or
  noise. The published checkpoint is from step 10,500, before this
  happened. Gradient clipping (max norm 1.0) has since been added to the
  training script for future runs, including the planned `xlm-roberta-xl`
  variant.

## Evaluation

Full evaluation of the published (`best`, step 10,500) checkpoint. Per
the dataset's own usage rules, `test`/`ood` were held out and evaluated
exactly once, after training — not used for any decision during training.

| split | n | cross-entropy | KL vs. target | top-1 acc. | ECE |
|---|---:|---:|---:|---:|---:|
| `validation` | 3,723 | 0.435 | 0.382 | 0.782 | 0.010 |
| `test` | 10,356 | 0.381 | 0.351 | 0.797 | 0.022 |
| `ood` | 15,701 | 0.575 | 0.561 | 0.800 | 0.069 |
| `calibration`* | 4,672 | 0.347 | 0.303 | 0.831 | 0.017 |

\* the `calibration` split is meant for calibration procedures, not model
evaluation — included here for completeness, not as an additional test set.

**By `kind`** (test split):

| kind | n | cross-entropy | top-1 acc. |
|---|---:|---:|---:|
| `choice` | 2,408 | 0.741 | 0.612 |
| `noul` | 6,364 | 0.271 | 0.855 |
| `score` | 1,584 | 0.278 | 0.850 |

ECE (expected calibration error, over the head's own max-softmax
confidence) stays low on `validation`/`test`/`calibration` (~0.01–0.02)
and roughly triples on `ood` (0.069) — the model's confidence is well
calibrated in-distribution but overconfident on unseen phrasing, which
tracks with `ood`'s own documentation: it's held-out wording/families
*within* the same source task types, not a general unseen-domain
benchmark, so this measures robustness to phrasing drift specifically.

Note `choice`'s top-1 accuracy (0.61) looks lower than `noul`/`score`
(0.85 each) but isn't directly comparable to them: `choice` items have up
to 5–6 options against `noul`'s fixed 2, so chance-level accuracy is much
lower to begin with.

## Limitations and Bias

- **No ordinal inductive bias for `score`.** The pointer mechanism treats
  options as an unordered set — it has no explicit notion that option 3
  is "between" options 2 and 4 on a scale. In practice `score`'s top-1
  accuracy and loss ended up comparable to `noul`'s (see
  [Evaluation](#evaluation)), so this isn't visible in aggregate metrics,
  but it means a wrong `score` prediction has no architectural pressure to
  land *near* the true rating rather than far from it — worth checking
  directly (e.g. mean rank distance on errors) before relying on this for
  anything where a near-miss and a wild miss should be scored differently.
- **`choice` lags the other two kinds.** 0.61 top-1 on `test` vs. 0.80+
  for `noul`/`score` — not directly comparable given `choice` items carry
  up to 5–6 options against `noul`'s fixed 2, but still the kind to watch
  most closely for a given application, and the main lever for further
  gains would be more `choice`-kind training data or longer training
  specifically on it.
- **Synthetic training data.** All labels come from templated/synthetic
  generation, not human annotation of real interactions. Control-latent
  patterns baked into the generators (explicit priority rules, fixed
  categories, etc.) may not transfer to messier real-world phrasing.
- **Sensitive to prompt truncation on pathologically long inputs.** State
  and question text are truncated to fit the encoder's context window if
  needed; individual options are never truncated to zero (the pointer
  mechanism needs an intact `</opt>` per option), but a very long option
  list with long option text can still hit the length budget — this is
  logged as a hard error rather than a silent truncation.
- **Multilingual claims are inherited, not verified.** See
  [Training Data](#training-data) above.
- **Not a general-purpose classifier out of the box.** The model expects
  its specific tagged prompt format; feeding it free-form text without the
  `<state>`/`<question>`/`<opt>` structure is out of distribution.

## Acknowledgements

This project's goal is to reproduce [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's first "System One Model": a "frontier-intelligence function
call" that turns unstructured state into typed, calibrated probabilistic
decisions instead of generated text, aimed at software that needs to *use*
a model's output directly rather than parse it out of a chat completion.

TypeSafe hasn't published Jev's weights, base model, or training data, so
this is a reproduction of the *idea*, not a distillation of the model
itself — built independently, from a different starting point:

| | Jev (as described publicly) | This model |
|---|---|---|
| Output | All structured values via a custom parallel sampler | A single typed decision (state + question + options → distribution) per forward pass |
| Base model | Not disclosed | `xlm-roberta-large`, chosen specifically for being bidirectionally pretrained (see [How It Works](#how-it-works)) |
| Training | "Reinforcement Learning for Calibrated Decisions (RLCD)" | Supervised: LoRA fine-tuning against labeled target distributions (soft cross-entropy) |
| Availability | Closed | Open weights (adapter + head) and open training code |

The core bet carried over is architectural, not the training method: that
a model built to emit *only* the typed decision — rather than a token
stream that happens to contain one — can be both faster and more directly
usable by calling software than prompting a general chat model for the
same answer.
