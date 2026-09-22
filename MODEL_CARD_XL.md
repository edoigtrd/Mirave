---
license: mit
base_model: facebook/xlm-roberta-xl
datasets:
  - ZefanCai/Open-Jev
library_name: peft
tags:
  - pointer-network
  - decision-head
  - calibration
  - lora
  - xlm-roberta
  - xlm-roberta-xl
  - routing
language:
  - multilingual
  - en
pipeline_tag: other
---

# Mirave XLM-RoBERTa-XL — Typed Decision Pointer Head

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

It is a LoRA adapter + a small pointer head (~13.1M parameters) on top of
the frozen [`facebook/xlm-roberta-xl`](https://huggingface.co/facebook/xlm-roberta-xl)
encoder (3.5B params). This card documents the **-xl** release. A smaller
[**-large**](MODEL_CARD_large.md) variant (560M backbone,
`FacebookAI/xlm-roberta-large`) also exists, sharing the same head design,
prompt format, and training dataset — see [Compared to -large](#compared-to--large)
below for how the two stack up.

Training/loading code, the dataset pipeline, and a Jev-API-compatible mock
inference server live in the project repository:
[github.com/edoigtrd/Mirave](https://github.com/edoigtrd/Mirave).

> **Status: trained.** Unlike the -large run, this one completed all 3
> epochs (7,419 steps) with validation loss and top-1 accuracy improving
> essentially monotonically throughout — no divergence (see
> [Training Procedure](#training-procedure)). The published weights are
> the `best` checkpoint by validation loss, at step 7,250/7,419. Only the
> `test` split has been evaluated so far (see [Evaluation](#evaluation));
> `validation`/`ood`/`calibration` numbers aren't included in this card.

## Model Details

| | |
|---|---|
| **Base model** | `facebook/xlm-roberta-xl` (3.5B params, 36 layers, hidden size 2560) |
| **Adaptation** | LoRA (rank 16, alpha 32, dropout 0.05) on `query`/`key`/`value`/`dense` in every attention and feed-forward block (~26.6M params), plus a fully fine-tuned token embedding table (~640.0M params) — ~666.6M trainable adapter parameters on top of the frozen backbone |
| **Added head** | `PointerHead` — two `Linear(2560, 2560)` projections (query, key), ~13.1M parameters |
| **Total parameters** | ~4.16B loaded (3.48B frozen backbone + ~679.8M trainable: ~666.6M adapter + ~13.1M head) |
| **New vocabulary** | 8 special tokens: `<kind>`, `</kind>`, `<state>`, `</state>`, `<question>`, `</question>`, `<opt>`, `</opt>` (same as -large) |
| **License** | MIT (inherited from the base model) |
| **Language** | Multilingual capability inherited from XLM-R's 100-language pretraining; see [Training Data](#training-data) for what the fine-tuning set actually covers |

## How It Works

Most "pick one of N options" setups either train a fixed-size classifier
head (capped at N, retrained if N changes) or ask a generative model to
output the option verbatim (slow, and its probability isn't a clean
distribution over exactly the candidate set). This model does neither.

The base model, `xlm-roberta-xl`, is bidirectional by pretraining (masked
language modeling, not causal), which matters here: every option gets to
see the full context and every other option through the same
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
padding the output layer. This is identical to -large's mechanism, just on
a bigger encoder — see -large's card for the fuller discussion of why LoRA
plus a fully-trainable embedding table is used instead of full fine-tuning
or a fully-frozen backbone.

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

The same prebuilt image that serves -large can serve this checkpoint
instead — same HTTP API, compatible with the real Jev API's documented
contract (`POST /v1/systemone` — see [docs.typesafe.ai](https://docs.typesafe.ai/)):

```bash
docker run --gpus all -p 8000:8000 \
    -e MIRAVE_CHECKPOINT=Edoigtrd/Mirave-4.2B-xlm-roberta-xl \
    edoigtrd/mirave-inference:latest
```

`MIRAVE_CHECKPOINT` must be set explicitly — the image defaults to the
-large repo. [hub.docker.com/r/edoigtrd/mirave-inference](https://hub.docker.com/r/edoigtrd/mirave-inference)

No GPU? Drop `--gpus all` and add `-e MIRAVE_DEVICE=cpu` — same image, just
much slower given the 3.5B backbone. Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MIRAVE_CHECKPOINT` | `Edoigtrd/Mirave-0.6B-xlm-roberta-large` | Local checkpoint path, or a HF repo id to download at startup. Set to `Edoigtrd/Mirave-4.2B-xlm-roberta-xl` for this model. |
| `MIRAVE_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `MIRAVE_API_KEY` | `dev-key` | Bearer token the mock server expects on `Authorization: Bearer <...>`. |
| `HF_TOKEN` | — | Only needed if `MIRAVE_CHECKPOINT` is pointed at a private HF repo instead of the (public) default above. |

Source and build instructions: `inference/` in
[github.com/edoigtrd/Mirave](https://github.com/edoigtrd/Mirave).

## Training Data

Fine-tuned on [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev)
(config `release-v2-redistributable`, CC0-1.0 for the generated content) —
the same dataset, split, and preprocessing as -large. See
[-large's Training Data section](MODEL_CARD_large.md#training-data) for
the full breakdown of task families, label `kind`s, and language coverage
caveats; nothing about the data changed between the two releases.

## Training Procedure

- **Optimizer**: AdamW, two parameter groups — LoRA adapters + head at
  `lr=2e-4`, the embedding table at `lr=2e-5` (10x lower, same ratio as
  -large) — weight decay 0.01.
- **Schedule**: linear warmup (6% of steps) then linear decay.
- **Precision**: bf16 autocast via 🤗 Accelerate, with gradient
  checkpointing enabled — needed to fit the 3.5B frozen backbone plus its
  ~640M-parameter full-rank embedding table.
- **Batching**: per-device batch size 8, gradient accumulation 4 (effective
  batch size 32); variable option counts per example, same custom collator
  as -large (pads sequences and options separately, masks padding options
  to `-inf` before the softmax; per-example loss is soft cross-entropy
  against the full `target` distribution).
- **Gradient clipping**: max grad norm 1.0, enabled for the entire run
  (added to the training script after -large's mid-training divergence —
  see [-large's Training Procedure](MODEL_CARD_large.md#training-procedure)).
  It held: validation loss decreased from 0.81 (step 250) to 0.30 (step
  7,250) essentially monotonically across all 3 epochs, with no late-run
  spike.
- **Model selection**: checkpointed against the `validation` split every
  250 steps; `best` is step 7,250 of 7,419 (val_loss 0.3039, val top-1
  0.8528) — the final step's eval (val_loss 0.3026) was marginally lower
  but arrived after training had already ended, so it wasn't checkpointed
  as a new best. The two are close enough that this is effectively
  "trained to convergence" rather than an early stop.
- **Hardware**: a single H100 SXM rented on [RunPod](https://runpod.io?ref=5jrts9za) —
  doesn't fit the 16GB consumer GPU used for -large, since the frozen
  3.5B backbone plus its full-rank embedding table need real headroom
  (see `scripts/train_xl.sh`). Full run (3 epochs, 7,419 steps) took
  ~3h24m wall-clock.

## Evaluation

Evaluation of the published (`best`, step 7,250) checkpoint on the `test`
split only — `validation`/`ood`/`calibration` haven't been run for this
checkpoint yet, so they're left out rather than guessed at.

| split | n | cross-entropy | KL vs. target | top-1 acc. | ECE |
|---|---:|---:|---:|---:|---:|
| `test` | 10,356 | 0.229 | 0.199 | 0.893 | 0.015 |

**By `kind`** (test split):

| kind | n | cross-entropy | top-1 acc. |
|---|---:|---:|---:|
| `choice` | 2,408 | 0.481 | 0.757 |
| `noul` | 6,364 | 0.143 | 0.939 |
| `score` | 1,584 | 0.192 | 0.912 |

As with -large, `choice`'s top-1 accuracy is the lowest of the three kinds
(0.757 vs. 0.91+ for `noul`/`score`), but isn't directly comparable to
them: `choice` items carry up to 5–6 options against `noul`'s fixed 2, so
chance-level accuracy is much lower to begin with. See
[Compared to -large](#compared-to--large) for how these numbers stack up
against the smaller model.

## Compared to -large

A smaller variant, [**Mirave-0.6B-xlm-roberta-large**](MODEL_CARD_large.md),
exists on top of `FacebookAI/xlm-roberta-large` instead of
`xlm-roberta-xl` — same head, same prompt format, same training data,
~1/6th the backbone parameters. On the shared `test` split (n=10,356):

| | cross-entropy | KL vs. target | top-1 acc. | ECE |
|---|---:|---:|---:|---:|
| **-large** | 0.381 | 0.351 | 0.797 | 0.022 |
| **-xl** (this card) | 0.229 | 0.199 | 0.893 | 0.015 |

| kind | -large CE | -large top-1 | -xl CE | -xl top-1 |
|---|---:|---:|---:|---:|
| `choice` | 0.741 | 0.612 | 0.481 | 0.757 |
| `noul` | 0.271 | 0.855 | 0.143 | 0.939 |
| `score` | 0.278 | 0.850 | 0.192 | 0.912 |

-xl is meaningfully better across every kind, most notably on `choice` —
the kind -large's card flags as its weakest point. It also trained more
cleanly: -large's run diverged late in its third epoch (see -large's
[Training Procedure](MODEL_CARD_large.md#training-procedure)); -xl's did
not. The trade-off is size and cost: -xl's frozen backbone alone is ~7GB
in bf16 (~14GB in fp32) versus -large's ~1.1GB/~2.2GB, before adapter and
head weights, and it needs a real cloud GPU to train — an H100-class
instance on something like [RunPod](https://runpod.io?ref=5jrts9za) (see
[Training Procedure](#training-procedure) above) — rather than a single
16GB consumer card. Pick -large for cheaper inference and faster
iteration, -xl when the accuracy gap above matters more than that cost —
particularly for `choice`-heavy workloads.

## Limitations and Bias

- **Incomplete evaluation coverage.** Only the `test` split has been
  scored for this checkpoint so far — no `validation` (beyond the
  training-time model-selection numbers above), `ood`, or `calibration`
  numbers exist yet. In particular, -large's `ood` cross-entropy/ECE were
  substantially worse than its in-distribution numbers (confidence was
  overconfident on unseen phrasing); until -xl's `ood` split is evaluated,
  don't assume its `test`-split gains above transfer to that setting.
- **No ordinal inductive bias for `score`.** The pointer mechanism treats
  options as an unordered set — it has no explicit notion that option 3
  is "between" options 2 and 4 on a scale. A wrong `score` prediction has
  no architectural pressure to land *near* the true rating rather than
  far from it — worth checking directly (e.g. mean rank distance on
  errors) before relying on this for anything where a near-miss and a
  wild miss should be scored differently.
- **`choice` still lags the other two kinds**, even though it improved the
  most of the three over -large. 0.757 top-1 on `test` vs. 0.91+ for
  `noul`/`score` — not directly comparable given `choice` items carry up
  to 5–6 options against `noul`'s fixed 2, but still the kind to watch
  most closely for a given application.
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
- **Heavier to run than -large.** The frozen 3.5B backbone means
  meaningfully more VRAM, disk, and download size for a modest accuracy
  gain outside of `choice`-heavy use — see
  [Compared to -large](#compared-to--large) for the concrete trade-off.

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
| Base model | Not disclosed | `xlm-roberta-xl`, chosen specifically for being bidirectionally pretrained (see [How It Works](#how-it-works)) |
| Training | "Reinforcement Learning for Calibrated Decisions (RLCD)" | Supervised: LoRA fine-tuning against labeled target distributions (soft cross-entropy) |
| Availability | Closed | Open weights (adapter + head) and open training code |

The core bet carried over is architectural, not the training method: that
a model built to emit *only* the typed decision — rather than a token
stream that happens to contain one — can be both faster and more directly
usable by calling software than prompting a general chat model for the
same answer.
