"""Runs the trained pointer head on new (state, question, kind, options).

Single example:
    uv run infer.py --checkpoint checkpoints/xlmr-large-pointer/best \\
        --kind choice --state "..." --question "..." \\
        --options "option a" "option b" "option c"

Batch (JSON Lines, one {"state":..., "question":..., "kind":..., "options":[...]} per line):
    uv run infer.py --checkpoint checkpoints/xlmr-large-pointer/best --jsonl examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from dataset import build_example, collate
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--jsonl", help="path to a JSONL file of examples; omit for a single CLI example")
    p.add_argument("--state", default="")
    p.add_argument("--question", default="")
    p.add_argument("--kind", default="choice")
    p.add_argument("--options", nargs="+", default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_examples(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.jsonl:
        with open(args.jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        if not args.options:
            raise SystemExit("--options is required when not using --jsonl")
        yield {
            "state": args.state,
            "question": args.question,
            "kind": args.kind,
            "options": args.options,
        }


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    tokenizer,
    examples: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    tokenized = [
        build_example(
            tokenizer,
            state=ex["state"],
            question=ex["question"],
            kind=ex["kind"],
            options=ex["options"],
            target=None,
            max_length=max_length,
        )
        for ex in examples
    ]
    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    results = []
    cursor = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch, labels=None).logits
        probs = torch.softmax(logits.masked_fill(~batch["opt_mask"], float("-inf")), dim=-1)

        for i in range(probs.size(0)):
            ex = examples[cursor]
            num_opts = len(ex["options"])
            option_probs = probs[i, :num_opts].tolist()
            results.append(
                {
                    "options": ex["options"],
                    "probs": option_probs,
                    "argmax": ex["options"][int(torch.tensor(option_probs).argmax())],
                }
            )
            cursor += 1
    return results


def main() -> None:
    args = parse_args()
    model, tokenizer = load_checkpoint(args.checkpoint, device=args.device)

    examples = list(load_examples(args))
    results = run_inference(
        model, tokenizer, examples, args.max_length, args.batch_size, args.device
    )

    for result in results:
        json.dump(result, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
