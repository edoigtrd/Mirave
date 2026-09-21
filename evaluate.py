"""Evaluates a trained checkpoint on a MongoDB split.

`test`/`ood` labels are public, so any use of them for development
decisions must be disclosed — this script is for reporting final numbers,
not for tuning hyperparameters (use `validation` for that, as train.py
already does).

Reports overall soft cross-entropy / KL-vs-target and top-1 accuracy, a
per-`kind` breakdown (never collapsed into one blended number, since
`target` semantics differ by kind), and an Expected Calibration Error over
the head's max-softmax confidence.

    uv run evaluate.py --checkpoint checkpoints/xlmr-large-pointer/best --split test
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial

import torch
from torch.utils.data import DataLoader

from data import DB_NAME, mongo_client
from dataset import PointerJevDataset, collate
from model import load_checkpoint

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["validation", "test", "ood", "calibration"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def soft_target_entropy(labels: torch.Tensor) -> torch.Tensor:
    safe = torch.where(labels > 0, labels, torch.ones_like(labels))
    return -(labels * torch.log(safe)).sum(dim=-1)


def expected_calibration_error(
    confidences: torch.Tensor, correct: torch.Tensor, num_bins: int
) -> float:
    bin_edges = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = confidences.numel()
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum().item()
        if count == 0:
            continue
        avg_conf = confidences[in_bin].mean().item()
        avg_acc = correct[in_bin].float().mean().item()
        ece += (count / n) * abs(avg_acc - avg_conf)
    return ece


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model, tokenizer = load_checkpoint(args.checkpoint, device=args.device)

    client = mongo_client()
    db = client[DB_NAME]
    ds = PointerJevDataset(db[args.split], tokenizer, max_length=args.max_length)
    kinds = [row["kind"] for row in ds.rows]
    client.close()

    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    all_confidences, all_correct = [], []
    per_kind = defaultdict(lambda: {"ce": 0.0, "kl": 0.0, "correct": 0, "n": 0})
    cursor = 0

    for batch in loader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        out = model(**batch)

        log_probs = torch.log_softmax(out.logits, dim=-1)
        ce = -(batch["labels"] * log_probs).sum(dim=-1)
        kl = ce - soft_target_entropy(batch["labels"])

        probs = log_probs.exp()
        preds = probs.argmax(dim=-1)
        gold = batch["labels"].argmax(dim=-1)
        correct = preds == gold
        confidence = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

        all_confidences.append(confidence.cpu())
        all_correct.append(correct.cpu())

        bsz = batch["input_ids"].size(0)
        for i in range(bsz):
            k = kinds[cursor + i]
            bucket = per_kind[k]
            bucket["ce"] += ce[i].item()
            bucket["kl"] += kl[i].item()
            bucket["correct"] += int(correct[i].item())
            bucket["n"] += 1
        cursor += bsz

    all_confidences = torch.cat(all_confidences)
    all_correct = torch.cat(all_correct)
    ece = expected_calibration_error(all_confidences, all_correct, args.ece_bins)

    total_n = sum(b["n"] for b in per_kind.values())
    total_ce = sum(b["ce"] for b in per_kind.values())
    total_kl = sum(b["kl"] for b in per_kind.values())
    total_correct = sum(b["correct"] for b in per_kind.values())

    print(f"split={args.split}  n={total_n}")
    print(f"overall: cross_entropy={total_ce / total_n:.4f}  kl_vs_target={total_kl / total_n:.4f}  "
          f"top1_acc={total_correct / total_n:.4f}  ECE={ece:.4f}")
    print()
    print(f"{'kind':<20}{'n':>8}{'cross_entropy':>16}{'kl_vs_target':>16}{'top1_acc':>12}")
    for k, b in sorted(per_kind.items()):
        print(
            f"{k:<20}{b['n']:>8}{b['ce'] / b['n']:>16.4f}{b['kl'] / b['n']:>16.4f}"
            f"{b['correct'] / b['n']:>12.4f}"
        )


if __name__ == "__main__":
    main()
