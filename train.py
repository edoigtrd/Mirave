"""Fine-tunes the PointerHead on top of xlm-roberta-large with LoRA.

Trains on the `train` collection only, model-selects against `validation`
— `calibration`/`test`/`ood` are never touched here.

Every run writes both to stdout and to a timestamped .log file under
--log-dir, so a lost terminal (e.g. an accidental Ctrl+C) doesn't lose the
run's history. Ctrl+C itself is caught: it saves an `interrupted/`
checkpoint before exiting instead of just dying.

    uv run train.py --output-dir checkpoints/xlmr-large-pointer
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from data import DB_NAME, mongo_client
from dataset import PointerJevDataset, collate
from model import BASE_MODEL, build_model, build_tokenizer, save_checkpoint

logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # huggingface_hub/httpx log one INFO line per HTTP HEAD/GET request
    # (cache checks etc.) — noisy and not useful in the training log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logger.info("logging to %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="checkpoints/xlmr-large-pointer")
    p.add_argument("--log-dir", default="logs")
    p.add_argument(
        "--base-model", default=BASE_MODEL,
        help="e.g. facebook/xlm-roberta-xl or facebook/xlm-roberta-xxl for the larger variants",
    )
    p.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="trade compute for activation memory — needed for xl/xxl, not for -large",
    )
    p.add_argument("--kind", default=None, help="restrict training to one `kind` (default: all)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4, help="LoRA adapter learning rate")
    p.add_argument(
        "--embedding-lr", type=float, default=None,
        help="learning rate for the fully-trainable word_embeddings table "
        "(default: --lr / 10). Kept separate from --lr because the embedding "
        "table is full-parameter fine-tuning of 256M already-well-trained "
        "weights, not a small LoRA adapter — the same LR for both is what "
        "produced the unstable warmup-phase loss spikes seen in practice.",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="gradient clipping threshold; guards against the kind of late-training "
        "loss divergence seen without it (loss jumping to a fixed high value and "
        "staying there — a saturated/blown-up weight, not normal noise)",
    )
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=500, help="optimizer steps between validation passes")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.embedding_lr is None:
        args.embedding_lr = args.lr / 10
    return args


def build_optimizer(model: torch.nn.Module, lr: float, embedding_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    embedding_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (embedding_params if "word_embeddings" in name else other_params).append(param)

    groups = [
        {"params": other_params, "lr": lr},
        {"params": embedding_params, "lr": embedding_lr},
    ]
    logger.info(
        "optimizer groups: %d LoRA/head params @ lr=%.1e, %d embedding params @ lr=%.1e",
        len(other_params), lr, len(embedding_params), embedding_lr,
    )
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def evaluate_loop(
    model: torch.nn.Module, loader: DataLoader, kinds: list[str] | None = None
) -> tuple[float, float, dict[str, tuple[float, float, int]]]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    per_kind: dict[str, list[float]] = {}
    cursor = 0

    for batch in loader:
        out = model(**batch)
        bsz = batch["input_ids"].size(0)
        preds = out.logits.argmax(dim=-1)
        gold = batch["labels"].argmax(dim=-1)
        correct = preds == gold

        total_loss += out.loss.item() * bsz
        total_correct += correct.sum().item()
        total_count += bsz

        if kinds is not None:
            log_probs = torch.log_softmax(out.logits, dim=-1)
            per_example_loss = -(batch["labels"] * log_probs).sum(dim=-1)
            for i in range(bsz):
                k = kinds[cursor + i]
                bucket = per_kind.setdefault(k, [0.0, 0, 0])  # loss_sum, correct, n
                bucket[0] += per_example_loss[i].item()
                bucket[1] += int(correct[i].item())
                bucket[2] += 1
            cursor += bsz

    model.train()
    breakdown = {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in per_kind.items()}
    return total_loss / total_count, total_correct / total_count, breakdown


def log_eval(step: int | str, val_loss: float, val_acc: float, breakdown: dict) -> None:
    logger.info("[eval] step %s val_loss %.4f val_top1 %.4f", step, val_loss, val_acc)
    for kind, (loss, acc, n) in sorted(breakdown.items()):
        logger.info("  [eval]   kind=%-10s n=%-6d loss=%.4f top1=%.4f", kind, n, loss, acc)


def main() -> None:
    args = parse_args()
    run_name = Path(args.output_dir).name
    setup_logging(Path(args.log_dir) / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    logger.info("args: %s", vars(args))

    set_seed(args.seed)

    client = mongo_client()
    db = client[DB_NAME]

    tokenizer = build_tokenizer(args.base_model)
    model = build_model(
        tokenizer,
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.backbone.print_trainable_parameters()

    train_ds = PointerJevDataset(db["train"], tokenizer, max_length=args.max_length, kind=args.kind)
    val_ds = PointerJevDataset(db["validation"], tokenizer, max_length=args.max_length, kind=args.kind)
    val_kinds = [row["kind"] for row in val_ds.rows]
    client.close()

    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    optimizer = build_optimizer(model, args.lr, args.embedding_lr, args.weight_decay)

    steps_per_epoch = -(-len(train_loader) // args.grad_accum)  # ceil div
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    output_dir = Path(args.output_dir)
    best_val_loss = float("inf")
    global_step = 0
    loss_window: list[float] = []

    try:
        for epoch in range(args.epochs):
            for batch in train_loader:
                with accelerator.accumulate(model):
                    out = model(**batch)
                    accelerator.backward(out.loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    loss_window.append(out.loss.item())

                    if global_step % args.log_every == 0:
                        avg_loss = sum(loss_window) / len(loss_window)
                        logger.info(
                            "epoch %d step %d/%d loss(avg over last %d)=%.4f lr=%.2e emb_lr=%.2e",
                            epoch, global_step, total_steps, len(loss_window),
                            avg_loss, scheduler.get_last_lr()[0], scheduler.get_last_lr()[1],
                        )
                        loss_window = []

                    if global_step % args.eval_every == 0:
                        val_loss, val_acc, breakdown = evaluate_loop(model, val_loader, val_kinds)
                        log_eval(global_step, val_loss, val_acc, breakdown)
                        if val_loss < best_val_loss and accelerator.is_main_process:
                            best_val_loss = val_loss
                            save_checkpoint(accelerator.unwrap_model(model), tokenizer, output_dir / "best")
                            logger.info("saved new best checkpoint (val_loss=%.4f) to %s", val_loss, output_dir / "best")
    except KeyboardInterrupt:
        logger.warning("interrupted at step %d — saving current state before exiting", global_step)
        if accelerator.is_main_process:
            save_checkpoint(accelerator.unwrap_model(model), tokenizer, output_dir / "interrupted")
            logger.info("saved interrupted checkpoint to %s", output_dir / "interrupted")
        raise

    val_loss, val_acc, breakdown = evaluate_loop(model, val_loader, val_kinds)
    logger.info("[final]")
    log_eval("final", val_loss, val_acc, breakdown)
    if accelerator.is_main_process:
        save_checkpoint(accelerator.unwrap_model(model), tokenizer, output_dir / "final")
        if val_loss < best_val_loss:
            save_checkpoint(accelerator.unwrap_model(model), tokenizer, output_dir / "best")


if __name__ == "__main__":
    main()
