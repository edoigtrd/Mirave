"""Prompt building and MongoDB-backed dataset for the pointer head.

Sequence layout:

    <s> <kind> ... </kind> <state> ... </state> <question> ... </question>
    <opt> ... </opt> <opt> ... </opt> ... </s>

`target` is never tokenized or placed in the sequence — it must never
reach the model as input, only as the training label.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    # Only needed for the PointerJevDataset type hint below — kept optional
    # so callers that just want build_example()/collate() (e.g. the
    # inference server) don't need pymongo installed.
    from pymongo.collection import Collection

# Options in this dataset are short labels (intents, answer choices, control
# actions). This cap is a safety net against a pathological long option,
# not an expected path.
MAX_OPTION_TOKENS = 128


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def build_example(
    tokenizer: PreTrainedTokenizerBase,
    state: Any,
    question: str,
    kind: str,
    options: list[str],
    target: list[float] | None = None,
    max_length: int = 512,
) -> dict[str, Any]:
    kind_open, kind_close, state_open, state_close, question_open, question_close, opt_open, opt_close = (
        tokenizer.convert_tokens_to_ids(
            [
                "<kind>", "</kind>",
                "<state>", "</state>",
                "<question>", "</question>",
                "<opt>", "</opt>",
            ]
        )
    )
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    kind_ids = tokenizer.encode(_as_text(kind), add_special_tokens=False)
    state_ids = tokenizer.encode(_as_text(state), add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    opt_ids_list = [
        tokenizer.encode(opt, add_special_tokens=False)[:MAX_OPTION_TOKENS]
        for opt in options
    ]

    # Fixed cost: cls + sep + one open/close pair per tag (kind, state,
    # question, each option) + the option tokens themselves. Options are
    # never truncated to zero below — the pointer mechanism needs an
    # intact </opt> per option no matter what.
    fixed = (
        2  # cls, sep
        + 2 + len(kind_ids)  # <kind>...</kind>
        + 2  # <state></state> tags (content added back below)
        + 2  # <question></question> tags (content added back below)
        + sum(2 + len(ids) for ids in opt_ids_list)  # <opt>...</opt> per option
    )
    budget = max_length - fixed
    if budget < 0:
        raise ValueError(
            f"{len(options)} options do not fit in max_length={max_length} "
            "even with empty state/question"
        )

    if len(question_ids) > budget:
        question_ids = question_ids[:budget]
        state_ids = []
    else:
        state_ids = state_ids[: budget - len(question_ids)]

    input_ids = [
        cls_id,
        kind_open, *kind_ids, kind_close,
        state_open, *state_ids, state_close,
        question_open, *question_ids, question_close,
    ]
    opt_positions = []
    for opt_ids in opt_ids_list:
        input_ids.append(opt_open)
        input_ids.extend(opt_ids)
        input_ids.append(opt_close)
        opt_positions.append(len(input_ids) - 1)
    input_ids.append(sep_id)

    assert len(input_ids) <= max_length

    example: dict[str, Any] = {
        "input_ids": input_ids,
        "opt_positions": opt_positions,
    }
    if target is not None:
        example["target"] = list(target)
    return example


class PointerJevDataset(Dataset):
    """Loads a MongoDB split collection and tokenizes on the fly.

    `target` is read here only to be used as a training label;
    `metadata`/`id`/`group_id`/`split`/`source` are never read at all.
    """

    def __init__(
        self,
        collection: Collection,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        kind: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        query = {"kind": kind} if kind else {}
        projection = {
            "_id": 0,
            "state": 1,
            "question": 1,
            "kind": 1,
            "options": 1,
            "target": 1,
        }
        self.rows = list(collection.find(query, projection))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return build_example(
            self.tokenizer,
            state=row["state"],
            question=row["question"],
            kind=row["kind"],
            options=row["options"],
            target=row["target"],
            max_length=self.max_length,
        )


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    max_k = max(len(b["opt_positions"]) for b in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    opt_positions = torch.zeros((batch_size, max_k), dtype=torch.long)
    opt_mask = torch.zeros((batch_size, max_k), dtype=torch.bool)
    labels = torch.zeros((batch_size, max_k), dtype=torch.float)

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        num_opts = len(item["opt_positions"])
        opt_positions[i, :num_opts] = torch.tensor(item["opt_positions"], dtype=torch.long)
        opt_mask[i, :num_opts] = True

        if "target" in item:
            labels[i, :num_opts] = torch.tensor(item["target"], dtype=torch.float)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "opt_positions": opt_positions,
        "opt_mask": opt_mask,
        "labels": labels,
    }
