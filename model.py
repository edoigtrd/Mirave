"""PointerHead architecture on top of FacebookAI/xlm-roberta-large.

A single forward pass produces a calibrated probability distribution over
a variable number of options, using a dot-product between a query
projected from the <s> (CLS-equivalent) hidden state and keys projected
from each option's </opt> hidden state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

BASE_MODEL = "FacebookAI/xlm-roberta-large"

# Generic per-role separators. <s> and </s> are already native to the
# XLM-R tokenizer and are not added here. <kind> gets its own tag pair
# (rather than being folded into the question text) since the model input
# must include `kind` alongside state/question/options.
SPECIAL_TOKENS = [
    "<kind>", "</kind>",
    "<state>", "</state>",
    "<question>", "</question>",
    "<opt>", "</opt>",
]

LORA_TARGET_MODULES = ["query", "key", "value", "dense"]
LORA_MODULES_TO_SAVE = ["word_embeddings"]

HEAD_WEIGHTS_FILE = "head.safetensors"
ADAPTER_SUBDIR = "adapter"
TOKENIZER_SUBDIR = "tokenizer"
CONFIG_FILE = "pointer_config.json"


def build_tokenizer(base_model: str = BASE_MODEL) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    # dataset.build_example truncates state/question itself to fit a chosen
    # max_length; this just silences the tokenizer's own length warning,
    # which fires on the pre-truncation encode() calls.
    tokenizer.model_max_length = int(1e9)
    return tokenizer


class PointerHead(nn.Module):
    """query = query_proj(h_cls); key_i = key_proj(h_opt_i); logit_i = query . key_i."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, h_cls: torch.Tensor, h_opts: torch.Tensor, opt_mask: torch.Tensor
    ) -> torch.Tensor:
        query = self.query_proj(h_cls)  # [B, H]
        keys = self.key_proj(h_opts)  # [B, K, H]
        logits = torch.einsum("bh,bkh->bk", query, keys)  # [B, K]
        # -1e9 rather than -inf: target is exactly 0 on masked slots, and
        # 0 * (-1e9) = 0, whereas 0 * -inf = nan.
        return logits.masked_fill(~opt_mask, -1e9)


@dataclass
class PointerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class PointerModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.head = PointerHead(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        opt_positions: torch.Tensor,
        opt_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_: object,
    ) -> PointerOutput:
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [B, T, H]

        h_cls = hidden[:, 0, :]
        idx = opt_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        h_opts = torch.gather(hidden, 1, idx)  # [B, K, H]

        logits = self.head(h_cls, h_opts, opt_mask)

        loss = None
        if labels is not None:
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(labels * log_probs).sum(dim=-1).mean()

        return PointerOutput(logits=logits, loss=loss)


def build_model(
    tokenizer: PreTrainedTokenizerBase,
    base_model: str = BASE_MODEL,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> PointerModel:
    backbone = AutoModel.from_pretrained(base_model)
    backbone.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=LORA_MODULES_TO_SAVE,
    )
    backbone = get_peft_model(backbone, lora_config)

    hidden_size = backbone.config.hidden_size
    return PointerModel(backbone, hidden_size)


def save_checkpoint(
    model: PointerModel, tokenizer: PreTrainedTokenizerBase, output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.backbone.save_pretrained(output_dir / ADAPTER_SUBDIR)
    tokenizer.save_pretrained(output_dir / TOKENIZER_SUBDIR)
    save_file(model.head.state_dict(), output_dir / HEAD_WEIGHTS_FILE)

    config = {
        "base_model": model.backbone.base_model.model.config._name_or_path,
        "hidden_size": model.backbone.config.hidden_size,
    }
    (output_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2))


def load_checkpoint(
    checkpoint_dir: str | Path, device: str | torch.device = "cpu"
) -> tuple[PointerModel, PreTrainedTokenizerBase]:
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / CONFIG_FILE).read_text())

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / TOKENIZER_SUBDIR)

    backbone = AutoModel.from_pretrained(config["base_model"])
    backbone.resize_token_embeddings(len(tokenizer))
    backbone = PeftModel.from_pretrained(backbone, checkpoint_dir / ADAPTER_SUBDIR)

    model = PointerModel(backbone, config["hidden_size"])
    head_state = load_file(checkpoint_dir / HEAD_WEIGHTS_FILE)
    model.head.load_state_dict(head_state)

    model.to(device)
    model.eval()
    return model, tokenizer
