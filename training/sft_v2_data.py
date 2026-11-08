"""SFT v2 data loading and tokenization module for Stage T3.

This module provides data loading, tokenization, and batch collation for SFT v2.
It supports LoRA adapters and full parameter fine-tuning.
It configures answer-only or dual-mode loss masking.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Callable

import yaml

LOGGER = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE: str = (
    "You are Second, a chess assistant. "
    "You explain chess positions and moves clearly using the provided facts."
)

try:
    from core.python.prompt import SYSTEM_MESSAGE
except ImportError:
    SYSTEM_MESSAGE = DEFAULT_SYSTEM_MESSAGE

DEFAULT_FALLBACK_LINEAR_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

ALLOWED_BASE_MODELS: set[str] = {
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
}


@dataclass
class SFTv2ConfigParams:
    """Parameters for Stage T3 Supervised Fine-Tuning v2."""

    model_name: str
    version: str
    size: str
    base_model: str
    text_only: bool
    enable_thinking: bool
    method: str
    lora_r: int | None
    lora_alpha: int | None
    lora_dropout: float
    target_modules: str | list[str]
    train_data_path: str
    val_data_path: str
    devset_path: str
    max_examples: int
    num_train_epochs: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    max_seq_length: int
    loss_on_answer_only: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    save_strategy: str
    save_steps: int
    eval_strategy: str
    eval_steps: int
    logging_steps: int
    bf16: bool
    gpu: str
    checkpoint_dir: str
    optimizer: str = "adamw_torch"
    raw_config: dict[str, Any] = field(default_factory=dict)


def _read_yaml_file(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file."""
    content = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(content)
    if isinstance(loaded, dict):
        return loaded
    return {}


def load_sft_v2_config(
    config_path: str | Path = "training/configs/sft_v2_a1_lora32.yaml",
) -> SFTv2ConfigParams:
    """Load and validate the SFT v2 recipe configuration.

    If validation fails, raise ValueError.
    Return validated SFTv2ConfigParams instance.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw = _read_yaml_file(path)
    mode = raw.get("mode", {})
    sft = raw.get("sft", {})

    base_model = raw.get("base_model", "")
    text_only = mode.get("text_only", True)
    enable_thinking = mode.get("enable_thinking", False)
    method = str(sft.get("method", "lora")).lower()

    if base_model not in ALLOWED_BASE_MODELS:
        raise ValueError(
            f"Invalid base_model: expected one of {sorted(ALLOWED_BASE_MODELS)}, got '{base_model}'"
        )
    if method not in ("lora", "full"):
        raise ValueError(f"Invalid method: expected 'lora' or 'full', got '{method}'")

    lora_r = sft.get("lora_r")
    lora_alpha = sft.get("lora_alpha")
    target_modules = sft.get("target_modules", "all-linear")

    if method == "lora":
        if not isinstance(lora_r, int) or lora_r <= 0:
            raise ValueError(f"Invalid lora_r: expected positive integer, got {lora_r}")
        if not isinstance(lora_alpha, int) or lora_alpha <= 0:
            raise ValueError(f"Invalid lora_alpha: expected positive integer, got {lora_alpha}")

    max_examples = sft.get("max_examples", 10000)
    num_train_epochs = sft.get("num_train_epochs", 2)
    learning_rate = sft.get("learning_rate", 0.0001)
    lr_scheduler_type = sft.get("lr_scheduler_type", "cosine")
    warmup_ratio = sft.get("warmup_ratio", 0.03)
    max_seq_length = sft.get("max_seq_length", 1024)
    loss_on_answer_only = sft.get("loss_on_answer_only", True)

    train_data = sft.get("train_data_path", "data/processed/sft_v2_train.jsonl")
    val_data = sft.get("val_data_path", "data/processed/sft_val.jsonl")
    devset = sft.get("devset_path", "eval/devset-1.0.jsonl")

    # Protection rule: devset must never point to testset
    if "testset" in devset.lower():
        raise ValueError("Frozen test set must never be used as devset in SFT v2.")

    return SFTv2ConfigParams(
        model_name=str(raw.get("model_name", "Second-1.0-2B-SFT-v2")),
        version=str(raw.get("version", "2.0")),
        size=str(raw.get("size", "2B")),
        base_model=base_model,
        text_only=bool(text_only),
        enable_thinking=bool(enable_thinking),
        method=method,
        lora_r=int(lora_r) if lora_r is not None else None,
        lora_alpha=int(lora_alpha) if lora_alpha is not None else None,
        lora_dropout=float(sft.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        train_data_path=str(train_data),
        val_data_path=str(val_data),
        devset_path=str(devset),
        max_examples=int(max_examples),
        num_train_epochs=int(num_train_epochs),
        learning_rate=float(learning_rate),
        lr_scheduler_type=str(lr_scheduler_type),
        warmup_ratio=float(warmup_ratio),
        max_seq_length=int(max_seq_length),
        loss_on_answer_only=bool(loss_on_answer_only),
        per_device_train_batch_size=int(sft.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(sft.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(sft.get("gradient_accumulation_steps", 2)),
        save_strategy=str(sft.get("save_strategy", "steps")),
        save_steps=int(sft.get("save_steps", 200)),
        eval_strategy=str(sft.get("eval_strategy", "steps")),
        eval_steps=int(sft.get("eval_steps", 200)),
        logging_steps=int(sft.get("logging_steps", 25)),
        bf16=bool(sft.get("bf16", True)),
        gpu=str(sft.get("gpu", "L4")),
        checkpoint_dir=str(sft.get("checkpoint_dir", "")),
        optimizer=str(sft.get("optimizer", "adamw_torch")),
        raw_config=raw,
    )


def build_sft_v2_prompt(
    user_message: str,
    thinking_mode: str = "off",
    system_message: str = SYSTEM_MESSAGE,
) -> str:
    """Build Qwen 3.5 prompt for thinking or non-thinking training.

    If thinking_mode is off, prompt ends after empty think tags.
    If thinking_mode is standard or quick, prompt ends after assistant start tag.
    Return formatted prompt string.
    """
    base = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if thinking_mode == "off":
        return base + "<think>\n\n</think>\n\n"
    return base


def load_sft_v2_records(
    path: str | Path,
    max_examples: int = 10000,
) -> list[dict[str, Any]]:
    """Load JSONL training records for SFT v2.

    If path does not exist, raise FileNotFoundError.
    Cap the returned records at max_examples.
    Return list of validated record dictionaries.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Record file not found: {path_obj}")

    records: list[dict[str, Any]] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)

            # Ensure user message
            user_msg = rec.get("user_message") or ""

            # Determine thinking mode
            thinking_mode = rec.get("thinking_mode", "off")

            # Ensure chat_prompt
            if not rec.get("chat_prompt"):
                rec["chat_prompt"] = build_sft_v2_prompt(
                    user_message=user_msg,
                    thinking_mode=thinking_mode,
                )

            # Ensure assistant response or comment
            if not rec.get("assistant_response"):
                comment = rec.get("comment") or rec.get("explanation") or ""
                if thinking_mode == "off":
                    rec["assistant_response"] = f"<think>\n\n</think>\n\n{comment}"
                else:
                    trace = rec.get("trace", "")
                    rec["assistant_response"] = f"<think>\n{trace}\n</think>\n\n{comment}"

            if not rec.get("comment"):
                rec["comment"] = rec.get("explanation") or rec["assistant_response"]

            records.append(rec)
            if len(records) >= max_examples:
                break

    LOGGER.info("Loaded %d SFT v2 records from %s", len(records), path_obj)
    return records


def tokenize_sft_v2_record(
    record: dict[str, Any],
    encode_fn: Callable[[str], list[int]],
    max_seq_length: int = 1024,
    loss_on_answer_only: bool = True,
    eos_token: str = "<|im_end|>\n",
) -> dict[str, list[int]]:
    """Tokenize an SFT v2 record with prompt masking.

    If loss_on_answer_only is True, mask prompt tokens in labels with -100.
    Tokenize assistant response and append eos_token.
    Truncate total tokens to max_seq_length.
    Return input_ids, labels, and attention_mask.
    """
    prompt_text = record.get("chat_prompt", "")
    thinking_mode = record.get("thinking_mode", "off")

    if loss_on_answer_only or thinking_mode == "off":
        prompt_ids = encode_fn(prompt_text)
        answer_text = (record.get("comment") or record.get("explanation") or "") + eos_token
        answer_ids = encode_fn(answer_text)

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)
    else:
        # Dual-mode thinking: train on full assistant response (<think> trace + answer)
        prompt_ids = encode_fn(prompt_text)
        resp_text = (record.get("assistant_response") or "") + eos_token
        resp_ids = encode_fn(resp_text)

        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + list(resp_ids)

    # Truncate to max_seq_length
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class SFTv2DataCollator:
    """Collate tokenized sequences into padded batches for PyTorch training."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100) -> None:
        """Initialize pad token identifiers."""
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        """Pad input sequences and return tensor batch dictionary."""
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            batch_labels.append(item["labels"] + [self.label_pad_token_id] * pad_len)
            batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)

        try:
            import torch

            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            }
        except ImportError:
            return {
                "input_ids": batch_input_ids,
                "labels": batch_labels,
                "attention_mask": batch_attention_mask,
            }
