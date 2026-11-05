"""SFT data loading and tokenization module for Project Second.

This module provides data loading, tokenization, and batch collation for Supervised Fine-Tuning.
It configures LoRA target modules and ensures answer-only loss masking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

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
class SFTConfigParams:
    """Parameters for Supervised Fine-Tuning."""

    model_name: str
    version: str
    size: str
    base_model: str
    text_only: bool
    enable_thinking: bool
    method: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str | list[str]
    train_data_path: str
    val_data_path: str
    testset_path: str
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
    grpo: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def sft(self) -> dict[str, Any]:
        """Return the SFT section of the raw configuration."""
        return self.raw_config.get("sft", {})

    @property
    def mode(self) -> dict[str, Any]:
        """Return the mode section of the raw configuration."""
        return self.raw_config.get("mode", {})


def _read_yaml_file(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file."""
    content = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(content)
    if isinstance(loaded, dict):
        return loaded
    return {}


def load_sft_config(
    config_path: str | Path = "training/configs/second-1.0-2b.yaml",
) -> SFTConfigParams:
    """Load and validate the Supervised Fine-Tuning configuration.

    If validation fails, raise ValueError.
    Return validated SFTConfigParams.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw = _read_yaml_file(path)
    mode = raw.get("mode", {})
    sft = raw.get("sft", {})
    grpo = raw.get("grpo", {})

    base_model = raw.get("base_model", "")
    text_only = mode.get("text_only")
    enable_thinking = mode.get("enable_thinking")

    lora_r = sft.get("lora_r")
    lora_alpha = sft.get("lora_alpha")
    target_modules = sft.get("target_modules")
    max_examples = sft.get("max_examples")
    num_train_epochs = sft.get("num_train_epochs")
    learning_rate = sft.get("learning_rate")
    lr_scheduler_type = sft.get("lr_scheduler_type")
    warmup_ratio = sft.get("warmup_ratio")
    max_seq_length = sft.get("max_seq_length")
    loss_on_answer_only = sft.get("loss_on_answer_only")
    save_steps = sft.get("save_steps")

    if base_model not in ALLOWED_BASE_MODELS:
        raise ValueError(
            f"Invalid base_model: expected one of {sorted(ALLOWED_BASE_MODELS)}, got '{base_model}'"
        )
    if text_only is not True:
        raise ValueError(f"Invalid text_only: expected True, got {text_only}")
    if enable_thinking is not False:
        raise ValueError(
            f"Invalid enable_thinking: expected False, got {enable_thinking}"
        )
    if not isinstance(lora_r, int) or lora_r <= 0:
        raise ValueError(f"Invalid lora_r: expected positive integer, got {lora_r}")
    if not isinstance(lora_alpha, int) or lora_alpha <= 0:
        raise ValueError(f"Invalid lora_alpha: expected positive integer, got {lora_alpha}")
    if target_modules != "all-linear":
        raise ValueError(
            f"Invalid target_modules: expected 'all-linear', got '{target_modules}'"
        )
    if not isinstance(max_examples, int) or max_examples > 20000 or max_examples <= 0:
        raise ValueError(
            f"Invalid max_examples: expected 1 <= max_examples <= 20000, got {max_examples}"
        )
    if not (isinstance(num_train_epochs, int) and 1 <= num_train_epochs <= 2):
        raise ValueError(
            f"Invalid num_train_epochs: expected 1 or 2, got {num_train_epochs}"
        )
    if learning_rate is None or not (0 < float(learning_rate) <= 0.01):
        raise ValueError(
            f"Invalid learning_rate: expected 0 < learning_rate <= 0.01, got {learning_rate}"
        )
    if lr_scheduler_type != "cosine":
        raise ValueError(
            f"Invalid lr_scheduler_type: expected 'cosine', got '{lr_scheduler_type}'"
        )
    if warmup_ratio is None or abs(float(warmup_ratio) - 0.03) > 1e-9:
        raise ValueError(
            f"Invalid warmup_ratio: expected 0.03, got {warmup_ratio}"
        )
    if max_seq_length != 1024:
        raise ValueError(
            f"Invalid max_seq_length: expected 1024, got {max_seq_length}"
        )
    if loss_on_answer_only is not True:
        raise ValueError(
            f"Invalid loss_on_answer_only: expected True, got {loss_on_answer_only}"
        )
    if save_steps != 200:
        raise ValueError(f"Invalid save_steps: expected 200, got {save_steps}")

    return SFTConfigParams(
        model_name=str(raw.get("model_name", "Second-1.0-2B")),
        version=str(raw.get("version", "1.0")),
        size=str(raw.get("size", "2B")),
        base_model=base_model,
        text_only=bool(text_only),
        enable_thinking=bool(enable_thinking),
        method=str(sft.get("method", "lora")),
        lora_r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(sft.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        train_data_path=str(sft.get("train_data_path", "data/processed/sft_train.jsonl")),
        val_data_path=str(sft.get("val_data_path", "data/processed/sft_val.jsonl")),
        testset_path=str(sft.get("testset_path", "eval/testset-1.0.jsonl")),
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
        save_steps=int(save_steps),
        eval_strategy=str(sft.get("eval_strategy", "steps")),
        eval_steps=int(sft.get("eval_steps", 200)),
        logging_steps=int(sft.get("logging_steps", 25)),
        bf16=bool(sft.get("bf16", True)),
        gpu=str(sft.get("gpu", "L4")),
        checkpoint_dir=str(sft.get("checkpoint_dir", "")),
        grpo=grpo,
        raw_config=raw,
    )


def build_non_thinking_chat_prompt(
    user_message: str,
    system_message: str = SYSTEM_MESSAGE,
) -> str:
    """Build the Qwen 3.5 non-thinking chat prompt.

    Return the formatted chat prompt with reasoning markers.
    """
    return (
        "<|im_start|>system\n"
        + system_message
        + "<|im_end|>\n<|im_start|>user\n"
        + user_message
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def load_sft_records(
    path: str | Path,
    max_examples: int = 20000,
) -> list[dict[str, Any]]:
    """Load JSONL records for Supervised Fine-Tuning.

    If path does not exist, raise FileNotFoundError.
    Cap the number of returned records at max_examples.
    Ensure each record contains chat_prompt and comment keys.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Record file not found: {path_obj}")

    records: list[dict[str, Any]] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)

            # Ensure comment or completion
            if "comment" not in rec or not rec["comment"]:
                if "completion" in rec and rec["completion"]:
                    rec["comment"] = rec["completion"]
                else:
                    raise ValueError("Record missing comment or completion.")
            if "completion" not in rec:
                rec["completion"] = rec["comment"]

            # Ensure chat_prompt
            if "chat_prompt" not in rec or not rec["chat_prompt"]:
                if "user_message" in rec and rec["user_message"]:
                    rec["chat_prompt"] = build_non_thinking_chat_prompt(
                        rec["user_message"]
                    )
                else:
                    raise ValueError("Record missing chat_prompt and user_message.")

            records.append(rec)
            if len(records) >= max_examples:
                break

    return records


def tokenize_sft_record(
    record: dict[str, Any],
    encode_fn: Callable[[str], list[int]],
    max_seq_length: int = 1024,
    eos_suffix: str = "<|im_end|>\n",
) -> dict[str, list[int]]:
    """Tokenize a single SFT record with answer-only loss labels.

    If prompt_text is not in record, build it from user_message.
    Tokenize prompt and answer separately.
    Mask prompt tokens in labels with -100.
    Truncate sequences to max_seq_length.
    """
    if "chat_prompt" in record and record["chat_prompt"]:
        prompt_text = record["chat_prompt"]
    elif "user_message" in record and record["user_message"]:
        prompt_text = build_non_thinking_chat_prompt(record["user_message"])
    else:
        raise ValueError("Record missing chat_prompt and user_message.")

    comment_val = record.get("comment") or record.get("completion")
    if comment_val is None:
        raise ValueError("Record missing comment or completion.")

    answer_text = str(comment_val).strip() + eos_suffix

    prompt_ids = encode_fn(prompt_text)
    answer_ids = encode_fn(answer_text)

    full_input_ids = prompt_ids + answer_ids
    full_labels = [-100] * len(prompt_ids) + answer_ids

    input_ids = full_input_ids[:max_seq_length]
    labels = full_labels[:max_seq_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def collate_sft_batch(
    features: list[dict[str, list[int]]],
    pad_token_id: int = 0,
    label_pad_token_id: int = -100,
) -> dict[str, list[list[int]]]:
    """Pad a batch of tokenized SFT records on the right.

    Find the maximum sequence length in the batch.
    Pad input_ids with pad_token_id.
    Pad attention_mask with 0.
    Pad labels with label_pad_token_id.
    """
    if not features:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

    max_len = max(len(feat["input_ids"]) for feat in features)

    batch_input_ids: list[list[int]] = []
    batch_attention_mask: list[list[int]] = []
    batch_labels: list[list[int]] = []

    for feat in features:
        curr_len = len(feat["input_ids"])
        pad_len = max_len - curr_len

        batch_input_ids.append(feat["input_ids"] + [pad_token_id] * pad_len)
        batch_attention_mask.append(feat["attention_mask"] + [0] * pad_len)
        batch_labels.append(feat["labels"] + [label_pad_token_id] * pad_len)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


class AnswerOnlyDataCollator:
    """Collate tokenized SFT features into padded batches.

    Pad input_ids, labels, and attention_mask to maximum sequence length.
    Convert lists to PyTorch tensors if available.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ) -> None:
        """Initialize collator settings."""
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, Any]:
        """Collate features and return padded batch."""
        batch = collate_sft_batch(
            features,
            pad_token_id=self.pad_token_id,
            label_pad_token_id=self.label_pad_token_id,
        )
        if self.return_tensors == "list":
            return batch

        try:
            import torch

            return {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in batch.items()
            }
        except ImportError:
            return batch


def resolve_text_linear_modules(
    named_modules: Iterable[tuple[str, Any]],
) -> list[str]:
    """Find unique leaf names for linear layers in the text model.

    Exclude modules from vision towers, embed tokens, and lm_head.
    Return sorted unique leaf module names.
    If no linear modules are found, return the default fallback list.
    """
    excluded_keywords = ("visual", "vision", "lm_head", "embed_tokens")
    found_leaves: set[str] = set()

    for name, module in named_modules:
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in excluded_keywords):
            continue

        module_type = type(module).__name__
        is_linear = module_type in {
            "Linear",
            "Linear4bit",
            "Linear8bitLt",
        }

        if not is_linear:
            continue

        leaf = name.split(".")[-1]
        if leaf:
            found_leaves.add(leaf)

    if not found_leaves:
        return list(DEFAULT_FALLBACK_LINEAR_MODULES)

    return sorted(found_leaves)
