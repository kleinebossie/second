"""Unit tests for training configuration and SFT data processing.

This module tests configuration loading, prompt building, tokenization, and batch collation.
All docstrings follow ASD-STE100 rules.
"""

from pathlib import Path
from typing import Any

import pytest

from training.sft_data import (
    AnswerOnlyDataCollator,
    SFTConfigParams,
    build_non_thinking_chat_prompt,
    collate_sft_batch,
    load_sft_config,
    load_sft_records,
    resolve_text_linear_modules,
    tokenize_sft_record,
)


class DummyLinear:
    """Mock linear module for testing."""

    def __init__(self, in_features: int = 64, out_features: int = 64) -> None:
        """Initialize mock layer dimensions."""
        self.in_features = in_features
        self.out_features = out_features


class DummyEmbedding:
    """Mock embedding module for testing."""

    def __init__(self, in_features: int = 64, out_features: int = 64) -> None:
        """Initialize mock layer dimensions."""
        self.in_features = in_features
        self.out_features = out_features


def dummy_encoder(text: str) -> list[int]:
    """Encode text to token identifiers using character code offsets."""
    return [ord(c) % 500 + 1 for c in text]


def test_load_default_sft_config() -> None:
    """Test loading and validating the default 2B configuration."""
    config_path = Path("training/configs/second-1.0-2b.yaml")
    params = load_sft_config(config_path)

    assert isinstance(params, SFTConfigParams)
    assert params.model_name == "Second-1.0-2B"
    assert params.version == "1.0"
    assert params.size == "2B"
    assert params.base_model == "Qwen/Qwen3.5-2B"
    assert params.text_only is True
    assert params.enable_thinking is False
    assert params.method == "lora"
    assert params.lora_r == 32
    assert params.lora_alpha == 32
    assert params.lora_dropout == 0.05
    assert params.target_modules == "all-linear"
    assert params.train_data_path == "data/processed/sft_train.jsonl"
    assert params.val_data_path == "data/processed/sft_val.jsonl"
    assert params.testset_path == "eval/testset-1.0.jsonl"
    assert params.max_examples == 20000
    assert params.num_train_epochs == 2
    assert params.learning_rate == 0.0001
    assert params.lr_scheduler_type == "cosine"
    assert params.warmup_ratio == 0.03
    assert params.max_seq_length == 1024
    assert params.loss_on_answer_only is True
    assert params.per_device_train_batch_size == 4
    assert params.per_device_eval_batch_size == 4
    assert params.gradient_accumulation_steps == 2
    assert params.save_strategy == "steps"
    assert params.save_steps == 200
    assert params.eval_strategy == "steps"
    assert params.eval_steps == 200
    assert params.logging_steps == 25
    assert params.bf16 is True
    assert params.gpu == "L4"
    assert params.checkpoint_dir == "/content/drive/MyDrive/second/checkpoints/sft-1.0-2b"

    # Verify GRPO configuration section
    assert params.grpo.get("num_generations") == 8
    assert params.grpo.get("max_completion_length") == 300
    assert params.grpo.get("temperature") == 0.9
    assert params.grpo.get("learning_rate") == 0.000005
    assert params.grpo.get("beta") == 0.02
    assert params.grpo.get("max_steps") == 500
    assert params.grpo.get("save_steps") == 50
    assert params.grpo.get("gpu") == "L4"


@pytest.mark.parametrize(
    ("override_key", "override_value", "error_match"),
    [
        ("base_model", "meta-llama/Llama-3", "Invalid base_model"),
        ("text_only", False, "Invalid text_only"),
        ("enable_thinking", True, "Invalid enable_thinking"),
        ("lora_r", 16, "Invalid lora_r"),
        ("lora_alpha", 16, "Invalid lora_alpha"),
        ("target_modules", "q_proj", "Invalid target_modules"),
        ("max_examples", 25000, "Invalid max_examples"),
        ("num_train_epochs", 3, "Invalid num_train_epochs"),
        ("learning_rate", 0.001, "Invalid learning_rate"),
        ("lr_scheduler_type", "linear", "Invalid lr_scheduler_type"),
        ("warmup_ratio", 0.1, "Invalid warmup_ratio"),
        ("max_seq_length", 512, "Invalid max_seq_length"),
        ("loss_on_answer_only", False, "Invalid loss_on_answer_only"),
        ("save_steps", 100, "Invalid save_steps"),
    ],
)
def test_load_sft_config_validations(
    tmp_path: Path,
    override_key: str,
    override_value: Any,
    error_match: str,
) -> None:
    """Test validation errors for invalid configuration values."""
    config_content = (
        'model_name: "Second-1.0-2B"\n'
        'version: "1.0"\n'
        'size: "2B"\n'
        'base_model: "Qwen/Qwen3.5-2B"\n'
        "mode:\n"
        "  text_only: true\n"
        "  enable_thinking: false\n"
        "sft:\n"
        '  method: "lora"\n'
        "  lora_r: 32\n"
        "  lora_alpha: 32\n"
        "  lora_dropout: 0.05\n"
        '  target_modules: "all-linear"\n'
        '  train_data_path: "data/processed/sft_train.jsonl"\n'
        '  val_data_path: "data/processed/sft_val.jsonl"\n'
        '  testset_path: "eval/testset-1.0.jsonl"\n'
        "  max_examples: 20000\n"
        "  num_train_epochs: 2\n"
        "  learning_rate: 0.0001\n"
        '  lr_scheduler_type: "cosine"\n'
        "  warmup_ratio: 0.03\n"
        "  max_seq_length: 1024\n"
        "  loss_on_answer_only: true\n"
        "  per_device_train_batch_size: 4\n"
        "  per_device_eval_batch_size: 4\n"
        "  gradient_accumulation_steps: 2\n"
        '  save_strategy: "steps"\n'
        "  save_steps: 200\n"
        '  eval_strategy: "steps"\n'
        "  eval_steps: 200\n"
        "  logging_steps: 25\n"
        "  bf16: true\n"
        '  gpu: "L4"\n'
        '  checkpoint_dir: "/content/drive/MyDrive/second/checkpoints/sft-1.0-2b"\n'
    )

    # Apply override
    if override_key in ("text_only", "enable_thinking"):
        old_line = f"  {override_key}: {'true' if override_key == 'text_only' else 'false'}"
        new_val_str = "true" if override_value else "false"
        new_line = f"  {override_key}: {new_val_str}"
        config_content = config_content.replace(old_line, new_line)
    elif override_key == "base_model":
        config_content = config_content.replace(
            'base_model: "Qwen/Qwen3.5-2B"', f'base_model: "{override_value}"'
        )
    else:
        # Field in sft section
        for line in config_content.splitlines():
            if line.strip().startswith(f"{override_key}:"):
                val_repr = (
                    f'"{override_value}"'
                    if isinstance(override_value, str)
                    else (
                        "true"
                        if override_value is True
                        else ("false" if override_value is False else str(override_value))
                    )
                )
                config_content = config_content.replace(
                    line, f"  {override_key}: {val_repr}"
                )
                break

    temp_file = tmp_path / "test_config.yaml"
    temp_file.write_text(config_content, encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        load_sft_config(temp_file)


def test_build_non_thinking_chat_prompt() -> None:
    """Test non-thinking prompt construction with Qwen markers."""
    user_msg = "task: explain_move\nFEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    prompt = build_non_thinking_chat_prompt(user_msg)

    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_end|>\n<|im_start|>user\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert user_msg in prompt


def test_load_real_sft_train_records() -> None:
    """Test loading first 50 records from processed SFT train dataset."""
    train_path = Path("data/processed/sft_train.jsonl")
    records = load_sft_records(train_path, max_examples=50)

    assert len(records) == 50
    for rec in records:
        assert "chat_prompt" in rec
        assert "comment" in rec
        assert rec["chat_prompt"].startswith("<|im_start|>system\n")
        assert "<think>\n\n</think>\n\n" in rec["chat_prompt"]
        assert len(rec["comment"]) > 0


def test_load_real_sft_val_records() -> None:
    """Test loading all 137 records from processed SFT validation dataset."""
    val_path = Path("data/processed/sft_val.jsonl")
    records = load_sft_records(val_path, max_examples=20000)

    assert len(records) == 137
    for rec in records:
        assert "chat_prompt" in rec
        assert "comment" in rec
        assert rec["chat_prompt"].startswith("<|im_start|>system\n")
        assert "<think>\n\n</think>\n\n" in rec["chat_prompt"]


def test_load_sft_records_missing_chat_prompt(tmp_path: Path) -> None:
    """Test automatic chat_prompt construction when only user_message exists."""
    sample_file = tmp_path / "sample.jsonl"
    line = (
        '{"user_message": "task: explain_position\\nFEN: 8/8/8/8/8/8/8/8 w - - 0 1", '
        '"comment": "The position is empty."}\n'
    )
    sample_file.write_text(line, encoding="utf-8")

    records = load_sft_records(sample_file, max_examples=10)
    assert len(records) == 1
    assert "chat_prompt" in records[0]
    assert records[0]["chat_prompt"].startswith("<|im_start|>system\n")
    assert "The position is empty." == records[0]["comment"]


def test_tokenize_sft_record_masks_prompt_tokens() -> None:
    """Test that labels mask all prompt tokens with -100."""
    record = {
        "chat_prompt": "<|im_start|>system\nTest<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "comment": "Good move.",
    }

    result = tokenize_sft_record(record, encode_fn=dummy_encoder, max_seq_length=1024)

    input_ids = result["input_ids"]
    labels = result["labels"]
    attention_mask = result["attention_mask"]

    prompt_ids = dummy_encoder(record["chat_prompt"])
    answer_ids = dummy_encoder("Good move.<|im_end|>\n")

    assert len(input_ids) == len(prompt_ids) + len(answer_ids)
    assert len(labels) == len(input_ids)
    assert len(attention_mask) == len(input_ids)

    # 100% of prompt tokens must have label -100
    prompt_len = len(prompt_ids)
    assert labels[:prompt_len] == [-100] * prompt_len

    # All answer tokens must be preserved in labels
    assert labels[prompt_len:] == answer_ids
    assert input_ids == prompt_ids + answer_ids
    assert attention_mask == [1] * len(input_ids)


def test_tokenize_sft_record_truncation() -> None:
    """Test sequence truncation when length exceeds max_seq_length."""
    record = {
        "chat_prompt": "Prompt text for truncation testing.",
        "comment": "Answer text that gets truncated.",
    }
    max_len = 25
    result = tokenize_sft_record(
        record,
        encode_fn=dummy_encoder,
        max_seq_length=max_len,
    )

    assert len(result["input_ids"]) == max_len
    assert len(result["labels"]) == max_len
    assert len(result["attention_mask"]) == max_len


def test_collate_sft_batch() -> None:
    """Test right-padding of input_ids, attention_mask, and labels."""
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        {"input_ids": [6, 7, 8, 9], "attention_mask": [1, 1, 1, 1], "labels": [-100, 7, 8, 9]},
    ]

    batch = collate_sft_batch(features, pad_token_id=0, label_pad_token_id=-100)

    assert batch["input_ids"] == [
        [1, 2, 3, 0],
        [4, 5, 0, 0],
        [6, 7, 8, 9],
    ]
    assert batch["attention_mask"] == [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]
    assert batch["labels"] == [
        [-100, -100, 3, -100],
        [-100, 5, -100, -100],
        [-100, 7, 8, 9],
    ]


def test_answer_only_data_collator() -> None:
    """Test AnswerOnlyDataCollator with list return mode."""
    collator = AnswerOnlyDataCollator(return_tensors="list")
    features = [
        {"input_ids": [10, 20], "attention_mask": [1, 1], "labels": [-100, 20]},
        {"input_ids": [30], "attention_mask": [1], "labels": [30]},
    ]
    batch = collator(features)
    assert batch["input_ids"] == [[10, 20], [30, 0]]
    assert batch["attention_mask"] == [[1, 1], [1, 0]]
    assert batch["labels"] == [[-100, 20], [30, -100]]


def test_resolve_text_linear_modules() -> None:
    """Test resolving linear leaf modules and excluding non-text layers."""
    named_modules = [
        ("model.layers.0.self_attn.q_proj", DummyLinear()),
        ("model.layers.0.self_attn.k_proj", DummyLinear()),
        ("model.layers.0.self_attn.v_proj", DummyLinear()),
        ("model.layers.0.self_attn.o_proj", DummyLinear()),
        ("model.layers.0.mlp.gate_proj", DummyLinear()),
        ("model.layers.0.mlp.up_proj", DummyLinear()),
        ("model.layers.0.mlp.down_proj", DummyLinear()),
        # Gated DeltaNet linear projections
        ("model.layers.1.linear_attn.in_proj", DummyLinear()),
        ("model.layers.1.linear_attn.out_proj", DummyLinear()),
        # Excluded modules
        ("lm_head", DummyLinear()),
        ("visual.patch_embed.proj", DummyLinear()),
        ("vision_tower.encoder.linear", DummyLinear()),
        ("model.embed_tokens", DummyEmbedding()),
    ]

    result = resolve_text_linear_modules(named_modules)

    expected = sorted([
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj",
        "out_proj",
    ])
    assert result == expected
    assert "lm_head" not in result
    assert "proj" not in result
    assert "linear" not in result
    assert "embed_tokens" not in result


def test_resolve_text_linear_modules_fallback() -> None:
    """Test default fallback when no linear modules are present."""
    result = resolve_text_linear_modules([])
    assert result == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
