"""Unit tests for Stage T3 SFT v2 data loading and configuration.

This module tests configuration loading, prompt generation, tokenization, and collation.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from pathlib import Path
import pytest

from training.sft_v2_data import (
    SFTv2ConfigParams,
    SFTv2DataCollator,
    build_sft_v2_prompt,
    load_sft_v2_config,
    load_sft_v2_records,
    tokenize_sft_v2_record,
)


def dummy_encoder(text: str) -> list[int]:
    """Encode text to token identifiers using character code offsets."""
    return [ord(c) % 500 + 1 for c in text]


def test_load_sft_v2_configs() -> None:
    """Test loading all Stage T3 ablation configurations."""
    # SFT-A1
    cfg_a1 = load_sft_v2_config("training/configs/sft_v2_a1_lora32.yaml")
    assert isinstance(cfg_a1, SFTv2ConfigParams)
    assert cfg_a1.method == "lora"
    assert cfg_a1.lora_r == 32
    assert cfg_a1.lora_alpha == 32
    assert cfg_a1.learning_rate == 0.0001
    assert "devset" in cfg_a1.devset_path

    # SFT-A4
    cfg_a4 = load_sft_v2_config("training/configs/sft_v2_a4_lora64.yaml")
    assert cfg_a4.method == "lora"
    assert cfg_a4.lora_r == 64
    assert cfg_a4.lora_alpha == 128
    assert cfg_a4.learning_rate == 0.00005

    # SFT-A5
    cfg_a5 = load_sft_v2_config("training/configs/sft_v2_a5_full.yaml")
    assert cfg_a5.method == "full"
    assert cfg_a5.lora_r is None
    assert cfg_a5.learning_rate == 0.00002
    assert cfg_a5.optimizer == "paged_adamw_8bit"

    # SFT-A6
    cfg_a6 = load_sft_v2_config("training/configs/sft_v2_a6_dual.yaml")
    assert cfg_a6.enable_thinking is True
    assert cfg_a6.loss_on_answer_only is False


def test_testset_contamination_prevention(tmp_path: Path) -> None:
    """Verify that using the testset raises ValueError."""
    bad_cfg = tmp_path / "bad_config.yaml"
    bad_cfg.write_text(
        """
base_model: "Qwen/Qwen3.5-2B"
sft:
  method: "lora"
  lora_r: 32
  lora_alpha: 32
  devset_path: "eval/testset-1.0.jsonl"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Frozen test set must never be used"):
        load_sft_v2_config(bad_cfg)


def test_build_sft_v2_prompt() -> None:
    """Test prompt building for thinking off and thinking standard."""
    user_msg = "Explain 1. e4 in this position."

    p_off = build_sft_v2_prompt(user_msg, thinking_mode="off")
    assert p_off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    p_on = build_sft_v2_prompt(user_msg, thinking_mode="standard")
    assert p_on.endswith("<|im_start|>assistant\n")
    assert "<think>\n\n</think>" not in p_on


def test_load_sft_v2_records() -> None:
    """Test loading records from sft_v2_train.jsonl."""
    records = load_sft_v2_records("data/processed/sft_v2_train.jsonl", max_examples=10)
    assert len(records) == 10
    for r in records:
        assert "chat_prompt" in r
        assert "assistant_response" in r
        assert "comment" in r


def test_tokenize_sft_v2_record_answer_only() -> None:
    """Test tokenization with answer-only loss masking."""
    rec = {
        "chat_prompt": "PROMPT ",
        "comment": "ANSWER",
        "thinking_mode": "off",
    }
    tok = tokenize_sft_v2_record(rec, encode_fn=dummy_encoder, max_seq_length=50)

    # Prompt tokens must be -100 in labels
    prompt_len = len(dummy_encoder("PROMPT "))
    assert tok["labels"][:prompt_len] == [-100] * prompt_len
    # Answer tokens must not be -100
    assert tok["labels"][prompt_len] != -100
    assert len(tok["input_ids"]) == len(tok["labels"])
    assert len(tok["input_ids"]) == len(tok["attention_mask"])


def test_sft_v2_data_collator() -> None:
    """Test batch collation and padding."""
    b1 = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]}
    b2 = {"input_ids": [4, 5], "labels": [-100, 5], "attention_mask": [1, 1]}

    collator = SFTv2DataCollator(pad_token_id=0, label_pad_token_id=-100)
    out = collator([b1, b2])

    assert len(out["input_ids"]) == 2
    assert len(out["input_ids"][0]) == 3
    assert len(out["input_ids"][1]) == 3
    # Check padding in b2
    val_padded = out["input_ids"][1][2] if isinstance(out["input_ids"], list) else out["input_ids"][1, 2].item()
    label_padded = out["labels"][1][2] if isinstance(out["labels"], list) else out["labels"][1, 2].item()
    assert val_padded == 0
    assert label_padded == -100
