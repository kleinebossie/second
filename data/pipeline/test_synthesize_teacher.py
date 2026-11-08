"""Tests for data/pipeline/synthesize_teacher.py.

All test functions follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from data.pipeline.synthesize_teacher import (
    DEFAULT_TEACHER_MODEL,
    TEACHER_SYSTEM_PROMPT,
    build_teacher_messages,
    emit_student_records,
    generate_mock_teacher_completion,
    parse_teacher_completion,
    run_teacher_synthesis,
)


def test_build_teacher_messages() -> None:
    """Verify teacher messages include system prompt and user message."""
    user_msg = "task: explain_move\nMove played: e4."
    messages = build_teacher_messages(user_msg)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == TEACHER_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == user_msg


def test_parse_teacher_completion_with_thinking() -> None:
    """Verify parse separates thinking trace from final explanation."""
    raw = (
        "<think>\n"
        "verdict: Blunder (loses queen)\n"
        "main_fact: White plays Qxe7#\n"
        "best_move: Be7\n"
        "coach_plan: Explain checkmate threat\n"
        "</think>\n\n"
        "Nf6 is a blunder because it allows White to deliver checkmate on e7. "
        "Instead, Black must play Be7 to cover the king."
    )
    result = parse_teacher_completion(raw)
    assert "verdict: Blunder" in result["trace"]
    assert "Nf6 is a blunder" in result["explanation"]
    assert "<think>" not in result["explanation"]


def test_parse_teacher_completion_without_thinking() -> None:
    """Verify parse handles outputs without thinking tags."""
    raw = "e4 is a solid opening move controlling the center."
    result = parse_teacher_completion(raw)
    assert result["trace"] == ""
    assert result["explanation"] == raw


def test_emit_student_records() -> None:
    """Verify emission of both non-thinking and thinking records."""
    item = {
        "user_message": "task: explain_move\nFEN: ...",
        "task": "explain_move",
        "level": "beginner",
        "detail": "short",
    }
    parsed = {
        "trace": "verdict: Best (controls d5)",
        "explanation": "e4 controls the central square d5.",
    }
    rec_off, rec_std = emit_student_records(item, parsed)

    # Check non-thinking
    assert rec_off["thinking_mode"] == "off"
    assert rec_off["assistant_response"].startswith("<think>\n\n</think>")
    assert rec_off["comment"] == "e4 controls the central square d5."
    assert rec_off["explanation"] == "e4 controls the central square d5."
    assert "e4 controls the central square d5." in rec_off["assistant_response"]

    # Check thinking
    assert rec_std["thinking_mode"] == "standard"
    assert "<think>\nverdict: Best (controls d5)\n</think>" in rec_std["assistant_response"]
    assert rec_std["comment"] == rec_std["assistant_response"]
    assert rec_std["explanation"] == "e4 controls the central square d5."
    assert rec_std["trace"] == "verdict: Best (controls d5)"


def test_generate_mock_teacher_completion() -> None:
    """Verify mock generator produces structured thinking tags and explanation."""
    item = {
        "task": "explain_move",
        "detail": "short",
        "verdict": "Mistake",
        "facts": ["After h3, Black's best reply is d5."],
        "user_message": "Move played: h3.\nVerdict: Mistake.\nBest move: O-O.",
    }
    mock_out = generate_mock_teacher_completion(item)
    assert "<think>" in mock_out
    assert "</think>" in mock_out
    assert "h3" in mock_out
    assert "mistake" in mock_out.lower()
    assert "O-O" in mock_out

    # Test explain_position mock
    pos_item = {
        "task": "explain_position",
        "detail": "long",
        "verdict": None,
        "facts": ["Top candidate moves: Nf3, c4, d4."],
        "user_message": "FEN: ...\nBest move: Nf3.",
    }
    mock_pos = generate_mock_teacher_completion(pos_item)
    assert "<think>" in mock_pos
    assert "verdict_or_eval:" in mock_pos
    assert "Nf3" in mock_pos
    assert len(mock_pos.split()) >= 95


def test_run_teacher_synthesis_mock(tmp_path: Path) -> None:
    """Verify run_teacher_synthesis executes in mock mode and saves output JSONL."""
    prompts_file = tmp_path / "prompts.jsonl"
    out_file = tmp_path / "completions.jsonl"

    items = [
        {"id": f"p_{i}", "task": "explain_position", "detail": "short", "user_message": "Position..."}
        for i in range(5)
    ]
    with open(prompts_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    saved_path = run_teacher_synthesis(
        prompts_path=prompts_file,
        output_path=out_file,
        mock=True,
    )
    assert saved_path.is_file()

    with open(saved_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 5
    assert lines[0]["id"] == "p_0"
    assert lines[0]["model"] == "mock-teacher"
    assert "<think>" in lines[0]["completion"]


def test_default_teacher_model() -> None:
    """Verify that DEFAULT_TEACHER_MODEL is Qwen/Qwen3.5-27B."""
    assert DEFAULT_TEACHER_MODEL == "Qwen/Qwen3.5-27B"


def test_run_teacher_synthesis_load_in_4bit_mock(tmp_path: Path) -> None:
    """Verify run_teacher_synthesis accepts load_in_4bit in mock mode."""
    prompts_file = tmp_path / "prompts.jsonl"
    out_file = tmp_path / "completions.jsonl"
    item = {"id": "p_0", "task": "explain_move", "detail": "short", "user_message": "Position..."}
    prompts_file.write_text(json.dumps(item) + "\n", encoding="utf-8")

    saved_path = run_teacher_synthesis(
        prompts_path=prompts_file,
        output_path=out_file,
        mock=True,
        load_in_4bit=True,
    )
    assert saved_path.is_file()
    lines = [json.loads(line) for line in saved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["id"] == "p_0"


def test_run_teacher_synthesis_load_in_4bit_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_teacher_synthesis configures 4-bit BitsAndBytes quantization and left-padding."""
    import sys
    from unittest.mock import MagicMock

    mock_torch = MagicMock()
    mock_torch.bfloat16 = "mock_bfloat16"
    mock_transformers = MagicMock()

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 151643
    mock_tokenizer.eos_token_id = 151643
    mock_tokenizer.apply_chat_template.return_value = "prompt text"
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.device = "cuda:0"
    mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    mock_inputs = {"input_ids": [[1, 2, 3]]}
    mock_tokenizer.return_value.to.return_value = mock_inputs
    mock_tokenizer.decode.return_value = "<think>plan</think> explanation"

    prompts_file = tmp_path / "prompts.jsonl"
    out_file = tmp_path / "completions.jsonl"
    item = {"id": "p_0", "task": "explain_move", "detail": "short", "user_message": "FEN..."}
    prompts_file.write_text(json.dumps(item) + "\n", encoding="utf-8")

    saved_path = run_teacher_synthesis(
        prompts_path=prompts_file,
        output_path=out_file,
        mock=False,
        load_in_4bit=True,
    )

    assert saved_path.is_file()
    assert mock_tokenizer.padding_side == "left"
    mock_transformers.BitsAndBytesConfig.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="mock_bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
    _, kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args
    assert "quantization_config" in kwargs
    assert kwargs["device_map"] == "auto"
    assert kwargs["torch_dtype"] == "mock_bfloat16"

    lines = [json.loads(line) for line in saved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["id"] == "p_0"
    assert lines[0]["completion"] == "<think>plan</think> explanation"


def test_run_teacher_synthesis_no_4bit_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_teacher_synthesis omits quantization_config when load_in_4bit is False."""
    import sys
    from unittest.mock import MagicMock

    mock_torch = MagicMock()
    mock_torch.bfloat16 = "mock_bfloat16"
    mock_transformers = MagicMock()

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 151643
    mock_tokenizer.eos_token_id = 151643
    mock_tokenizer.apply_chat_template.return_value = "prompt text"
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.device = "cuda:0"
    mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    mock_inputs = {"input_ids": [[1, 2, 3]]}
    mock_tokenizer.return_value.to.return_value = mock_inputs
    mock_tokenizer.decode.return_value = "<think>plan</think> explanation"

    prompts_file = tmp_path / "prompts.jsonl"
    out_file = tmp_path / "completions.jsonl"
    item = {"id": "p_0", "task": "explain_move", "detail": "short", "user_message": "FEN..."}
    prompts_file.write_text(json.dumps(item) + "\n", encoding="utf-8")

    saved_path = run_teacher_synthesis(
        prompts_path=prompts_file,
        output_path=out_file,
        mock=False,
        load_in_4bit=False,
    )

    assert saved_path.is_file()
    mock_transformers.BitsAndBytesConfig.assert_not_called()
    _, kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args
    assert "quantization_config" not in kwargs


