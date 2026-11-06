"""Unit tests for GRPO configuration and prompt dataset loading.

This module tests configuration loading, prompt preparation, and reward monitoring.
All docstrings follow ASD-STE100 rules.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from rewards.api import grpo_reward_fn
from training.grpo_data import (
    GRPOConfigParams,
    MonitoredGRPOReward,
    detect_anomalies,
    load_grpo_config,
    load_rl_prompt_records,
    sample_outputs_for_inspection,
)


def test_load_default_grpo_config() -> None:
    """Test loading and validating the default Phase 6 GRPO configuration."""
    config_path = Path("training/configs/second-1.0-2b.yaml")
    params = load_grpo_config(config_path)

    assert isinstance(params, GRPOConfigParams)
    assert params.model_name == "Second-1.0-2B"
    assert params.version == "1.0"
    assert params.size == "2B"
    assert params.base_model == "Qwen/Qwen3.5-2B"
    assert params.text_only is True
    assert params.enable_thinking is False
    assert params.sft_adapter_dir == "/content/drive/MyDrive/second/checkpoints/sft-1.0-2b/final_adapter"
    assert params.rl_prompts_path == "data/processed/rl_prompts.jsonl"
    assert params.val_data_path == "data/processed/sft_val.jsonl"
    assert params.testset_path == "eval/testset-1.0.jsonl"
    assert params.num_generations == 8
    assert params.max_completion_length == 300
    assert params.max_prompt_length == 512
    assert params.temperature == 0.9
    assert params.learning_rate == 0.000005
    assert params.beta == 0.02
    assert params.max_steps == 500
    assert params.warmup_steps == 50
    assert params.save_steps == 50
    assert params.eval_steps == 50
    assert params.logging_steps == 10
    assert params.save_total_limit == 3
    assert params.per_device_train_batch_size == 8
    assert params.gradient_accumulation_steps == 1
    assert params.gpu == "L4"
    assert params.checkpoint_dir == "/content/drive/MyDrive/second/checkpoints/grpo-1.0-2b"

    assert params.grpo.get("num_generations") == 8
    assert params.mode.get("text_only") is True


def test_load_grpo_config_missing_file() -> None:
    """Test that a missing file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_grpo_config("nonexistent_grpo_config.yaml")


@pytest.mark.parametrize(
    ("override_key", "override_value", "error_match"),
    [
        ("base_model", "meta-llama/Llama-3", "Invalid base_model"),
        ("text_only", False, "Invalid text_only"),
        ("enable_thinking", True, "Invalid enable_thinking"),
        ("num_generations", 4, "Invalid num_generations"),
        ("max_completion_length", 200, "Invalid max_completion_length"),
        ("temperature", 0.2, "Invalid temperature"),
        ("temperature", 2.0, "Invalid temperature"),
        ("learning_rate", 0.0, "Invalid learning_rate"),
        ("learning_rate", 0.005, "Invalid learning_rate"),
        ("beta", -0.05, "Invalid beta"),
        ("beta", 0.8, "Invalid beta"),
        ("max_steps", 200, "Invalid max_steps"),
        ("max_steps", 800, "Invalid max_steps"),
        ("save_steps", 10, "Invalid save_steps"),
        ("save_steps", 60, "Invalid save_steps"),
        ("per_device_train_batch_size", 3, "Invalid batch size"),
        ("per_device_train_batch_size", 0, "Invalid per_device_train_batch_size"),
        ("gradient_accumulation_steps", 0, "Invalid gradient_accumulation_steps"),
    ],
)
def test_load_grpo_config_validations(
    tmp_path: Path,
    override_key: str,
    override_value: Any,
    error_match: str,
) -> None:
    """Test validation errors for invalid GRPO configuration values."""
    config_file = Path("training/configs/second-1.0-2b.yaml")
    config_content = config_file.read_text(encoding="utf-8")
    data = yaml.safe_load(config_content)

    if override_key in ("text_only", "enable_thinking"):
        data["mode"][override_key] = override_value
    elif override_key == "base_model":
        data["base_model"] = override_value
    else:
        data["grpo"][override_key] = override_value

    temp_file = tmp_path / "test_grpo_config.yaml"
    temp_file.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        load_grpo_config(temp_file)


def test_load_real_rl_prompt_records() -> None:
    """Test loading first 50 records from processed RL prompts file."""
    path = Path("data/processed/rl_prompts.jsonl")
    records = load_rl_prompt_records(path, max_prompts=50, shuffle=False)

    assert len(records) == 50
    for rec in records:
        assert "prompt" in rec
        assert "chat_prompt" in rec
        assert "user_message" in rec
        assert rec["prompt"] == rec["chat_prompt"]
        assert rec["prompt"].startswith("<|im_start|>system\n")
        assert rec["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert "eval_before" not in rec
        assert "eval_after" not in rec


def test_load_rl_prompt_records_deterministic_shuffling() -> None:
    """Test deterministic shuffling with fixed random seeds."""
    path = Path("data/processed/rl_prompts.jsonl")
    records_seed42_a = load_rl_prompt_records(path, max_prompts=20, seed=42, shuffle=True)
    records_seed42_b = load_rl_prompt_records(path, max_prompts=20, seed=42, shuffle=True)
    records_seed99 = load_rl_prompt_records(path, max_prompts=20, seed=99, shuffle=True)
    records_unshuffled = load_rl_prompt_records(path, max_prompts=20, shuffle=False)

    ids_42_a = [r["id"] for r in records_seed42_a]
    ids_42_b = [r["id"] for r in records_seed42_b]
    ids_99 = [r["id"] for r in records_seed99]
    ids_unshuffled = [r["id"] for r in records_unshuffled]

    assert ids_42_a == ids_42_b
    assert ids_42_a != ids_99
    assert ids_42_a != ids_unshuffled


def test_load_rl_prompt_records_synthetic_fallback(tmp_path: Path) -> None:
    """Test fallback generation of chat_prompt and user_message."""
    jsonl_file = tmp_path / "prompts.jsonl"
    line1 = (
        '{"id": "test_1", "user_message": "task: explain_move\\nFEN: 8/8/8/8/8/8/8/8 w - - 0 1", '
        '"eval_before": {"pvs": []}, "eval_after": {"pvs": []}}\n'
    )
    line2 = (
        '{"id": "test_2", "chat_prompt": "<|im_start|>system\\nTest\\n<|im_end|>\\n'
        '<|im_start|>user\\ntask: explain_position\\nFEN: 8/8/8/8/8/8/8/8 w - - 0 1'
        '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"}\n'
    )
    jsonl_file.write_text(line1 + line2, encoding="utf-8")

    records = load_rl_prompt_records(jsonl_file, shuffle=False)
    assert len(records) == 2

    # Record 1 had user_message, built chat_prompt
    assert records[0]["prompt"] == records[0]["chat_prompt"]
    assert "<think>\n\n</think>\n\n" in records[0]["chat_prompt"]
    assert "eval_before" not in records[0]
    assert "eval_after" not in records[0]

    # Record 2 had chat_prompt, extracted user_message
    assert "task: explain_position" in records[1]["user_message"]
    assert records[1]["prompt"] == records[1]["chat_prompt"]


def test_load_rl_prompt_records_missing_both_raises(tmp_path: Path) -> None:
    """Test that missing user_message and chat_prompt raises ValueError."""
    jsonl_file = tmp_path / "bad.jsonl"
    jsonl_file.write_text('{"id": "bad_rec"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing chat_prompt and user_message"):
        load_rl_prompt_records(jsonl_file)


def test_load_rl_prompt_records_missing_file() -> None:
    """Test that nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="RL prompt file not found"):
        load_rl_prompt_records("nonexistent_rl_prompts.jsonl")


def test_monitored_grpo_reward_attributes_and_call() -> None:
    """Test reward monitor naming attribute and scoring output parity."""
    assert MonitoredGRPOReward.__name__ == "grpo_rule_reward"

    monitor = MonitoredGRPOReward(log_interval_batches=2, sample_size=5, seed=42)
    assert monitor.__name__ == "grpo_rule_reward"

    prompt = (
        "task: explain_move\nlevel: beginner\ndetail: short\n\n"
        "FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4\n"
        "White to move, move 4.\n"
        "Move played: Ng5.\n"
        "Verdict: Good.\n"
        "White's winning chances: 54% → 53%.\n"
        "Best move: d3.\n"
        "Best line: d3 Bc5 c3 d6.\n"
        "Facts:\n"
        "- Ng5 attacks the pawn on f7 together with the bishop on c4.\n"
        "- Black's best reply is d5.\n"
        "- Material: equal.\n"
        "- Phase: opening.\n"
        "- Opening: Italian Game, Two Knights Defense."
    )
    completion = "Ng5 attacks f7 and develops the knight. Black should defend with d5."

    expected_rewards = grpo_reward_fn([prompt], [completion])
    rewards = monitor([prompt], [completion])

    assert rewards == expected_rewards
    assert monitor.call_count == 1
    assert len(monitor.current_window) == 1
    assert monitor.current_window[0]["reward"] == expected_rewards[0]
    assert monitor.current_window[0]["passed_all_hard"] is True
    assert monitor.current_window[0]["word_count"] > 0
    assert monitor.current_window[0]["opening_prefix"] == "ng5 attacks f7 and"
    assert len(monitor.step_summaries) == 0


def test_monitored_grpo_reward_logging_interval() -> None:
    """Test window summarization and deterministic sampling at interval steps."""
    monitor = MonitoredGRPOReward(log_interval_batches=2, sample_size=2, seed=42)

    prompt = (
        "task: explain_position\nlevel: beginner\ndetail: short\n\n"
        "FEN: 8/8/8/8/8/8/8/8 w - - 0 1\n"
        "Facts:\n- Material: equal."
    )
    completion = "The board has no pieces on any square."

    # First call: window receives 1 sample, no logging yet
    monitor([prompt], [completion])
    assert monitor.call_count == 1
    assert len(monitor.current_window) == 1
    assert len(monitor.step_summaries) == 0

    # Second call: hits interval 2, logs summary and clears window
    monitor([prompt], [completion])
    assert monitor.call_count == 2
    assert len(monitor.current_window) == 0
    assert len(monitor.step_summaries) == 1
    summary = monitor.step_summaries[0]
    assert summary["step"] == 2
    assert summary["sample_count"] == 2
    assert "mean_reward" in summary
    assert "length_collapse" in summary

    assert 2 in monitor.sampled_outputs
    assert len(monitor.sampled_outputs[2]) == 2


def test_monitored_grpo_reward_exception_handling() -> None:
    """Test that malformed prompt input produces negative one reward."""
    monitor = MonitoredGRPOReward()
    bad_prompt = "Malformed text without FEN header"
    completion = "Irrelevant completion."

    rewards = monitor([bad_prompt], [completion])
    assert rewards == [-1.0]
    assert monitor.current_window[0]["passed_all_hard"] is False
    assert "scoring_exception" in monitor.current_window[0]["failed_checks"]


def test_detect_anomalies_clean() -> None:
    """Test anomaly detection on normal outputs without collapse or repetition."""
    records = [
        {
            "word_count": 30,
            "passed_all_hard": True,
            "reward": 0.8,
            "opening_prefix": f"prefix number {i} text",
        }
        for i in range(12)
    ]

    anomalies = detect_anomalies(records)
    assert anomalies["length_collapse"] is False
    assert anomalies["robotic_text"] is False
    assert anomalies["high_hard_fail"] is False
    assert anomalies["avg_word_count"] == 30.0
    assert anomalies["hard_fail_rate"] == 0.0
    assert anomalies["mean_reward"] == 0.8


def test_detect_anomalies_length_collapse() -> None:
    """Test detection of length collapse when average words are under 15."""
    records = [
        {
            "word_count": 10,
            "passed_all_hard": True,
            "reward": 0.5,
            "opening_prefix": f"short prefix {i} words",
        }
        for i in range(10)
    ]

    anomalies = detect_anomalies(records)
    assert anomalies["length_collapse"] is True
    assert anomalies["avg_word_count"] == 10.0


def test_detect_anomalies_robotic_text() -> None:
    """Test detection of robotic repeated opening prefixes exceeding 40 percent."""
    records = []
    # 5 identical prefixes out of 10 samples = 50% (> 40%)
    for i in range(10):
        prefix = "this move is a" if i < 5 else f"different prefix {i} text"
        records.append({
            "word_count": 25,
            "passed_all_hard": True,
            "reward": 0.6,
            "opening_prefix": prefix,
        })

    anomalies = detect_anomalies(records)
    assert anomalies["robotic_text"] is True
    assert anomalies["top_opening_prefix"] == "this move is a"
    assert anomalies["top_opening_share"] == 0.5


def test_detect_anomalies_high_hard_fail() -> None:
    """Test detection of high hard-fail rate exceeding 50 percent."""
    records = [
        {
            "word_count": 25,
            "passed_all_hard": i < 4,  # 6 fails out of 10 = 60%
            "reward": -1.0 if i >= 4 else 0.7,
            "opening_prefix": f"prefix {i} text here",
        }
        for i in range(10)
    ]

    anomalies = detect_anomalies(records)
    assert anomalies["high_hard_fail"] is True
    assert anomalies["hard_fail_rate"] == 60.0


def test_detect_anomalies_empty() -> None:
    """Test anomaly detection on an empty record list."""
    anomalies = detect_anomalies([])
    assert anomalies["length_collapse"] is False
    assert anomalies["robotic_text"] is False
    assert anomalies["high_hard_fail"] is False
    assert anomalies["avg_word_count"] == 0.0
    assert anomalies["hard_fail_rate"] == 0.0


def test_sample_outputs_for_inspection() -> None:
    """Test deterministic record sampling for human inspection."""
    records = [{"id": f"rec_{i}"} for i in range(50)]

    # Sample size larger than record count returns all items
    sample_large = sample_outputs_for_inspection(records[:5], count=10)
    assert len(sample_large) == 5

    # Sample 20 items deterministically
    sample_a = sample_outputs_for_inspection(records, count=20, seed=42)
    sample_b = sample_outputs_for_inspection(records, count=20, seed=42)
    sample_diff = sample_outputs_for_inspection(records, count=20, seed=99)

    assert len(sample_a) == 20
    assert sample_a == sample_b
    assert sample_a != sample_diff


def test_grpo_notebook_syntax_and_bundle_contents(tmp_path: Path) -> None:
    """Verify that training/grpo.ipynb has valid Python syntax and bundle builder works."""
    import ast
    import json
    import zipfile
    from training.build_colab_bundle import build_phase6_colab_bundle

    nb_path = Path("training/grpo.ipynb")
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) == 10

    for idx, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        clean_lines = [
            line if not line.lstrip().startswith("!") else f"# {line}"
            for line in source.splitlines()
        ]
        ast.parse("\n".join(clean_lines), filename=f"grpo_cell_{idx}")

    out_zip = tmp_path / "test_phase6_bundle.zip"
    built = build_phase6_colab_bundle(output_path=out_zip)
    assert built.is_file()
    with zipfile.ZipFile(built, "r") as zf:
        archive_names = set(zf.namelist())
        assert "data/processed/rl_prompts.jsonl" in archive_names
        assert "data/processed/sft_val.jsonl" in archive_names
        assert "eval/testset-1.0.jsonl" in archive_names
        assert "eval/results/sft-1.0-2b_outputs.jsonl" in archive_names
        assert "eval/results/baselines/base-qwen-2b.jsonl" in archive_names
        assert "training/configs/second-1.0-2b.yaml" in archive_names
        assert "training/grpo_data.py" in archive_names
        assert "training/evaluate_grpo.py" in archive_names

