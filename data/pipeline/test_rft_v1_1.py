"""Unit tests for data/pipeline/rft_v1_1.py.

This module verifies open-weight model enforcement, Reward v3 RFT filtering, and prompt rebuilding.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.pipeline.rft_v1_1 import (
    DEFAULT_OPEN_WEIGHT_MODEL,
    _apply_chat_template_no_thinking,
    _resolve_user_message,
    build_rft_messages,
    evaluate_rft_candidate,
    filter_rft_dataset,
    generate_rft_candidates,
    rebuild_v1_1_prompts,
    select_best_rft_candidate,
    validate_open_weight_model,
)
from training.grpo_data import load_grpo_config
from training.sft_data import load_sft_config


def _sample_6_exd4_item() -> dict[str, Any]:
    """Return a formatted 6... exd4 prompt item for RFT unit tests."""
    fen = "rnb1k2r/ppppqppp/5n2/4p3/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R b KQkq - 0 6"
    facts = [
        "After exd4, White's best reply is Nxd4.",
        "After exd4, White's recapture Nxd4 centralizes the knight.",
        "Best move e4 keeps central space and tension instead of trading on d4.",
        "Unlike exd4, e4 prevents White from centralizing pieces.",
        "Material: equal.",
        "Phase: opening.",
    ]
    user_msg = (
        "task: explain_move\nlevel: club\ndetail: short\n\n"
        f"FEN: {fen}\nBlack to move, move 6.\nMove played: exd4.\n"
        "Verdict: Inaccuracy.\nBlack's winning chances: 49% → 43%.\n"
        "Best move: e4.\nBest line: e4 Ng1 Bxc3 Bxc3.\nFacts:\n"
        + "\n".join(f"- {f}" for f in facts)
    )
    return {
        "id": "rft_sample_1",
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "fen": fen,
        "move_uci": "e5d4",
        "verdict": "Inaccuracy",
        "facts": facts,
        "user_message": user_msg,
    }


def test_validate_open_weight_model_blocks_proprietary_apis() -> None:
    """Allow Qwen open-weight models and reject Gemini, GPT, and Claude identifiers."""
    assert validate_open_weight_model(DEFAULT_OPEN_WEIGHT_MODEL) == DEFAULT_OPEN_WEIGHT_MODEL
    assert validate_open_weight_model("Qwen/Qwen3.5-9B") == "Qwen/Qwen3.5-9B"
    assert validate_open_weight_model("Qwen/Qwen3.5-27B-GPTQ-Int4") == "Qwen/Qwen3.5-27B-GPTQ-Int4"
    assert validate_open_weight_model("unsloth/Qwen3.5-9B-bnb-4bit") == "unsloth/Qwen3.5-9B-bnb-4bit"
    assert validate_open_weight_model("/checkpoints/second-1.1-9b") == "/checkpoints/second-1.1-9b"
    assert validate_open_weight_model("./local_qwen_ckpt") == "./local_qwen_ckpt"

    for bad_model in (
        "gemini-2.5-flash",
        "google/gemini-2.0-flash",
        "/tmp/gemini-local",
        "./gpt-4o-local",
        "gpt",
        "gpt4",
        "gpt-4o",
        "openai/gpt-5",
        "claude-3-5-sonnet",
        "o3-mini",
        "unknown-vendor/custom-model",
    ):
        with pytest.raises(ValueError):
            validate_open_weight_model(bad_model)


def test_build_rft_messages_and_evaluate_candidate() -> None:
    """Reject boilerplate or missing positional grounding and accept clean RFT prose."""
    item = _sample_6_exd4_item()
    msgs = build_rft_messages(item["user_message"])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "Fast mode" in msgs[0]["content"]

    boilerplate_cand = (
        "Playing exd4 is an inaccuracy because it blocks coordination and creates unnecessary weaknesses. "
        "Instead, e4 keeps a stronger overall position for Black in the opening."
    )
    res_bad = evaluate_rft_candidate(item, boilerplate_cand)
    assert not res_bad["passed"]
    assert "hard_vague_boilerplate" in res_bad["failed_gates"]

    good_cand = (
        "Playing exd4 is an inaccuracy because capturing releases central tension too early "
        "and allows White to centralize the knight with Nxd4. Instead, pushing e4 claims space "
        "in the center, keeps the position flexible, and stops White from occupying the d4 square."
    )
    res_good = evaluate_rft_candidate(item, good_cand)
    assert res_good["passed"]
    assert res_good["reward"] >= 0.82


def test_select_best_and_filter_rft_dataset(tmp_path: Path) -> None:
    """Verify Best-of-K RFT filtering emits Fast-mode records and telemetry."""
    item = _sample_6_exd4_item()
    bad_cand = "This move blocks coordination. Play e4 instead."
    good_cand = (
        "Playing exd4 is an inaccuracy because capturing releases central tension too early "
        "and lets White centralize the knight with Nxd4. Instead, pushing e4 claims space "
        "in the center, keeps the position flexible, and stops White from claiming the d4 square."
    )

    rec, tel = select_best_rft_candidate(
        item,
        [bad_cand, good_cand],
        model_name="Qwen/Qwen3.5-27B",
    )
    assert rec is not None
    assert rec["thinking_mode"] == "off"
    assert rec["reward_version"] == "v3"
    assert rec["assistant_response"].startswith("<think>\n\n</think>\n\n")
    assert tel["candidates_passed"] == 1

    in_path = tmp_path / "rft_candidates.jsonl"
    out_path = tmp_path / "rft_verified.jsonl"
    row_with_cands = dict(item)
    row_with_cands["candidates"] = [bad_cand, good_cand]
    in_path.write_text(json.dumps(row_with_cands) + "\n", encoding="utf-8")

    stats = filter_rft_dataset(in_path, out_path, model_name="Qwen/Qwen3.5-27B")
    assert stats["accepted_prompts"] == 1
    assert out_path.exists()
    out_lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(out_lines) == 1
    assert out_lines[0]["rft_model"] == "Qwen/Qwen3.5-27B"

    with pytest.raises(ValueError):
        filter_rft_dataset(in_path, out_path, model_name="gemini-2.5-flash")


def test_rebuild_v1_1_prompts(tmp_path: Path) -> None:
    """Verify prompt rebuilder regenerates v1.1 facts and suppresses noisy checks."""
    fen = "r1bqkb1r/pppp1ppp/8/4p3/2Bn4/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5"
    old_record = {
        "id": "old_1",
        "task": "explain_position",
        "level": "club",
        "detail": "short",
        "fen": fen,
        "move_uci": None,
        "eval_before": {
            "fen": fen,
            "pvs": [{"score": {"type": "cp", "value": 35}, "moves": ["f3d4", "e5d4"]}],
        },
        "eval_after": None,
        "facts": ["White has a check: Bxf7+."],
        "user_message": "stale",
        "chat_prompt": "stale",
    }
    in_p = tmp_path / "old.jsonl"
    out_p = tmp_path / "new.jsonl"
    in_p.write_text(json.dumps(old_record) + "\n", encoding="utf-8")

    count = rebuild_v1_1_prompts(in_p, out_p)
    assert count == 1
    rebuilt = json.loads(out_p.read_text(encoding="utf-8").strip())
    assert "White has a check: Bxf7+." not in rebuilt["facts"]
    assert "White can capture the knight on d4 with Nxd4." in rebuilt["facts"]


def test_rft_best_move_fallback_mock_generate_and_grpo_config(tmp_path: Path) -> None:
    """Verify eval_before best_move_san fallback, mock RFT generation, and v1.1 GRPO config."""
    item = _sample_6_exd4_item()
    item_minimal = {
        "id": "rft_min_1",
        "task": item["task"],
        "level": item["level"],
        "detail": item["detail"],
        "fen": item["fen"],
        "move_uci": item["move_uci"],
        "verdict": item["verdict"],
        "facts": item["facts"],
        "user_message": "",
        "eval_before": {
            "fen": item["fen"],
            "pvs": [{"score": {"type": "cp", "value": 16}, "moves": ["e5e4", "f3g1"]}],
        },
    }
    good_cand = (
        "Playing exd4 is an inaccuracy because capturing releases central tension too early "
        "and lets White centralize the knight with Nxd4. Instead, pushing e4 claims space "
        "in the center, keeps the position flexible, and stops White from occupying the d4 square."
    )
    res_eval = evaluate_rft_candidate(item_minimal, good_cand)
    assert res_eval["soft_scores"]["better_move"] == 1.0

    # Mock generate + filter round trip
    prompts_p = tmp_path / "prompts.jsonl"
    cands_p = tmp_path / "cands.jsonl"
    filtered_p = tmp_path / "filtered.jsonl"
    prompts_p.write_text(json.dumps(item) + "\n", encoding="utf-8")

    gen_stats = generate_rft_candidates(
        prompts_p,
        cands_p,
        model_name="Qwen/Qwen3.5-27B",
        num_samples=2,
        mock=True,
    )
    assert gen_stats["prompts_processed"] == 1
    assert gen_stats["backend"] == "mock"
    filt_stats = filter_rft_dataset(cands_p, filtered_p, model_name="Qwen/Qwen3.5-27B")
    assert filt_stats["accepted_prompts"] == 1

    with pytest.raises(ValueError):
        generate_rft_candidates(prompts_p, cands_p, model_name="gemini-2.5-pro", mock=True)

    # Validate second-1.1-recipe.yaml loads cleanly with load_sft_config and load_grpo_config
    sft_cfg = load_sft_config("training/configs/second-1.1-recipe.yaml")
    assert sft_cfg.base_model == "Qwen/Qwen3.5-9B"
    assert sft_cfg.lora_r == 32

    cfg = load_grpo_config("training/configs/second-1.1-recipe.yaml")
    assert cfg.reward_version == "v3"
    assert cfg.loss_type == "dapo"
    assert cfg.multi_objective_aggregation == "normalize_then_sum"
    assert cfg.val_data_path == "data/processed/sft_v1_1_val.jsonl"
    assert cfg.devset_path == "eval/devset-1.1.jsonl"
    assert cfg.testset_path == "eval/testset-1.1.jsonl"

    # _apply_chat_template_no_thinking falls back when enable_thinking raises TypeError
    class _LegacyTokenizer:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            return f"legacy:{len(messages)}"

    assert _apply_chat_template_no_thinking(_LegacyTokenizer(), build_rft_messages("hello")) == "legacy:2"
    assert "FEN:" in _resolve_user_message(item_minimal)

    # select_best_rft_candidate rebuilds user_message and chat_prompt when empty and eval_before exists
    item_no_prompt = dict(item_minimal)
    rec_fallback, _ = select_best_rft_candidate(item_no_prompt, [good_cand], model_name="Qwen/Qwen3.5-27B")
    assert rec_fallback is not None
    assert "FEN:" in rec_fallback["user_message"]
    assert "FEN:" in rec_fallback["chat_prompt"]

    # Mock RFT generation + filtering on full devset-1.1 (160) and testset-1.1 (320)
    for ds_path, min_keep in (
        ("eval/devset-1.1.jsonl", 158),
        ("eval/testset-1.1.jsonl", 310),
    ):
        cands_tmp = tmp_path / f"{Path(ds_path).stem}_cands.jsonl"
        filt_tmp = tmp_path / f"{Path(ds_path).stem}_filt.jsonl"
        generate_rft_candidates(ds_path, cands_tmp, model_name="Qwen/Qwen3.5-27B", num_samples=2, mock=True)
        ds_stats = filter_rft_dataset(cands_tmp, filt_tmp, model_name="Qwen/Qwen3.5-27B")
        assert ds_stats["accepted_prompts"] >= min_keep, f"{ds_path}: {ds_stats}"

    dev_cands_rows = [
        json.loads(line)
        for line in (tmp_path / "devset-1.1_cands.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all("in variant" not in cand.lower() for r in dev_cands_rows for cand in r["candidates"])
    dev_filt_rows = [
        json.loads(line)
        for line in (tmp_path / "devset-1.1_filt.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev_all_rows = [json.loads(line) for line in Path("eval/devset-1.1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    long_total = sum(1 for r in dev_all_rows if r.get("detail") == "long")
    long_kept = sum(1 for r in dev_filt_rows if r.get("detail") == "long")
    assert long_kept == long_total
