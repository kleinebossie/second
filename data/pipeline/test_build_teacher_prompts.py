"""Unit tests for data/pipeline/build_teacher_prompts.py.

All test functions follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from data.pipeline.build_teacher_prompts import (
    DEFAULT_DEVSET_PATH,
    DEFAULT_TESTSET_PATH,
    detect_candidate_motifs,
    format_teacher_record,
    load_exclusion_sets,
    stratify_teacher_prompts,
    verify_teacher_prompts,
)


def test_detect_candidate_motifs() -> None:
    """Verify motif detector identifies tactical motifs and phases from facts."""
    fen = "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"
    facts = [
        "White has a check: Bxf7+.",
        "Nd5 forks the queen on d8 and the pawn on c7.",
        "Phase: opening.",
    ]
    motifs = detect_candidate_motifs(
        fen=fen,
        move_uci="c3d5",
        facts=facts,
        eval_before=None,
    )
    assert "fork" in motifs
    assert "opening_safety" in motifs


def test_load_exclusion_sets() -> None:
    """Verify exclusion loader returns non-empty sets from devset and testset."""
    ex_games, ex_fens = load_exclusion_sets(
        devset_path=DEFAULT_DEVSET_PATH,
        testset_path=DEFAULT_TESTSET_PATH,
    )
    assert len(ex_games) > 0
    assert len(ex_fens) > 0


def test_format_teacher_record() -> None:
    """Verify teacher record formatting produces valid schema and Qwen prompt."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    eb = {
        "fen": fen,
        "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["e7e5", "g1f3"]}],
    }
    ea = {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["g1f3"]}],
    }
    cand = {
        "game_id": "test_game_123",
        "fen": fen,
        "source": "lichess_games",
        "eval_before": eb,
        "eval_after": ea,
        "move_uci": "e7e5",
        "verdict": "Good",
    }
    rec = format_teacher_record(
        cand=cand,
        idx=42,
        task="explain_move",
        level="club",
        detail="short",
    )
    assert rec["id"] == "teacher_prompt_000042"
    assert rec["task"] == "explain_move"
    assert rec["level"] == "club"
    assert rec["detail"] == "short"
    assert rec["verdict"] == "Good"
    assert "Move played: e5." in rec["user_message"]
    assert "<|im_start|>assistant" in rec["chat_prompt"]
    assert isinstance(rec["facts"], list)
    assert len(rec["facts"]) > 0


def test_verify_teacher_prompts_zero_contamination(tmp_path: Path) -> None:
    """Verify contamination detector raises ValueError when devset FEN matches."""
    dev_file = tmp_path / "devset.jsonl"
    test_file = tmp_path / "testset.jsonl"

    dev_fen = "8/8/8/8/8/8/8/4K2k w - - 0 1"
    with open(dev_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"game_id": "game_dev", "fen": dev_fen}) + "\n")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"game_id": "game_test", "fen": "8/8/8/8/8/8/8/7K w - - 0 1"}) + "\n")

    clean_records = [
        {
            "id": "prompt_0",
            "game_id": "game_clean",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "verdict": "Good",
            "motif_tags": ["opening_safety"],
        }
    ]
    summary = verify_teacher_prompts(clean_records, devset_path=dev_file, testset_path=test_file)
    assert summary["zero_contamination"] is True

    # Contaminated record
    contaminated = [
        {
            "id": "prompt_1",
            "game_id": "game_clean",
            "fen": dev_fen,
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "verdict": "Good",
            "motif_tags": [],
        }
    ]
    with pytest.raises(ValueError, match="Contamination detected"):
        verify_teacher_prompts(contaminated, devset_path=dev_file, testset_path=test_file)


def test_verify_teacher_prompts_enforce_quotas(tmp_path: Path) -> None:
    """Verify enforce_quotas raises ValueError when motif targets are violated."""
    dev_file = tmp_path / "devset.jsonl"
    test_file = tmp_path / "testset.jsonl"
    dev_file.write_text("", encoding="utf-8")
    test_file.write_text("", encoding="utf-8")

    # Record missing forks (0.0% < 15.0%)
    records = [
        {
            "id": "prompt_0",
            "game_id": "game_0",
            "fen": "8/8/8/8/8/8/8/k6K w - - 0 1",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "verdict": "Good",
            "motif_tags": ["opening_safety"],
        }
    ]
    with pytest.raises(ValueError, match="Motif quota failed for 'fork'"):
        verify_teacher_prompts(records, devset_path=dev_file, testset_path=test_file, enforce_quotas=True)


def test_stratify_teacher_prompts_mock() -> None:
    """Verify prompt stratification enforces exact task and verdict allocations."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    eb = {"fen": fen, "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["e7e5"]}]}
    ea = {"fen": fen, "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["g1f3"]}]}

    verdicts = ["Blunder", "Mistake", "Inaccuracy", "Best", "Good", "Only move"]
    candidates_by_verdict = {
        v: [
            {
                "game_id": f"game_{v}_{i}",
                "fen": fen,
                "source": "lichess_games",
                "eval_before": eb,
                "eval_after": ea,
                "move_uci": "e7e5",
                "verdict": v,
            }
            for i in range(30)
        ]
        for v in verdicts
    }
    position_candidates = [
        {
            "game_id": f"game_pos_{i}",
            "fen": fen,
            "source": "lichess_games",
            "eval_before": eb,
            "move_uci": None,
            "eval_after": None,
            "verdict": None,
        }
        for i in range(30)
    ]

    records = stratify_teacher_prompts(
        candidates_by_verdict=candidates_by_verdict,
        position_candidates=position_candidates,
        target_count=20,
        seed=42,
    )

    assert len(records) == 20
    move_count = sum(1 for r in records if r["task"] == "explain_move")
    pos_count = sum(1 for r in records if r["task"] == "explain_position")
    assert move_count == 16  # 80%
    assert pos_count == 4    # 20%

    short_count = sum(1 for r in records if r["detail"] == "short")
    long_count = sum(1 for r in records if r["detail"] == "long")
    assert short_count == 14  # 70%
    assert long_count == 6    # 30%
