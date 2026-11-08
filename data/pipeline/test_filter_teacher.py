"""Unit tests for data/pipeline/filter_teacher.py.

All test functions follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import pytest

from data.pipeline.filter_teacher import (
    evaluate_candidate_completion,
    filter_teacher_completions,
)


@pytest.fixture
def mock_prompt_item() -> dict[str, str]:
    """Provide a standard explain_move prompt dictionary for testing."""
    return {
        "id": "prompt_test_001",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "verdict": "Mistake",
        "move_uci": "h2h3",
        "facts": [
            "After h3, Black's best reply is d5.",
            "Material: equal.",
            "Phase: opening.",
        ],
        "user_message": (
            "task: explain_move\nlevel: club\ndetail: short\n\n"
            "FEN: r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5\n"
            "White to move, move 5.\nMove played: h3.\nVerdict: Mistake.\n"
            "White's winning chances: 52% → 38%.\nBest move: O-O.\n"
            "Facts:\n- After h3, Black's best reply is d5.\n- Material: equal.\n- Phase: opening."
        ),
    }


def test_evaluate_candidate_completion_valid(mock_prompt_item: dict[str, str]) -> None:
    """Verify evaluation accepts completion with valid trace, prose, and reward."""
    raw_completion = (
        "<think>\n"
        "verdict_or_eval: Mistake (wastes a tempo in the opening)\n"
        "main_fact: After h3, Black's best reply is d5.\n"
        "best_move: O-O\n"
        "coach_plan: Explain why h3 is slow, cite Black's d5 break, recommend castling\n"
        "</think>\n\n"
        "Playing h3 is a mistake that wastes critical time in the opening. "
        "Although White hopes to prevent pins along the kingside, Black can strike immediately with d5 to open the center. "
        "Instead of this passive move, White should play O-O to secure the king and complete development."
    )
    result = evaluate_candidate_completion(mock_prompt_item, raw_completion)
    assert result["accepted"] is True
    assert result["reward"] >= 0.82
    assert result["failed_gates"] == []
    assert result["word_count"] >= 40
    assert result["word_count"] <= 75


def test_evaluate_candidate_completion_missing_think(mock_prompt_item: dict[str, str]) -> None:
    """Verify evaluation rejects completion lacking <think> tags."""
    raw_completion = (
        "Playing h3 is a mistake that wastes critical time in the opening. "
        "Black can strike immediately with d5 to open the center. "
        "Instead, White should play O-O to castle safely."
    )
    result = evaluate_candidate_completion(mock_prompt_item, raw_completion)
    assert result["accepted"] is False
    assert "missing_thinking_tags" in result["failed_gates"]


def test_evaluate_candidate_completion_first_person_voice(mock_prompt_item: dict[str, str]) -> None:
    """Verify evaluation rejects completion containing first-person pronouns."""
    raw_completion = (
        "<think>\n"
        "verdict_or_eval: Mistake (wastes a tempo)\n"
        "main_fact: After h3, Black's best reply is d5.\n"
        "best_move: O-O\n"
        "coach_plan: Show refutation\n"
        "</think>\n\n"
        "I believe that h3 is a mistake because my analysis shows that Black plays d5. "
        "In my opinion, we should play O-O instead to protect our king."
    )
    result = evaluate_candidate_completion(mock_prompt_item, raw_completion)
    assert result["accepted"] is False
    assert "explanation_hard_violation" in result["failed_gates"]


def test_evaluate_candidate_completion_too_short(mock_prompt_item: dict[str, str]) -> None:
    """Verify evaluation rejects completion below minimum word threshold."""
    raw_completion = (
        "<think>\n"
        "verdict_or_eval: Mistake\n"
        "main_fact: d5\n"
        "best_move: O-O\n"
        "coach_plan: brief\n"
        "</think>\n\n"
        "h3 is a mistake. Play O-O instead."
    )
    result = evaluate_candidate_completion(mock_prompt_item, raw_completion)
    assert result["accepted"] is False
    assert "length_gate" in result["failed_gates"]


def test_filter_teacher_completions_multi_sample(mock_prompt_item: dict[str, str]) -> None:
    """Verify filter stops on first passing candidate in multi-sample list."""
    bad_cand = "h3 is bad because reasons."
    good_cand = (
        "<think>\n"
        "verdict_or_eval: Mistake (wastes a tempo in the opening)\n"
        "main_fact: After h3, Black's best reply is d5.\n"
        "best_move: O-O\n"
        "coach_plan: Explain why h3 is slow, cite Black's d5 break, recommend castling\n"
        "</think>\n\n"
        "Playing h3 is a mistake that wastes critical time in the opening. "
        "Although White hopes to prevent pins along the kingside, Black can strike immediately with d5 to open the center. "
        "Instead of this passive move, White should play O-O to secure the king and complete development."
    )

    prompts = [mock_prompt_item]
    completions_by_id = {
        mock_prompt_item["id"]: [bad_cand, good_cand]
    }

    nt_recs, th_recs, stats = filter_teacher_completions(
        prompts=prompts,
        completions_by_id=completions_by_id,
        max_candidates_per_prompt=4,
    )

    assert len(nt_recs) == 1
    assert len(th_recs) == 1
    assert stats["accepted_prompts"] == 1
    assert stats["total_completions_evaluated"] == 2
    assert nt_recs[0]["thinking_mode"] == "off"
    assert th_recs[0]["thinking_mode"] == "standard"


def test_evaluate_candidate_completion_trace_opening_claim(mock_prompt_item: dict[str, str]) -> None:
    """Verify evaluation rejects completion with unverified opening in thinking trace."""
    raw_completion = (
        "<think>\n"
        "verdict_or_eval: Mistake (wastes a tempo in the Sicilian Dragon)\n"
        "main_fact: After h3, Black's best reply is d5.\n"
        "best_move: O-O\n"
        "coach_plan: Explain why h3 is slow, cite Black's d5 break, recommend castling\n"
        "</think>\n\n"
        "Playing h3 is a mistake that wastes critical time in the opening. "
        "Although White hopes to prevent pins along the kingside, Black can strike immediately with d5 to open the center. "
        "Instead of this passive move, White should play O-O to secure the king and complete development."
    )
    result = evaluate_candidate_completion(mock_prompt_item, raw_completion)
    assert result["accepted"] is False
    assert "trace_hard_violation" in result["failed_gates"]
    assert result["trace_hard_checks"]["trace_opening_claims"] is False


def test_evaluate_candidate_completion_trace_relational_claim(mock_prompt_item: dict[str, str]) -> None:
    """Verify evaluation rejects completion with false relational claim in thinking trace."""
    raw_completion = (
        "<think>\n"
        "verdict_or_eval: Mistake (queen attacks the rook on a1)\n"
        "main_fact: After h3, Black's best reply is d5.\n"
        "best_move: O-O\n"
        "coach_plan: Explain why h3 is slow, cite Black's d5 break, recommend castling\n"
        "</think>\n\n"
        "Playing h3 is a mistake that wastes critical time in the opening. "
        "Although White hopes to prevent pins along the kingside, Black can strike immediately with d5 to open the center. "
        "Instead of this passive move, White should play O-O to secure the king and complete development."
    )
    result = evaluate_candidate_completion(mock_prompt_item, raw_completion)
    assert result["accepted"] is False
    assert "trace_hard_violation" in result["failed_gates"]
    assert result["trace_hard_checks"]["trace_relational_claims"] is False


def test_evaluate_mock_teacher_completion_passes(mock_prompt_item: dict[str, str]) -> None:
    """Verify mock generator completion passes all evaluation gates."""
    from data.pipeline.synthesize_teacher import generate_mock_teacher_completion

    mock_text = generate_mock_teacher_completion(mock_prompt_item)
    result = evaluate_candidate_completion(mock_prompt_item, mock_text)
    assert result["accepted"] is True
    assert result["reward"] >= 0.82
    assert result["failed_gates"] == []
