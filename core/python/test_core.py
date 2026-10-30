"""Unit tests for the Second chess shared core.

This test module verifies fact extraction, verdict classification, and prompt generation.
It tests winning chance calculations and opening book lookups.
It tests golden file parity when golden files exist.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import pytest

from core.python import (
    SYSTEM_MESSAGE,
    build_prompt_messages,
    build_user_message,
    classify_verdict,
    extract_facts,
    format_qwen_chat_prompt,
    win_chance,
)
from core.python.facts import (
    PIECE_VALUES,
    compute_checks_and_captures_facts,
    compute_hanging_facts,
    compute_king_safety_facts,
    compute_material_fact,
    compute_moved_piece_attack_fact,
    compute_phase_fact,
    compute_reply_and_material_change_facts,
    compute_threat_fact,
    round_pct,
    score_to_white_cp,
    uci_line_to_san,
    win_chance_side,
    win_chance_white,
)
from core.python.openings import OPENING_BOOK, lookup_opening


# --- Tests for Win Chances and Score Conversion ---


def test_score_to_white_cp_centipawns() -> None:
    """Verify centipawn score conversion."""
    score = {"type": "cp", "value": 150}
    assert score_to_white_cp(score) == 150.0

    score_neg = {"type": "cp", "value": -75}
    assert score_to_white_cp(score_neg) == -75.0


def test_score_to_white_cp_mate() -> None:
    """Verify mate score conversion to bounded centipawns."""
    mate_in_one = {"type": "mate", "value": 1}
    assert score_to_white_cp(mate_in_one) == 9990.0

    mate_in_three = {"type": "mate", "value": 3}
    assert score_to_white_cp(mate_in_three) == 9970.0

    mated_in_one = {"type": "mate", "value": -1}
    assert score_to_white_cp(mated_in_one) == -9990.0

    mate_zero = {"type": "mate", "value": 0}
    assert score_to_white_cp(mate_zero) == 0.0


def test_win_chance_symmetry() -> None:
    """Verify that White and Black winning chances sum to 100 percent."""
    score = {"type": "cp", "value": 65}
    w = win_chance(score, is_white=True)
    b = win_chance(score, is_white=False)
    assert math.isclose(w + b, 100.0, abs_tol=1e-6)


def test_win_chance_extremes() -> None:
    """Verify win chances near minimum and maximum bounds."""
    mate_score = {"type": "mate", "value": 1}
    assert win_chance_white(mate_score) > 99.9

    mated_score = {"type": "mate", "value": -1}
    assert win_chance_white(mated_score) < 0.1

    equal_score = {"type": "cp", "value": 0}
    assert math.isclose(win_chance_white(equal_score), 50.0, abs_tol=1e-6)


def test_round_pct() -> None:
    """Verify integer percentage rounding."""
    assert round_pct(50.4) == 50
    assert round_pct(50.5) == 51
    assert round_pct(50.6) == 51
    assert round_pct(0.0) == 0
    assert round_pct(100.0) == 100


# --- Tests for Move Quality Verdict Classification ---


def test_classify_verdict_only_move() -> None:
    """Verify Only move verdict when second PV loses winning chance."""
    eval_before = {
        "fen": "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "pvs": [
            {"score": {"type": "cp", "value": 100}, "moves": ["e2e4"]},
            {"score": {"type": "cp", "value": -200}, "moves": ["d2d4"]},
        ],
    }
    eval_after = {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "pvs": [{"score": {"type": "cp", "value": 95}, "moves": ["e7e5"]}],
    }
    verdict = classify_verdict(eval_before, eval_after, "e2e4")
    assert verdict == "Only move"


def test_classify_verdict_only_move_single_legal_move() -> None:
    """Verify Only move verdict when the board has only one legal move."""
    fen = "7K/r7/8/8/8/8/8/7k w - - 0 1"
    eval_before = {
        "fen": fen,
        "pvs": [{"score": {"type": "cp", "value": -1000}, "moves": ["h8g8"]}],
    }
    verdict = classify_verdict(eval_before, None, "h8g8")
    assert verdict == "Only move"


def test_classify_verdict_best() -> None:
    """Verify Best verdict when move matches top PV with low drop."""
    eval_before = {
        "fen": "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "pvs": [
            {"score": {"type": "cp", "value": 30}, "moves": ["e2e4"]},
            {"score": {"type": "cp", "value": 25}, "moves": ["d2d4"]},
        ],
    }
    eval_after = {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "pvs": [{"score": {"type": "cp", "value": 28}, "moves": ["e7e5"]}],
    }
    verdict = classify_verdict(eval_before, eval_after, "e2e4")
    assert verdict == "Best"


def test_classify_verdict_good() -> None:
    """Verify Good verdict for alternative moves with low drop."""
    eval_before = {
        "fen": "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "pvs": [
            {"score": {"type": "cp", "value": 30}, "moves": ["e2e4"]},
            {"score": {"type": "cp", "value": 25}, "moves": ["d2d4"]},
        ],
    }
    eval_after = {
        "fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        "pvs": [{"score": {"type": "cp", "value": 24}, "moves": ["d7d5"]}],
    }
    verdict = classify_verdict(eval_before, eval_after, "d2d4")
    assert verdict == "Good"


def test_classify_verdict_inaccuracy() -> None:
    """Verify Inaccuracy verdict for drops between 5 and 10 percent."""
    eval_before = {
        "fen": "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["e2e4"]}],
    }
    eval_after = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        "pvs": [{"score": {"type": "cp", "value": -80}, "moves": ["d7d5"]}],
    }
    verdict = classify_verdict(eval_before, eval_after, "g1f3")
    assert verdict == "Inaccuracy"


def test_classify_verdict_mistake() -> None:
    """Verify Mistake verdict for drops between 10 and 20 percent."""
    eval_before = {
        "fen": "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["e2e4"]}],
    }
    eval_after = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "pvs": [{"score": {"type": "cp", "value": -150}, "moves": ["e7e5"]}],
    }
    verdict = classify_verdict(eval_before, eval_after, "a2a3")
    assert verdict == "Mistake"


def test_classify_verdict_blunder() -> None:
    """Verify Blunder verdict for drops of at least 20 percent."""
    eval_before = {
        "fen": "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["e2e4"]}],
    }
    eval_after = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "pvs": [{"score": {"type": "cp", "value": -300}, "moves": ["e7e5"]}],
    }
    verdict = classify_verdict(eval_before, eval_after, "f2f3")
    assert verdict == "Blunder"


# --- Tests for Material Balance Facts ---


def test_compute_material_fact_equal() -> None:
    """Verify equal material description."""
    board = chess.Board()
    assert compute_material_fact(board) == "Material: equal."


def test_compute_material_fact_single_pieces() -> None:
    """Verify single piece advantage descriptions."""
    # White is up a pawn
    b1 = chess.Board("rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert compute_material_fact(b1) == "White is up a pawn."

    # Black is up two pawns
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PP1PP1PP/RNBQKBNR w KQkq - 0 1")
    assert compute_material_fact(b2) == "Black is up two pawns."

    # White is up a knight
    b3 = chess.Board("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert compute_material_fact(b3) == "White is up a knight."

    # Black is up a bishop
    b4 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1")
    assert compute_material_fact(b4) == "Black is up a bishop."

    # White is up a rook
    b5 = chess.Board("1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert compute_material_fact(b5) == "White is up a rook."

    # Black is up a queen
    b6 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    assert compute_material_fact(b6) == "Black is up a queen."


def test_compute_material_fact_exchange() -> None:
    """Verify exchange advantage description."""
    # White rook vs Black knight
    b = chess.Board("4k1n1/8/8/8/8/8/8/4K1R1 w - - 0 1")
    assert compute_material_fact(b) == "White is up the exchange."


def test_compute_material_fact_fallbacks() -> None:
    """Verify fallback descriptions for composite material advantages."""
    # White is up minor piece and a pawn (4 points)
    b4 = chess.Board("4k3/8/8/8/8/8/4P3/4KN2 w - - 0 1")
    assert compute_material_fact(b4) == "White is up a minor piece and a pawn in material."

    # White is up two minor pieces (6 points)
    b6 = chess.Board("4k3/8/8/8/8/8/8/4KBB1 w - - 0 1")
    assert compute_material_fact(b6) == "White is up two minor pieces in material."

    # White is up a rook and two pawns (7 points)
    b7 = chess.Board("4k3/8/8/8/8/8/3PP3/4K1R1 w - - 0 1")
    assert compute_material_fact(b7) == "White is up a rook and two pawns in material."

    # White is up a rook and a minor piece (8 points)
    b8 = chess.Board("4k3/8/8/8/8/8/8/4K1RN w - - 0 1")
    assert compute_material_fact(b8) == "White is up a rook and a minor piece in material."


# --- Tests for Game Phase Facts ---


def test_compute_phase_fact() -> None:
    """Verify phase detection for opening, middlegame, and endgame."""
    b_open = chess.Board()
    assert compute_phase_fact(b_open) == "Phase: opening."

    b_end = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 50")
    assert compute_phase_fact(b_end) == "Phase: endgame."

    b_mid = chess.Board("r1b1k2r/pp1n1ppp/2p1pn2/q2p4/2PP4/2N1PN2/PP1Q1PPP/R3KB1R w KQkq - 1 12")
    assert compute_phase_fact(b_mid) == "Phase: middlegame."


# --- Tests for King Safety Facts ---


def test_compute_king_safety_facts() -> None:
    """Verify check reporting and castling rights reporting."""
    # King in check
    b_chk = chess.Board("rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3")
    assert compute_king_safety_facts(b_chk) == ["Black king is in check on e8."]

    # Uncastled kings on move ten
    b_nocastle = chess.Board("r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 10 10")
    assert compute_king_safety_facts(b_nocastle) == [
        "White has not castled.",
        "Black has not castled.",
    ]


# --- Tests for Hanging Pieces Facts ---


def test_compute_hanging_facts() -> None:
    """Verify detection of undefended attacked pieces."""
    b = chess.Board("rnbqkb1r/ppp1pppp/3p4/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    facts = compute_hanging_facts(b)
    assert facts == ["The knight on e5 is not defended."]


# --- Tests for Checks and Captures Facts ---


def test_compute_checks_and_captures_facts() -> None:
    """Verify detection of legal checks and high value captures."""
    b = chess.Board("r1bqkb1r/pppp1ppp/8/4p3/2Bn4/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5")
    facts = compute_checks_and_captures_facts(b)
    assert "White has a check: Bxf7+." in facts
    assert "White can capture the knight on d4 with Nxd4." in facts


# --- Tests for Threats Facts ---


def test_compute_threat_fact_mate() -> None:
    """Verify checkmate threat identification."""
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3")
    threat = compute_threat_fact(b, chess.WHITE)
    assert threat == "White threatens Qxf7#."


def test_compute_threat_fact_winning_capture() -> None:
    """Verify winning capture threat identification."""
    b = chess.Board("4k3/1ppppppp/8/n7/8/8/1PPPPPPP/R3KBNR b - - 0 1")
    threat = compute_threat_fact(b, chess.WHITE)
    assert threat == "White threatens Rxa5."


def test_compute_threat_fact_safe_check() -> None:
    """Verify safe check threat identification."""
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4")
    threat = compute_threat_fact(b, chess.WHITE)
    assert threat == "White threatens Bxf7+."


# --- Tests for Moved-Piece Attack Facts ---


def test_compute_moved_piece_attack_fact() -> None:
    """Verify attack facts generated after a move."""
    b_before = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    move = chess.Move.from_uci("f3g5")
    b_after = b_before.copy()
    b_after.push(move)

    attack_fact = compute_moved_piece_attack_fact(b_before, move, b_after)
    assert attack_fact == "Ng5 attacks the pawn on f7 together with the bishop on c4."


def test_compute_moved_piece_attack_fact_king_returns_none() -> None:
    """Verify king moves never generate an attack fact."""
    b_before = chess.Board("4k3/8/8/8/8/4n3/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("e1f2")
    b_after = b_before.copy()
    b_after.push(move)
    assert compute_moved_piece_attack_fact(b_before, move, b_after) is None


# --- Tests for Opponent Best Reply and Material Changes ---


def test_compute_reply_and_material_change_facts() -> None:
    """Verify opponent reply and subsequent material win facts."""
    b_before = chess.Board("r1bqkb1r/pp1p1ppp/2n5/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 5")
    move = chess.Move.from_uci("f3e5")
    b_after = b_before.copy()
    b_after.push(move)

    eval_after = {
        "pvs": [
            {
                "score": {"type": "cp", "value": -150},
                "moves": ["d8a5", "b1c3", "a5e5"],
            }
        ]
    }
    facts = compute_reply_and_material_change_facts(b_before, move, b_after, eval_after)
    assert len(facts) == 2
    assert facts[0] == "After Nxe5, Black's best reply is Qa5."
    assert facts[1] == "After Qa5, Black wins the knight on e5."


def test_compute_reply_and_material_change_facts_en_passant() -> None:
    """Verify en passant capture is handled in reply material changes."""
    b_before = chess.Board("rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2")
    move = chess.Move.from_uci("d7d5")
    b_after = b_before.copy()
    b_after.push(move)
    eval_after = {
        "pvs": [
            {
                "score": {"type": "cp", "value": 150},
                "moves": ["e5d6"],
            }
        ]
    }
    facts = compute_reply_and_material_change_facts(b_before, move, b_after, eval_after)
    assert len(facts) == 2
    assert facts[0] == "After d5, White's best reply is exd6."
    assert facts[1] == "After d5, White captures the pawn on d6 with exd6."


# --- Tests for Opening Book Lookup ---


def test_lookup_opening() -> None:
    """Verify opening lookup with direct FEN and move pushing."""
    assert len(OPENING_BOOK) == 49

    # Lookup start position after 1. e4
    fen_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert lookup_opening(fen_e4) == "King's Pawn Game"

    # Lookup move into Sicilian Defense
    fen_start = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert lookup_opening(fen_start, "c7c5") == "Sicilian Defense"

    # Unknown opening returns None
    assert lookup_opening("8/8/8/8/8/8/8/4K2k w - - 0 1") is None


# --- Tests for Full Prompt Formatting ---


def test_build_user_message_explain_move() -> None:
    """Verify explain_move user message formatting."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    eval_before = {
        "fen": fen,
        "pvs": [
            {"score": {"type": "cp", "value": 45}, "moves": ["d2d3", "f8c5", "c2c3", "d7d6"]},
            {"score": {"type": "cp", "value": 40}, "moves": ["b1c3"]},
        ],
    }
    eval_after = {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4",
        "pvs": [
            {"score": {"type": "cp", "value": 35}, "moves": ["d7d5", "e4d5", "c6a5"]}
        ],
    }
    msg = build_user_message("explain_move", "club", "short", fen, eval_before, eval_after, "f3g5")
    lines = msg.split("\n")

    assert lines[0] == "task: explain_move"
    assert lines[1] == "level: club"
    assert lines[2] == "detail: short"
    assert lines[3] == ""
    assert lines[4] == f"FEN: {fen}"
    assert lines[5] == "White to move, move 4."
    assert lines[6] == "Move played: Ng5."
    assert lines[7] == "Verdict: Good."
    assert lines[8] == "White's winning chances: 54% \u2192 53%."
    assert lines[9] == "Best move: d3."
    assert lines[10] == "Best line: d3 Bc5 c3 d6."
    assert lines[11] == "Facts:"
    assert "- Ng5 attacks the pawn on f7 together with the bishop on c4." in lines
    assert "- After Ng5, Black's best reply is d5." in lines
    assert "- Material: equal." in lines
    assert "- Phase: opening." in lines
    assert "- Opening: Italian Game, Two Knights Defense." in lines


def test_build_user_message_explain_position() -> None:
    """Verify explain_position user message formatting."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    eval_before = {
        "fen": fen,
        "pvs": [
            {"score": {"type": "cp", "value": 45}, "moves": ["d2d3", "f8c5"]},
            {"score": {"type": "cp", "value": 40}, "moves": ["b1c3", "f8c5"]},
        ],
    }
    msg = build_user_message("explain_position", "beginner", "long", fen, eval_before)
    lines = msg.split("\n")

    assert lines[0] == "task: explain_position"
    assert lines[1] == "level: beginner"
    assert lines[2] == "detail: long"
    assert lines[3] == ""
    assert lines[4] == f"FEN: {fen}"
    assert lines[5] == "White to move, move 4."
    assert lines[6] == "White's winning chances: 54%."
    assert lines[7] == "Best move: d3."
    assert lines[8] == "Best line: d3 Bc5."
    assert lines[9] == "Facts:"
    assert "- Top candidate moves: d3, Nc3." in lines
    assert "- Material: equal." in lines


def test_build_prompt_messages_and_chat_prompt() -> None:
    """Verify message structure and Qwen chat template formatting."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    eval_before = {
        "fen": fen,
        "pvs": [{"score": {"type": "cp", "value": 15}, "moves": ["c7c5"]}],
    }
    messages = build_prompt_messages("explain_position", "club", "short", fen, eval_before)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_MESSAGE
    assert messages[1]["role"] == "user"

    chat_prompt = format_qwen_chat_prompt(messages[1]["content"])
    assert chat_prompt.startswith(f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n")
    assert chat_prompt.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


# --- Golden Parity Tests ---


def test_golden_parity() -> None:
    """Verify byte-for-byte parity against golden file."""
    golden_path = PROJECT_ROOT / "core" / "golden" / "golden_200.json"
    assert golden_path.exists(), f"Golden file not found at {golden_path}."

    with open(golden_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 200, f"Expected 200 golden cases, got {len(cases)}."

    for idx, case in enumerate(cases):
        task = case["task"]
        level = case["level"]
        detail = case["detail"]
        fen = case["fen"]
        eval_before = case["eval_before"]
        eval_after = case.get("eval_after")
        move_uci = case.get("move_uci")

        actual_user_msg = build_user_message(
            task=task,
            level=level,
            detail=detail,
            fen=fen,
            eval_before=eval_before,
            eval_after=eval_after,
            move_uci=move_uci,
        )
        assert actual_user_msg == case["expected_user_message"], f"User message mismatch on case {idx}"

        actual_chat_prompt = format_qwen_chat_prompt(actual_user_msg)
        assert actual_chat_prompt == case["expected_chat_prompt"], f"Chat prompt mismatch on case {idx}"
