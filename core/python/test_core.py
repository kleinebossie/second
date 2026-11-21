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
    compute_line_comparison_fact,
    compute_material_fact,
    compute_move_positional_fact,
    compute_moved_piece_attack_fact,
    compute_phase_fact,
    compute_position_structure_fact,
    compute_recapture_activity_fact,
    compute_refutation_reason_fact,
    compute_reply_and_material_change_facts,
    compute_threat_fact,
    round_pct,
    score_to_white_cp,
    uci_line_to_san,
    win_chance_side,
    win_chance_white,
)
from core.python.openings import OPENING_BOOK, lookup_opening
from rewards.checks import check_piece_on_square


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
    """Verify noisy checks are filtered while supported or forcing checks remain."""
    b = chess.Board("r1bqkb1r/pppp1ppp/8/4p3/2Bn4/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5")
    facts = compute_checks_and_captures_facts(b)
    assert "White has a check: Bxf7+." not in facts
    assert "White can capture the knight on d4 with Nxd4." in facts


def test_compute_checks_and_captures_stockfish_pv_and_drop() -> None:
    """Verify PV checks within one hundred centipawns qualify and weak PV checks drop."""
    b = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5")
    ev_close = {
        "fen": b.fen(),
        "pvs": [
            {"score": {"type": "cp", "value": 25}, "moves": ["c1g5", "h7h6"]},
            {"score": {"type": "cp", "value": -40}, "moves": ["c4f7", "e8f7"]},
        ],
    }
    assert "White has a check: Bxf7+." in compute_checks_and_captures_facts(b, ev_close)

    ev_far = {
        "fen": b.fen(),
        "pvs": [
            {"score": {"type": "cp", "value": 30}, "moves": ["c1g5", "h7h6"]},
            {"score": {"type": "cp", "value": -120}, "moves": ["c4f7", "e8f7"]},
        ],
    }
    assert "White has a check: Bxf7+." not in compute_checks_and_captures_facts(b, ev_far)


def test_compute_checks_and_captures_forcing_checks() -> None:
    """Verify mate, material-winning, royal-fork, and castling-strip checks qualify."""
    # Checkmate check
    b_mate = chess.Board("6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1")
    assert "White has a check: Rd8#." in compute_checks_and_captures_facts(b_mate)

    # Material-winning check (pawn captures knight with check)
    b_win_cap = chess.Board("8/4k3/3n4/4P3/8/8/8/4K3 w - - 0 1")
    assert "White has a check: exd6+." in compute_checks_and_captures_facts(b_win_cap)

    # Safe royal fork check
    b_fork = chess.Board("r3k2r/ppp2ppp/8/3N4/8/8/PPP2PPP/R1B1K2R w KQkq - 0 10")
    assert "White has a check: Nxc7+." in compute_checks_and_captures_facts(b_fork)

    # Safe check forcing king-only replies and stripping castling rights
    b_castle_strip = chess.Board("4k2r/4pppp/8/8/2B5/8/8/4K3 w k - 0 15")
    assert "White has a check: Bb5+." in compute_checks_and_captures_facts(b_castle_strip)


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
    assert facts[1] == "After d5, White captures the pawn on d5 with exd6."


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


def test_build_user_message_explain_move_capture() -> None:
    """Verify material reflects the position after a capture move."""
    fen = "r4rk1/4ppbp/2p3p1/1p1Q4/p2P4/2P2N2/PPK2PPP/3R3R b - - 0 19"
    eval_before = {
        "fen": fen,
        "pvs": [
            {"score": {"type": "cp", "value": 0}, "moves": ["c6d5", "f3e5"]}
        ],
    }
    eval_after = {
        "fen": "r4rk1/4ppbp/6p1/1p1p4/p2P4/2P2N2/PPK2PPP/3R3R w - - 0 20",
        "pvs": [
            {"score": {"type": "cp", "value": 0}, "moves": ["f3e5"]}
        ],
    }
    msg = build_user_message("explain_move", "club", "short", fen, eval_before, eval_after, "c6d5")
    lines = msg.split("\n")

    assert "Move played: cxd5." in lines
    assert "- Material: equal." in lines
    assert "- White is up a queen in material." not in lines


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


# --- Tests for Positional and Phase 11 Facts ---


def test_compute_recapture_activity_fact() -> None:
    """Verify detection of opponent recapture centralization and activity gain."""
    b_before = chess.Board("rnb1k2r/ppppqppp/5n2/4p3/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R b KQkq - 0 6")
    move = chess.Move.from_uci("e5d4")
    b_after = b_before.copy()
    b_after.push(move)
    eval_after = {"pvs": [{"score": {"type": "cp", "value": 74}, "moves": ["f3d4", "e8g8"]}]}
    fact = compute_recapture_activity_fact(b_before, move, b_after, eval_after)
    assert fact == "After exd4, White's recapture Nxd4 centralizes the knight."

    b_act = chess.Board("r2qk2r/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 0 7")
    m_act = chess.Move.from_uci("g4f3")
    ba_act = b_act.copy()
    ba_act.push(m_act)
    ev_act = {"pvs": [{"score": {"type": "cp", "value": 70}, "moves": ["d1f3", "c6d4"]}]}
    assert (
        compute_recapture_activity_fact(b_act, m_act, ba_act, ev_act)
        == "After Bxf3, White's recapture Qxf3 increases queen activity."
    )


def test_compute_refutation_reason_fact_positional() -> None:
    """Verify refutation facts cover central space, castling, safety, and open files."""
    # 6... exd4 vs 6... e4
    b = chess.Board("rnb1k2r/ppppqppp/5n2/4p3/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R b KQkq - 0 6")
    m = chess.Move.from_uci("e5d4")
    ba = b.copy()
    ba.push(m)
    eb = {"fen": b.fen(), "pvs": [{"score": {"type": "cp", "value": 16}, "moves": ["e5e4", "f3g1"]}]}
    ea = {"fen": ba.fen(), "pvs": [{"score": {"type": "cp", "value": 74}, "moves": ["f3d4"]}]}
    assert (
        compute_refutation_reason_fact(b, m, ba, eb, ea)
        == "Best move e4 gains space and attacks the knight on f3 instead of trading in the center."
    )

    # Rook on open file refutation
    b_rook = chess.Board("3r2k1/pp3ppp/2p5/8/8/2P5/PP3PPP/R4RK1 w - - 0 20")
    m_rook = chess.Move.from_uci("a2a3")
    ba_rook = b_rook.copy()
    ba_rook.push(m_rook)
    eb_rook = {"fen": b_rook.fen(), "pvs": [{"score": {"type": "cp", "value": 45}, "moves": ["a1d1"]}]}
    ea_rook = {"fen": ba_rook.fen(), "pvs": [{"score": {"type": "cp", "value": -60}, "moves": ["d8d2"]}]}
    assert (
        compute_refutation_reason_fact(b_rook, m_rook, ba_rook, eb_rook, ea_rook)
        == "Best move Rad1 places the rook on an open file."
    )


def test_compute_line_comparison_fact() -> None:
    """Verify line comparison identifies centralization and central pawn differences."""
    b = chess.Board("rnb1k2r/ppppqppp/5n2/4p3/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R b KQkq - 0 6")
    m = chess.Move.from_uci("e5d4")
    ba = b.copy()
    ba.push(m)
    eb = {"fen": b.fen(), "pvs": [{"score": {"type": "cp", "value": 16}, "moves": ["e5e4", "f3g1", "b4c3", "d2c3"]}]}
    ea = {"fen": ba.fen(), "pvs": [{"score": {"type": "cp", "value": 74}, "moves": ["f3d4", "e8g8", "f1d3"]}]}
    assert (
        compute_line_comparison_fact(b, m, ba, eb, ea)
        == "Unlike exd4, e4 prevents White from centralizing pieces."
    )


def test_compute_move_positional_and_structure_facts() -> None:
    """Verify quiet move positional facts and board structure facts."""
    b_castle = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    m_castle = chess.Move.from_uci("e1g1")
    ba_castle = b_castle.copy()
    ba_castle.push(m_castle)
    assert compute_move_positional_fact(b_castle, m_castle, ba_castle) == "O-O castles to safeguard the king."

    b_tension = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3")
    assert compute_position_structure_fact(b_tension) == "Central pawn tension exists between d4 and e5."

    b_space = chess.Board("rnbqkbnr/pp3ppp/2p1p3/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5")
    assert compute_position_structure_fact(b_space) == "White has more central space."


def test_edge_cases_checks_rook_files_line_comparison_and_illegal_move() -> None:
    """Verify en passant check, defended promo check, O-O+, half-open->open rook, mate comparison, and illegal move."""
    # En passant check (defended by Nc4 -> safe capture)
    b_ep = chess.Board("8/4k3/8/2Pp4/2N5/8/8/6K1 w - d6 0 1")
    assert compute_checks_and_captures_facts(b_ep) == ["White has a check: cxd6+."]

    # Promotion with check: undefended picks =Q+ over =R+ and =N without duplicating same-square underpromotions
    b_p_undef = chess.Board("3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    assert compute_checks_and_captures_facts(b_p_undef) == [
        "White has a check: exd8=Q+.",
    ]
    b_p_def = chess.Board("q2r3k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    assert compute_checks_and_captures_facts(b_p_def) == [
        "White has a check: exd8=Q+.",
    ]

    # Castling check via Stockfish PV
    b_cas = chess.Board("5k2/8/8/8/8/8/8/4K2R w K - 0 1")
    assert compute_checks_and_captures_facts(
        b_cas,
        {"pvs": [{"score": {"type": "cp", "value": 120}, "moves": ["e1g1"]}]},
    ) == ["White has a check: O-O+."]

    # Illegal move_uci throws ValueError in both extract_facts and build_user_message
    fen_start = chess.STARTING_FEN
    eval_dummy = {"fen": fen_start, "pvs": [{"score": {"type": "cp", "value": 20}, "moves": ["e2e4"]}]}
    with pytest.raises(ValueError):
        extract_facts("explain_move", fen_start, eval_dummy, eval_dummy, "e2e5")
    with pytest.raises(ValueError):
        build_user_message("explain_move", "club", "short", fen_start, eval_dummy, eval_dummy, "e2e5")

    # Rook moves from half-open file (a7 Black pawn) to fully open d-file
    b_rk = chess.Board("3k4/p4ppp/8/8/8/8/PP3PPP/R5K1 w - - 0 20")
    m_bad = chess.Move.from_uci("a2a3")
    ba_bad = b_rk.copy()
    ba_bad.push(m_bad)
    eb_rk = {"fen": b_rk.fen(), "pvs": [{"score": {"type": "cp", "value": 60}, "moves": ["a1d1"]}]}
    ea_rk = {"fen": ba_bad.fen(), "pvs": [{"score": {"type": "cp", "value": -20}, "moves": ["d8c7"]}]}
    assert compute_refutation_reason_fact(b_rk, m_bad, ba_bad, eb_rk, ea_rk) == (
        "Best move Rd1+ places the rook on an open file."
    )
    m_ad1 = chess.Move.from_uci("a1d1")
    ba_ad1 = b_rk.copy()
    ba_ad1.push(m_ad1)
    assert compute_move_positional_fact(b_rk, m_ad1, ba_ad1) == "Rd1+ places the rook on the open d-file."

    # Line comparison: checkmate delivered, checkmate avoided, mat_diff == 2 labeled as material
    b_mate = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 25")
    mh3 = chess.Move.from_uci("h2h3")
    bah3 = b_mate.copy()
    bah3.push(mh3)
    assert compute_line_comparison_fact(
        b_mate,
        mh3,
        bah3,
        {"fen": b_mate.fen(), "pvs": [{"score": {"type": "mate", "value": 1}, "moves": ["d1d8"]}]},
        {"fen": bah3.fen(), "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["h7h6", "g1h2"]}]},
    ) == "Unlike h3, the best line with Rd8# leads to checkmate."

    b_av = chess.Board("4r1k1/5ppp/8/8/8/8/PR3PPP/6K1 w - - 0 25")
    ma3 = chess.Move.from_uci("a2a3")
    baa3 = b_av.copy()
    baa3.push(ma3)
    assert compute_line_comparison_fact(
        b_av,
        ma3,
        baa3,
        {"fen": b_av.fen(), "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["g2g3", "e8e1", "g1g2", "g7g6"]}]},
        {"fen": baa3.fen(), "pvs": [{"score": {"type": "mate", "value": -1}, "moves": ["e8e1"]}]},
    ) == "Unlike a3, g3 avoids checkmate."

    # Exchange loss (mat_diff == 2) emits "material", never "a pawn"
    b_ex = chess.Board("3r2k1/pp3ppp/8/3N4/8/8/PP3PPP/3R2K1 b - - 0 22")
    mrxd5 = chess.Move.from_uci("d8d5")
    barxd5 = b_ex.copy()
    barxd5.push(mrxd5)
    assert compute_line_comparison_fact(
        b_ex,
        mrxd5,
        barxd5,
        {"fen": b_ex.fen(), "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["g8f8", "d5e3", "g7g6", "d1c1"]}]},
        {"fen": barxd5.fen(), "pvs": [{"score": {"type": "cp", "value": 100}, "moves": ["d1d5", "g8f8", "g1f1"]}]},
    ) == "Unlike Rxd5, the best line with Kf8 avoids losing material."

    # Refutation piece capture (Bxc6) does not claim "instead of trading" and f3 does not claim "gains central space"
    b_bxc6 = chess.Board("r1bqkb1r/pp1p1ppp/2n1pn2/1Bp5/3PP3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 5")
    mbxc6 = chess.Move.from_uci("b5c6")
    babxc6 = b_bxc6.copy()
    babxc6.push(mbxc6)
    ref_bxc6 = compute_refutation_reason_fact(
        b_bxc6,
        mbxc6,
        babxc6,
        {"fen": b_bxc6.fen(), "pvs": [{"score": {"type": "cp", "value": 75}, "moves": ["c2c4"]}]},
        {"fen": babxc6.fen(), "pvs": [{"score": {"type": "cp", "value": -20}, "moves": ["d7c6"]}]},
    )
    assert ref_bxc6 is not None and "instead of trading" not in ref_bxc6

    b_f3n = chess.Board("rnbqkb1r/pppp1ppp/8/4p3/4P1n1/8/PPPP1PPP/RNBQKBNR w KQkq - 2 4")
    mf3_bad = chess.Move.from_uci("a2a3")
    baf3_bad = b_f3n.copy()
    baf3_bad.push(mf3_bad)
    assert compute_refutation_reason_fact(
        b_f3n,
        mf3_bad,
        baf3_bad,
        {"fen": b_f3n.fen(), "pvs": [{"score": {"type": "cp", "value": 60}, "moves": ["f2f3"]}]},
        {"fen": baf3_bad.fen(), "pvs": [{"score": {"type": "cp", "value": -30}, "moves": ["g4f6"]}]},
    ) == "Best move f3 attacks the knight on g4."

    # Odd-ply PV comparison trims to even plies so uncompensated half-move capture is not called winning material
    b_odd = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    ma3_odd = chess.Move.from_uci("a2a3")
    baa3_odd = b_odd.copy()
    baa3_odd.push(ma3_odd)
    assert compute_line_comparison_fact(
        b_odd,
        ma3_odd,
        baa3_odd,
        {"fen": b_odd.fen(), "pvs": [{"score": {"type": "cp", "value": 40}, "moves": ["f1b5", "c8d7", "b5c6"]}]},
        {"fen": baa3_odd.fen(), "pvs": [{"score": {"type": "cp", "value": -10}, "moves": ["d7d6", "f1c4"]}]},
    ) is None

    # Pawn capturing non-pawn piece does not claim "releases central tension"
    b_ptp = chess.Board("4k3/8/8/4n3/3P4/8/8/4K3 w - - 0 1")
    m_dxe5_n = chess.Move.from_uci("d4e5")
    ba_dxe5_n = b_ptp.copy()
    ba_dxe5_n.push(m_dxe5_n)
    assert compute_move_positional_fact(b_ptp, m_dxe5_n, ba_dxe5_n) is None

    b_ptp_pawn = chess.Board("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1")
    m_dxe5_p = chess.Move.from_uci("d4e5")
    ba_dxe5_p = b_ptp_pawn.copy()
    ba_dxe5_p.push(m_dxe5_p)
    assert compute_move_positional_fact(b_ptp_pawn, m_dxe5_p, ba_dxe5_p) == "dxe5 releases central tension on e5."

    # Rook capture removing last enemy pawn on destination file claims open file post-move
    b_ropen = chess.Board("6k1/R3p3/8/8/8/8/P7/6K1 w - - 0 1")
    m_rxe7 = chess.Move.from_uci("a7e7")
    ba_rxe7 = b_ropen.copy()
    ba_rxe7.push(m_rxe7)
    assert compute_move_positional_fact(b_ropen, m_rxe7, ba_rxe7) == "Rxe7 places the rook on the open e-file."

    # Chess960 king-takes-rook castling UCI rejected on standard FEN; standard e1g1 accepted
    fen_c960 = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    ev_c960 = {"fen": fen_c960, "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["e1g1"]}]}
    assert extract_facts("explain_move", fen_c960, ev_c960, ev_c960, "e1g1")
    with pytest.raises(ValueError):
        extract_facts("explain_move", fen_c960, ev_c960, ev_c960, "e1h1")
    with pytest.raises(ValueError):
        build_user_message("explain_move", "club", "short", fen_c960, ev_c960, ev_c960, "e1h1")

    # 4-ply line ending in a capture immediately recaptured on ply 5 (dev_152 scenario) does not claim material loss
    b_dev152 = chess.Board("r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQ1RK1 w kq - 2 7")
    m_c4 = chess.Move.from_uci("c2c4")
    ba_c4 = b_dev152.copy()
    ba_c4.push(m_c4)
    eb_dev152 = {
        "fen": b_dev152.fen(),
        "pvs": [
            {
                "score": {"type": "cp", "value": 30},
                "moves": ["f1e1", "h7h6", "h2h3", "c6a5", "b3c2", "c7c5"],
            }
        ],
    }
    ea_dev152 = {
        "fen": ba_c4.fen(),
        "pvs": [
            {
                "score": {"type": "cp", "value": -34},
                "moves": ["c6a5", "c1d2", "a5b3", "a2b3", "c8e6"],
            }
        ],
    }
    lc_dev152 = compute_line_comparison_fact(b_dev152, m_c4, ba_c4, eb_dev152, ea_dev152)
    assert lc_dev152 is None or "avoids losing material" not in lc_dev152

    # When mover is already down a pawn (mat_before == -1) and best line wins back to 0, emit "wins a pawn."
    b_down_p = chess.Board("4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1")
    m_h3 = chess.Move.from_uci("e1f1")
    ba_h3 = b_down_p.copy()
    ba_h3.push(m_h3)
    lc_down_p = compute_line_comparison_fact(
        b_down_p,
        m_h3,
        ba_h3,
        {"fen": b_down_p.fen(), "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["f3e5", "e8e7"]}]},
        {"fen": ba_h3.fen(), "pvs": [{"score": {"type": "cp", "value": -120}, "moves": ["e8e7"]}]},
    )
    assert lc_down_p == "Unlike Kf1, the best line with Nxe5 wins a pawn."


# --- Golden Parity Tests ---


def _assert_golden_file(filename: str, expected_count: int) -> None:
    """Verify byte-for-byte prompt parity, PV legality, and Rule 9 compliance for a golden file."""
    golden_path = PROJECT_ROOT / "core" / "golden" / filename
    assert golden_path.exists(), f"Golden file not found at {golden_path}."

    with open(golden_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == expected_count, f"Expected {expected_count} cases in {filename}, got {len(cases)}."

    for idx, case in enumerate(cases):
        board_chk = chess.Board(case["fen"])
        assert board_chk.is_valid(), f"{filename} case {idx} invalid board FEN"
        for pv in case["eval_before"].get("pvs", []):
            sim = board_chk.copy(stack=False)
            for uci in pv.get("moves", []):
                mv = chess.Move.from_uci(uci)
                assert mv in sim.legal_moves, f"{filename} case {idx} illegal eval_before PV move {uci}"
                sim.push(mv)

        if case.get("move_uci"):
            played_mv = chess.Move.from_uci(case["move_uci"])
            assert played_mv in board_chk.legal_moves, f"{filename} case {idx} illegal move_uci {case['move_uci']}"
            after_chk = board_chk.copy(stack=False)
            after_chk.push(played_mv)
            ea = case.get("eval_after")
            if ea is not None:
                if ea.get("fen"):
                    assert chess.Board(ea["fen"]).fen() == after_chk.fen(), (
                        f"{filename} case {idx} eval_after.fen mismatch"
                    )
                for pv in ea.get("pvs", []):
                    sim_after = after_chk.copy(stack=False)
                    for uci in pv.get("moves", []):
                        mv = chess.Move.from_uci(uci)
                        assert mv in sim_after.legal_moves, (
                            f"{filename} case {idx} illegal eval_after PV move {uci}"
                        )
                        sim_after.push(mv)

        actual_user_msg = build_user_message(
            task=case["task"],
            level=case["level"],
            detail=case["detail"],
            fen=case["fen"],
            eval_before=case["eval_before"],
            eval_after=case.get("eval_after"),
            move_uci=case.get("move_uci"),
        )
        assert actual_user_msg == case["expected_user_message"], f"{filename} user message mismatch on case {idx}"

        actual_chat_prompt = format_qwen_chat_prompt(actual_user_msg)
        assert actual_chat_prompt == case["expected_chat_prompt"], f"{filename} chat prompt mismatch on case {idx}"

        facts = extract_facts(
            task=case["task"],
            fen=case["fen"],
            eval_before=case["eval_before"],
            eval_after=case.get("eval_after"),
            move_uci=case.get("move_uci"),
        )
        chk = check_piece_on_square(" ".join(facts), case["fen"], case.get("move_uci"))
        assert chk.passed, f"{filename} case {idx} Rule 9 violation: {chk.reason}"


def test_golden_parity() -> None:
    """Verify byte-for-byte parity and Rule 9 compliance on golden_200.json."""
    _assert_golden_file("golden_200.json", 200)


def test_tactical_golden_parity() -> None:
    """Verify byte-for-byte parity and Rule 9 compliance on golden_tactical_30.json."""
    _assert_golden_file("golden_tactical_30.json", 30)


def test_positional_golden_parity() -> None:
    """Verify byte-for-byte parity and Rule 9 compliance on golden_positional_30.json."""
    _assert_golden_file("golden_positional_30.json", 30)
