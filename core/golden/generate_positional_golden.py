"""Generate thirty positional and check-filter golden test cases for v1.1.

This script builds parity test cases for Phase 11 fact extractor upgrades.
It covers check filtering, recapture activity, refutation space, and line comparison.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess

from core.python import build_user_message, format_qwen_chat_prompt


def _fen_after(moves_uci: list[str]) -> str:
    """Play UCI moves from the starting position and return the final FEN."""
    board = chess.Board()
    for uci in moves_uci:
        board.push_uci(uci)
    return board.fen()


def _after_move(fen: str, move_uci: str) -> str:
    """Play one UCI move from fen and return the resulting FEN."""
    board = chess.Board(fen)
    board.push_uci(move_uci)
    return board.fen()


def build_positional_cases() -> list[dict[str, Any]]:
    """Build thirty targeted Phase 11 positional and check-filter cases."""
    bogo_fen = _fen_after(
        ["d2d4", "g8f6", "g1f3", "e7e6", "c2c4", "f8b4", "c1d2", "d8e7", "b1c3", "e6e5", "e2e3"]
    )
    bogo_after_exd4 = _after_move(bogo_fen, "e5d4")

    # 0. Canonical 6... exd4 vs 6... e4 positional inaccuracy
    case_0 = {
        "task": "explain_move",
        "fen": bogo_fen,
        "move_uci": "e5d4",
        "eval_before": {
            "fen": bogo_fen,
            "pvs": [
                {"score": {"type": "cp", "value": 16}, "moves": ["e5e4", "f3g1", "b4c3", "d2c3"]},
                {"score": {"type": "cp", "value": 74}, "moves": ["e5d4", "f3d4", "e8g8"]},
            ],
        },
        "eval_after": {
            "fen": bogo_after_exd4,
            "pvs": [
                {"score": {"type": "cp", "value": 74}, "moves": ["f3d4", "e8g8", "f1d3", "b8c6"]}
            ],
        },
        "positional_tag": "bogo_6_exd4_vs_e4",
    }

    # 1. Noisy check suppression: Bxf7+ is unsafe sacrifice, not in PV
    fen_1 = "r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 4"
    case_1 = {
        "task": "explain_move",
        "fen": fen_1,
        "move_uci": "f3d4",
        "eval_before": {
            "fen": fen_1,
            "pvs": [{"score": {"type": "cp", "value": 65}, "moves": ["f3d4", "e5d4", "e1g1", "g8f6"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_1, "f3d4"),
            "pvs": [{"score": {"type": "cp", "value": 65}, "moves": ["e5d4", "e1g1", "g8f6"]}],
        },
        "positional_tag": "suppress_noisy_check_after_move",
    }

    # 2. Noisy check suppressed in explain_position when PV omits junk check Bxf7+
    case_2 = {
        "task": "explain_position",
        "fen": fen_1,
        "move_uci": None,
        "eval_before": {
            "fen": fen_1,
            "pvs": [
                {"score": {"type": "cp", "value": 65}, "moves": ["f3d4", "e5d4"]},
                {"score": {"type": "cp", "value": 10}, "moves": ["e1g1", "d4f3"]},
            ],
        },
        "eval_after": None,
        "positional_tag": "suppress_noisy_check_position",
    }

    # 3. Valid check retained via Stockfish PV2 within 100 cp
    fen_3 = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"
    case_3 = {
        "task": "explain_position",
        "fen": fen_3,
        "move_uci": None,
        "eval_before": {
            "fen": fen_3,
            "pvs": [
                {"score": {"type": "cp", "value": 25}, "moves": ["c1g5", "h7h6"]},
                {"score": {"type": "cp", "value": -20}, "moves": ["c4b5", "d7d6"]},
                {"score": {"type": "cp", "value": -40}, "moves": ["c4f7", "e8f7"]},
            ],
        },
        "eval_after": None,
        "positional_tag": "retain_pv3_check_within_100cp",
    }

    # 4. Royal fork check retained without PV (safe knight check forking king and rook)
    fen_4 = "r3k2r/pppb1ppp/8/n2N4/8/3B4/PPP2PPP/R3K2R w KQkq - 0 10"
    case_4 = {
        "task": "explain_position",
        "fen": fen_4,
        "move_uci": None,
        "eval_before": {
            "fen": fen_4,
            "pvs": [{"score": {"type": "cp", "value": 350}, "moves": ["e1g1", "e8g8"]}],
        },
        "eval_after": None,
        "positional_tag": "retain_royal_fork_check_no_pv",
    }

    # 5. PV2 check excluded when > 100 cp worse than PV1
    case_5 = {
        "task": "explain_position",
        "fen": fen_3,
        "move_uci": None,
        "eval_before": {
            "fen": fen_3,
            "pvs": [
                {"score": {"type": "cp", "value": 30}, "moves": ["c1g5", "h7h6"]},
                {"score": {"type": "cp", "value": -120}, "moves": ["c4f7", "e8f7"]},
            ],
        },
        "eval_after": None,
        "positional_tag": "suppress_pv2_check_over_100cp_drop",
    }

    # 6. Safe check that forces king-only replies stripping Black castling rights
    fen_6 = "4k2r/4pppp/7n/8/2B5/8/4PPPP/4K2R w k - 0 15"
    case_6 = {
        "task": "explain_position",
        "fen": fen_6,
        "move_uci": None,
        "eval_before": {
            "fen": fen_6,
            "pvs": [{"score": {"type": "cp", "value": 50}, "moves": ["e1f1", "e8f8"]}],
        },
        "eval_after": None,
        "positional_tag": "forced_king_strip_castling_check",
    }

    # 7. Recapture centralizes queen
    fen_7 = "r1bqkb1r/ppp2ppp/2np1n2/4p3/3PP3/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 0 5"
    case_7 = {
        "task": "explain_move",
        "fen": fen_7,
        "move_uci": "e5d4",
        "eval_before": {
            "fen": fen_7,
            "pvs": [
                {"score": {"type": "cp", "value": 35}, "moves": ["f8e7", "f1e2", "e8g8", "e1g1"]},
                {"score": {"type": "cp", "value": 60}, "moves": ["e5d4", "d1d4", "f8e7", "c1e3"]},
            ],
        },
        "eval_after": {
            "fen": _after_move(fen_7, "e5d4"),
            "pvs": [{"score": {"type": "cp", "value": 60}, "moves": ["d1d4", "f8e7", "c1e3"]}],
        },
        "positional_tag": "recapture_centralizes_queen",
    }

    # 8. Recapture increases queen activity from back rank (Bxf3 -> Qxf3)
    fen_8 = "r2qk2r/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 0 7"
    case_8 = {
        "task": "explain_move",
        "fen": fen_8,
        "move_uci": "g4f3",
        "eval_before": {
            "fen": fen_8,
            "pvs": [
                {"score": {"type": "cp", "value": 10}, "moves": ["h7h6", "h2h3", "g4h5", "c1e3"]},
                {"score": {"type": "cp", "value": 70}, "moves": ["g4f3", "d1f3", "c6d4", "f3d1"]},
            ],
        },
        "eval_after": {
            "fen": _after_move(fen_8, "g4f3"),
            "pvs": [{"score": {"type": "cp", "value": 70}, "moves": ["d1f3", "c6d4", "f3d1"]}],
        },
        "positional_tag": "recapture_increases_queen_activity",
    }

    # 9. Recapture centralizes rook
    fen_9_pos = "3r2k1/pp3ppp/8/n2N4/8/8/PP3PPP/3R2K1 b - - 0 22"
    case_9 = {
        "task": "explain_move",
        "fen": fen_9_pos,
        "move_uci": "d8d5",
        "eval_before": {
            "fen": fen_9_pos,
            "pvs": [
                {"score": {"type": "cp", "value": 0}, "moves": ["g8f8", "d5e3", "g7g6", "d1c1"]},
                {"score": {"type": "cp", "value": 100}, "moves": ["d8d5", "d1d5", "g8f8", "g1f1"]},
            ],
        },
        "eval_after": {
            "fen": _after_move(fen_9_pos, "d8d5"),
            "pvs": [{"score": {"type": "cp", "value": 100}, "moves": ["d1d5", "g8f8", "g1f1"]}],
        },
        "positional_tag": "recapture_centralizes_rook",
    }

    # 10. Refutation: central pawn claims space on d4 instead of passive pawn move
    fen_10 = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    case_10 = {
        "task": "explain_move",
        "fen": fen_10,
        "move_uci": "a2a3",
        "eval_before": {
            "fen": fen_10,
            "pvs": [{"score": {"type": "cp", "value": 60}, "moves": ["d2d4", "e5d4", "f3d4", "g8f6"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_10, "a2a3"),
            "pvs": [{"score": {"type": "cp", "value": -45}, "moves": ["g8f6", "b1c3", "f8c5"]}],
        },
        "positional_tag": "refutation_claims_central_space",
    }

    # 11. Refutation: pawn push c4 gains space and challenges center
    fen_11 = "rnbqkb1r/pp2pppp/2p2n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 4"
    case_11 = {
        "task": "explain_move",
        "fen": fen_11,
        "move_uci": "h2h3",
        "eval_before": {
            "fen": fen_11,
            "pvs": [{"score": {"type": "cp", "value": 45}, "moves": ["c2c4", "e7e6", "b1c3", "f8e7"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_11, "h2h3"),
            "pvs": [{"score": {"type": "cp", "value": -25}, "moves": ["c8f5", "c2c4", "e7e6"]}],
        },
        "positional_tag": "refutation_gains_space_challenges_center",
    }

    # 12. Refutation: pawn push keeps central space and tension instead of trading on c5
    fen_12 = "r1bqkb1r/pp1p1ppp/2n1pn2/2p5/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5"
    case_12 = {
        "task": "explain_move",
        "fen": fen_12,
        "move_uci": "d4c5",
        "eval_before": {
            "fen": fen_12,
            "pvs": [{"score": {"type": "cp", "value": 85}, "moves": ["d4d5", "e6d5", "e4d5", "c6d4"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_12, "d4c5"),
            "pvs": [{"score": {"type": "cp", "value": -10}, "moves": ["f8c5", "f1d3", "d7d6"]}],
        },
        "positional_tag": "refutation_central_pawn_attacks_knight",
    }

    # 13. Refutation: pawn push c4 keeps space and tension instead of trading on e5
    fen_13_legal = "rnbqkb1r/ppp2ppp/5n2/3pp3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4"
    case_13 = {
        "task": "explain_move",
        "fen": fen_13_legal,
        "move_uci": "d4e5",
        "eval_before": {
            "fen": fen_13_legal,
            "pvs": [{"score": {"type": "cp", "value": 75}, "moves": ["c2c4", "e5d4", "f3d4", "f8b4"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_13_legal, "d4e5"),
            "pvs": [{"score": {"type": "cp", "value": -15}, "moves": ["f6e4", "b1c3", "f8b4"]}],
        },
        "positional_tag": "refutation_keeps_space_instead_of_trade",
    }

    # 14. Refutation: castling to keep the king safe
    fen_14 = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 13"
    case_14 = {
        "task": "explain_move",
        "fen": fen_14,
        "move_uci": "h2h3",
        "eval_before": {
            "fen": fen_14,
            "pvs": [{"score": {"type": "cp", "value": 65}, "moves": ["e1g1", "d7d6", "c2c3", "e8g8"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_14, "h2h3"),
            "pvs": [{"score": {"type": "cp", "value": -35}, "moves": ["e8g8", "b1c3", "d7d6"]}],
        },
        "positional_tag": "refutation_castles_king_safety",
    }

    # 15. Refutation: move attacked piece to safety
    fen_15 = "r2qk2r/ppp2ppp/2n5/3p4/3Pn1b1/2N2N1P/PPPB1PP1/R2Q1RK1 b kq - 0 13"
    case_15 = {
        "task": "explain_move",
        "fen": fen_15,
        "move_uci": "a7a6",
        "eval_before": {
            "fen": fen_15,
            "pvs": [{"score": {"type": "cp", "value": -30}, "moves": ["g4h5", "f1e1", "e8g8", "c3e4"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_15, "a7a6"),
            "pvs": [{"score": {"type": "cp", "value": 85}, "moves": ["h3g4", "h7h5", "g4h5"]}],
        },
        "positional_tag": "refutation_saves_attacked_bishop",
    }

    # 16. Refutation: place rook on open d-file
    fen_16 = "r2r2k1/pp3ppp/2p5/8/8/2P5/PP3PPP/R4RK1 w - - 0 20"
    case_16 = {
        "task": "explain_move",
        "fen": fen_16,
        "move_uci": "a2a3",
        "eval_before": {
            "fen": fen_16,
            "pvs": [{"score": {"type": "cp", "value": 45}, "moves": ["a1d1", "d8d1", "f1d1", "a8d8"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_16, "a2a3"),
            "pvs": [{"score": {"type": "cp", "value": -60}, "moves": ["d8d2", "f1d1", "d2f2"]}],
        },
        "positional_tag": "refutation_rook_open_file",
    }

    # 17. Refutation: centralizes knight (balanced Sicilian structure)
    fen_17_quiet = _fen_after(
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "f2f3", "e7e6"]
    )
    case_17 = {
        "task": "explain_move",
        "fen": fen_17_quiet,
        "move_uci": "a2a3",
        "eval_before": {
            "fen": fen_17_quiet,
            "pvs": [{"score": {"type": "cp", "value": 60}, "moves": ["c3d5", "e6d5", "e4d5", "f8e7"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_17_quiet, "a2a3"),
            "pvs": [{"score": {"type": "cp", "value": -45}, "moves": ["f8e7", "c1e3", "e8g8"]}],
        },
        "positional_tag": "refutation_centralizes_knight",
    }

    # 18. Refutation: increases queen activity (balanced middlegame)
    fen_18_clean = _fen_after(
        [
            "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6",
            "f2f3", "e8g8", "c1e3", "c7c5", "g1e2", "b8c6", "d1d2", "c5d4",
            "e2d4", "c6d4", "e3d4", "c8e6", "e1c1", "d8a5", "c1b1", "f8c8",
            "d2e1", "a7a6",
        ]
    )
    case_18 = {
        "task": "explain_move",
        "fen": fen_18_clean,
        "move_uci": "a2a3",
        "eval_before": {
            "fen": fen_18_clean,
            "pvs": [{"score": {"type": "cp", "value": 60}, "moves": ["e1h4", "a5c7", "d4f6", "g7f6"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_18_clean, "a2a3"),
            "pvs": [{"score": {"type": "cp", "value": -40}, "moves": ["b7b5", "c3d5", "a5d8"]}],
        },
        "positional_tag": "refutation_increases_queen_activity",
    }

    # 19. Line comparison: best line keeps stronger central pawn control
    fen_19 = "rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5"
    case_19 = {
        "task": "explain_move",
        "fen": fen_19,
        "move_uci": "e4d5",
        "eval_before": {
            "fen": fen_19,
            "pvs": [{"score": {"type": "cp", "value": 65}, "moves": ["e4e5", "f6d7", "c3e2", "c7c5"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_19, "e4d5"),
            "pvs": [{"score": {"type": "cp", "value": -15}, "moves": ["e6d5", "f1d3", "f8e7"]}],
        },
        "positional_tag": "line_comparison_central_pawns",
    }

    # 20. Line comparison: best line gives better king safety (castling vs staying in center)
    fen_20 = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    case_20 = {
        "task": "explain_move",
        "fen": fen_20,
        "move_uci": "h2h3",
        "eval_before": {
            "fen": fen_20,
            "pvs": [{"score": {"type": "cp", "value": 45}, "moves": ["e1g1", "d7d6", "d2d3", "e8g8"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_20, "h2h3"),
            "pvs": [{"score": {"type": "cp", "value": -55}, "moves": ["d7d6", "b1c3", "h7h6"]}],
        },
        "positional_tag": "line_comparison_king_safety",
    }

    # 21. Line comparison: avoids losing a pawn or material
    fen_21 = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    case_21 = {
        "task": "explain_move",
        "fen": fen_21,
        "move_uci": "f1a6",
        "eval_before": {
            "fen": fen_21,
            "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["f1b5", "a7a6", "b5a4", "g8f6"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_21, "f1a6"),
            "pvs": [{"score": {"type": "cp", "value": -300}, "moves": ["b7a6", "e1g1", "g8f6"]}],
        },
        "positional_tag": "line_comparison_material_loss",
    }

    # 22. Quiet move positional fact: castling to safeguard king
    fen_22 = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    case_22 = {
        "task": "explain_move",
        "fen": fen_22,
        "move_uci": "e1g1",
        "eval_before": {
            "fen": fen_22,
            "pvs": [{"score": {"type": "cp", "value": 35}, "moves": ["e1g1", "d7d6", "d2d3", "e8g8"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_22, "e1g1"),
            "pvs": [{"score": {"type": "cp", "value": 35}, "moves": ["d7d6", "d2d3", "e8g8"]}],
        },
        "positional_tag": "quiet_move_castles",
    }

    # 23. Quiet move positional fact: pawn claims central space on d4
    fen_23 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 2"
    case_23 = {
        "task": "explain_move",
        "fen": fen_23,
        "move_uci": "d2d4",
        "eval_before": {
            "fen": fen_23,
            "pvs": [{"score": {"type": "cp", "value": 25}, "moves": ["d2d4", "e5d4", "f3d4", "b8c6"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_23, "d2d4"),
            "pvs": [{"score": {"type": "cp", "value": 25}, "moves": ["e5d4", "f3d4", "b8c6"]}],
        },
        "positional_tag": "quiet_move_claims_space",
    }

    # 24. Quiet move positional fact: pawn gains space and challenges center on c4
    fen_24 = "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2"
    case_24 = {
        "task": "explain_move",
        "fen": fen_24,
        "move_uci": "c2c4",
        "eval_before": {
            "fen": fen_24,
            "pvs": [{"score": {"type": "cp", "value": 35}, "moves": ["c2c4", "e7e6", "g1f3", "d7d5"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_24, "c2c4"),
            "pvs": [{"score": {"type": "cp", "value": 35}, "moves": ["e7e6", "g1f3", "d7d5"]}],
        },
        "positional_tag": "quiet_move_c4_challenges_center",
    }

    # 25. Quiet move positional fact: minor develops toward center
    fen_25 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    case_25 = {
        "task": "explain_move",
        "fen": fen_25,
        "move_uci": "g1f3",
        "eval_before": {
            "fen": fen_25,
            "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["g1f3", "b8c6", "f1b5", "a7a6"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_25, "g1f3"),
            "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["b8c6", "f1b5", "a7a6"]}],
        },
        "positional_tag": "quiet_move_develops_knight",
    }

    # 26. Quiet move positional fact: rook placed on half-open file
    fen_26 = "r2q1rk1/pp2ppbp/2np1np1/8/4P3/2N1BP2/PPPQB1PP/R3K2R w KQ - 0 11"
    case_26 = {
        "task": "explain_move",
        "fen": fen_26,
        "move_uci": "a1d1",
        "eval_before": {
            "fen": fen_26,
            "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["a1d1", "d8a5", "e1g1", "a8c8"]}],
        },
        "eval_after": {
            "fen": _after_move(fen_26, "a1d1"),
            "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["d8a5", "e1g1", "a8c8"]}],
        },
        "positional_tag": "quiet_move_rook_half_open_file",
    }

    # 27. Explain position: central pawn tension between d4 and e5
    fen_27 = "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3"
    case_27 = {
        "task": "explain_position",
        "fen": fen_27,
        "move_uci": None,
        "eval_before": {
            "fen": fen_27,
            "pvs": [
                {"score": {"type": "cp", "value": 45}, "moves": ["e5d4", "f3d4"]},
                {"score": {"type": "cp", "value": 85}, "moves": ["g8f6", "d4e5"]},
            ],
        },
        "eval_after": None,
        "positional_tag": "position_central_pawn_tension",
    }

    # 28. Explain position: White has more central space
    fen_28 = "rnbqkbnr/pp1p1ppp/2p1p3/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5"
    case_28 = {
        "task": "explain_position",
        "fen": fen_28,
        "move_uci": None,
        "eval_before": {
            "fen": fen_28,
            "pvs": [
                {"score": {"type": "cp", "value": 75}, "moves": ["g1f3", "g8f6"]},
                {"score": {"type": "cp", "value": 65}, "moves": ["f1d3", "d7d5"]},
            ],
        },
        "eval_after": None,
        "positional_tag": "position_more_central_space",
    }

    # 29. Explain position: open central d-file
    fen_29 = "r1b1k2r/ppp1bppp/2n2n2/q7/2B5/2N2N2/PPP2PPP/R1BQK2R w KQkq - 0 7"
    case_29 = {
        "task": "explain_position",
        "fen": fen_29,
        "move_uci": None,
        "eval_before": {
            "fen": fen_29,
            "pvs": [
                {"score": {"type": "cp", "value": 65}, "moves": ["e1g1", "e8g8"]},
                {"score": {"type": "cp", "value": 55}, "moves": ["d1e2", "c8g4"]},
            ],
        },
        "eval_after": None,
        "positional_tag": "position_open_d_file",
    }

    cases_spec = [
        case_0, case_1, case_2, case_3, case_4,
        case_5, case_6, case_7, case_8, case_9,
        case_10, case_11, case_12, case_13, case_14,
        case_15, case_16, case_17, case_18, case_19,
        case_20, case_21, case_22, case_23, case_24,
        case_25, case_26, case_27, case_28, case_29,
    ]
    assert len(cases_spec) == 30, f"Expected 30 cases, got {len(cases_spec)}"

    golden_items: list[dict[str, Any]] = []
    levels = ["beginner", "club"]
    details = ["short", "long"]

    for idx, spec in enumerate(cases_spec):
        level = "club" if idx == 0 else levels[idx % 2]
        detail = "short" if idx == 0 else details[(idx // 2) % 2]
        task = spec["task"]
        fen = spec["fen"]
        move_uci = spec["move_uci"]
        eval_before = spec["eval_before"]
        eval_after = spec["eval_after"]

        # Verify board validity, move_uci, PV move legality, and eval_after FEN before formatting
        b_chk = chess.Board(fen)
        assert b_chk.is_valid(), f"Case {idx}: invalid board FEN {fen}"
        for pv in eval_before.get("pvs", []):
            sb = b_chk.copy(stack=False)
            for uci in pv.get("moves", []):
                m = chess.Move.from_uci(uci)
                assert m in sb.legal_moves, f"Case {idx}: illegal eval_before PV move {uci}"
                sb.push(m)
        if move_uci is not None:
            m_played = chess.Move.from_uci(move_uci)
            assert m_played in b_chk.legal_moves, f"Case {idx}: illegal move_uci {move_uci}"
            ba_chk = b_chk.copy(stack=False)
            ba_chk.push(m_played)
            if eval_after is not None:
                if eval_after.get("fen"):
                    assert eval_after["fen"] == ba_chk.fen(), f"Case {idx}: eval_after.fen mismatch"
                for pv in eval_after.get("pvs", []):
                    sa = ba_chk.copy(stack=False)
                    for uci in pv.get("moves", []):
                        m = chess.Move.from_uci(uci)
                        assert m in sa.legal_moves, f"Case {idx}: illegal eval_after PV move {uci}"
                        sa.push(m)

        user_msg = build_user_message(
            task=task,
            level=level,
            detail=detail,
            fen=fen,
            eval_before=eval_before,
            eval_after=eval_after,
            move_uci=move_uci,
        )
        chat_prompt = format_qwen_chat_prompt(user_msg)

        golden_items.append({
            "id": idx,
            "task": task,
            "level": level,
            "detail": detail,
            "fen": fen,
            "move_uci": move_uci,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "expected_user_message": user_msg,
            "expected_chat_prompt": chat_prompt,
            "positional_tag": spec["positional_tag"],
        })

    return golden_items


def main() -> None:
    """Write golden_positional_30.json to disk."""
    cases = build_positional_cases()
    out_path = Path(__file__).resolve().parent / "golden_positional_30.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(cases)} positional golden cases to {out_path}.")


if __name__ == "__main__":
    main()
