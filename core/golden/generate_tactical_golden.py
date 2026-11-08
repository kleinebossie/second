"""Generate 30 tactical golden test cases for the Second chess assistant.

This module builds golden test cases for tactical motifs:
- Fork / Double attack
- Pin (absolute and relative)
- Skewer
- Discovered attack / check
- Forced checkmate / mate threat
- Refutation reason on bad moves
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


def build_tactical_cases() -> list[dict[str, Any]]:
    """Build thirty targeted tactical test cases."""
    cases_spec: list[dict[str, Any]] = [
        # --- FORKS (0 to 4) ---
        # Case 0: Knight fork on King and Rook
        {
            "fen": "r3k2r/ppp2ppp/8/3N4/8/8/PPP2PPP/R1B1K2R w KQkq - 0 1",
            "move_uci": "d5c7",
            "eval_before": {
                "fen": "r3k2r/ppp2ppp/8/3N4/8/8/PPP2PPP/R1B1K2R w KQkq - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 350}, "moves": ["d5c7", "e8d8", "c7a8"]}],
            },
            "eval_after": {
                "fen": "r3k2r/ppN2ppp/8/8/8/8/PPP2PPP/R1B1K2R b KQkq - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 350}, "moves": ["e8d8", "c7a8"]}],
            },
            "motif_tag": "fork_knight_king_rook",
        },
        # Case 1: Pawn fork on two minor pieces
        {
            "fen": "r1bqk2r/ppp2ppp/3b1n2/8/4P3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1",
            "move_uci": "e4e5",
            "eval_before": {
                "fen": "r1bqk2r/ppp2ppp/3b1n2/8/4P3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 250}, "moves": ["e4e5", "d6e5", "d1d8", "e8d8"]}],
            },
            "eval_after": {
                "fen": "r1bqk2r/ppp2ppp/3b1n2/4P3/8/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 250}, "moves": ["d6e5", "d1d8", "e8d8"]}],
            },
            "motif_tag": "fork_pawn_minors",
        },
        # Case 2: Queen fork on King and Bishop
        {
            "fen": "4k3/8/8/8/4b3/8/PP2PPPP/RNBQ1RK1 w - - 0 1",
            "move_uci": "d1a4",
            "eval_before": {
                "fen": "4k3/8/8/8/4b3/8/PP2PPPP/RNBQ1RK1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["d1a4", "e8f7", "a4e4"]}],
            },
            "eval_after": {
                "fen": "4k3/8/8/8/Q3b3/8/PP2PPPP/RNB2RK1 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["e8f7", "a4e4"]}],
            },
            "motif_tag": "fork_queen_king_bishop",
        },
        # Case 3: Reply fork
        {
            "fen": "r3k2r/ppp2ppp/8/8/3n4/8/PPP1NPPP/R3K2R w KQkq - 0 1",
            "move_uci": "h2h3",
            "eval_before": {
                "fen": "r3k2r/ppp2ppp/8/8/3n4/8/PPP1NPPP/R3K2R w KQkq - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["e2d4"]}],
            },
            "eval_after": {
                "fen": "r3k2r/ppp2ppp/8/8/3n4/7P/PPP1NPP1/R3K2R b KQkq - 0 1",
                "pvs": [{"score": {"type": "cp", "value": -350}, "moves": ["d4c2", "e1d1", "c2a1"]}],
            },
            "motif_tag": "fork_reply_knight",
        },
        # Case 4: Rook fork on two undefended minor pieces
        {
            "fen": "5k2/b6n/8/8/8/8/8/4R2K w - - 0 1",
            "move_uci": "e1e7",
            "eval_before": {
                "fen": "5k2/b6n/8/8/8/8/8/4R2K w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 200}, "moves": ["e1e7", "f8g8", "e7a7"]}],
            },
            "eval_after": {
                "fen": "5k2/b3R2n/8/8/8/8/8/7K b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 200}, "moves": ["f8g8", "e7a7"]}],
            },
            "motif_tag": "fork_rook_minors",
        },

        # --- PINS (5 to 9) ---
        # Case 5: Absolute pin by bishop to King
        {
            "fen": "r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4",
            "move_uci": "f1b5",
            "eval_before": {
                "fen": "r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4",
                "pvs": [{"score": {"type": "cp", "value": 40}, "moves": ["f1b5", "a7a6", "b5a4"]}],
            },
            "eval_after": {
                "fen": "r1bqkbnr/ppp2ppp/2np4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4",
                "pvs": [{"score": {"type": "cp", "value": 40}, "moves": ["a7a6", "b5a4"]}],
            },
            "motif_tag": "pin_absolute_bishop_king",
        },
        # Case 6: Relative pin by bishop to Queen
        {
            "fen": "r1b1k2r/ppp1qppp/2np1n2/4p3/4P3/3P1N2/PPP1BPPP/RNBQK2R w KQkq - 0 6",
            "move_uci": "c1g5",
            "eval_before": {
                "fen": "r1b1k2r/ppp1qppp/2np1n2/4p3/4P3/3P1N2/PPP1BPPP/RNBQK2R w KQkq - 0 6",
                "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["c1g5", "h7h6"]}],
            },
            "eval_after": {
                "fen": "r1b1k2r/ppp1qppp/2np1n2/4p1B1/4P3/3P1N2/PPP1BPPP/RN1QK2R b KQkq - 1 6",
                "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["h7h6", "g5f6"]}],
            },
            "motif_tag": "pin_relative_bishop_queen",
        },
        # Case 7: Absolute pin by rook to King
        {
            "fen": "4k3/8/8/4n3/8/8/8/R5K1 w - - 0 1",
            "move_uci": "a1e1",
            "eval_before": {
                "fen": "4k3/8/8/4n3/8/8/8/R5K1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["a1e1", "e8d7", "e1e5"]}],
            },
            "eval_after": {
                "fen": "4k3/8/8/4n3/8/8/8/4R1K1 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["e8d7", "e1e5"]}],
            },
            "motif_tag": "pin_absolute_rook_king",
        },
        # Case 8: Relative pin by rook to Queen
        {
            "fen": "3qk3/8/8/3b4/8/8/8/R3K3 w Q - 0 1",
            "move_uci": "a1d1",
            "eval_before": {
                "fen": "3qk3/8/8/3b4/8/8/8/R3K3 w Q - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 250}, "moves": ["a1d1", "d8e7"]}],
            },
            "eval_after": {
                "fen": "3qk3/8/8/3b4/8/8/8/3RK3 b Q - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 250}, "moves": ["d8e7", "e1f2"]}],
            },
            "motif_tag": "pin_relative_rook_queen",
        },
        # Case 9: Reply pin
        {
            "fen": "r1bqk2r/ppp2ppp/2np1n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 5",
            "move_uci": "h2h3",
            "eval_before": {
                "fen": "r1bqk2r/ppp2ppp/2np1n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 5",
                "pvs": [{"score": {"type": "cp", "value": 20}, "moves": ["f1e2"]}],
            },
            "eval_after": {
                "fen": "r1bqk2r/ppp2ppp/2np1n2/4p3/4P3/3P1N1P/PPP2PP1/RNBQKB1R b KQkq - 0 5",
                "pvs": [{"score": {"type": "cp", "value": -10}, "moves": ["c8g4", "f1e2", "g4f3"]}],
            },
            "motif_tag": "pin_reply_bishop",
        },

        # --- SKEWERS (10 to 14) ---
        # Case 10: Rook skewers King and Queen
        {
            "fen": "4q3/8/4k3/8/5R2/8/8/6K1 w - - 0 1",
            "move_uci": "f4e4",
            "eval_before": {
                "fen": "4q3/8/4k3/8/5R2/8/8/6K1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 800}, "moves": ["f4e4", "e6d6", "e4e8"]}],
            },
            "eval_after": {
                "fen": "4q3/8/4k3/8/4R3/8/8/6K1 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 800}, "moves": ["e6d6", "e4e8"]}],
            },
            "motif_tag": "skewer_rook_king_queen",
        },
        # Case 11: Bishop skewers King and Rook
        {
            "fen": "8/7r/8/5k2/8/5B2/8/6K1 w - - 0 1",
            "move_uci": "f3e4",
            "eval_before": {
                "fen": "8/7r/8/5k2/8/5B2/8/6K1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["f3e4", "f5e4", "g1g2"]}],
            },
            "eval_after": {
                "fen": "8/7r/8/5k2/4B3/8/8/6K1 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["f5e4", "g1g2"]}],
            },
            "motif_tag": "skewer_bishop_king_rook",
        },
        # Case 12: Queen skewers King and Rook
        {
            "fen": "r7/8/8/3k4/8/8/8/1Q4K1 w - - 0 1",
            "move_uci": "b1e4",
            "eval_before": {
                "fen": "r7/8/8/3k4/8/8/8/1Q4K1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 500}, "moves": ["b1e4", "d5e4", "g1f2"]}],
            },
            "eval_after": {
                "fen": "r7/8/8/3k4/4Q3/8/8/6K1 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 500}, "moves": ["d5e4", "g1f2"]}],
            },
            "motif_tag": "skewer_queen_king_rook",
        },
        # Case 13: Rook skewers Queen and Rook
        {
            "fen": "4r1k1/8/4q3/8/5R2/8/8/6K1 w - - 0 1",
            "move_uci": "f4e4",
            "eval_before": {
                "fen": "4r1k1/8/4q3/8/5R2/8/8/6K1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 400}, "moves": ["f4e4", "e6f7", "e4e8"]}],
            },
            "eval_after": {
                "fen": "4r1k1/8/4q3/8/4R3/8/8/6K1 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 400}, "moves": ["e6f7", "e4e8"]}],
            },
            "motif_tag": "skewer_rook_queen_rook",
        },
        # Case 14: Reply skewer
        {
            "fen": "4Q3/8/8/8/4K3/P7/8/3r2k1 w - - 0 1",
            "move_uci": "a3a4",
            "eval_before": {
                "fen": "4Q3/8/8/8/4K3/P7/8/3r2k1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["e8h5"]}],
            },
            "eval_after": {
                "fen": "4Q3/8/8/8/P3K3/8/8/3r2k1 b - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": -800}, "moves": ["d1e1", "e4f3", "e1e8"]}],
            },
            "motif_tag": "skewer_reply_rook",
        },

        # --- DISCOVERED ATTACKS / CHECKS (15 to 19) ---
        # Case 15: Discovered check against King
        {
            "fen": "4k3/8/2N5/8/B7/8/8/4K3 w - - 0 1",
            "move_uci": "c6e5",
            "eval_before": {
                "fen": "4k3/8/2N5/8/B7/8/8/4K3 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["c6e5", "e8e7", "e5g6"]}],
            },
            "eval_after": {
                "fen": "4k3/8/8/4N3/B7/8/8/4K3 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["e8e7", "e5g6"]}],
            },
            "motif_tag": "discovered_check_bishop_king",
        },
        # Case 16: Discovered attack against Queen
        {
            "fen": "4q1k1/8/8/4N3/8/8/8/4R1K1 w - - 0 1",
            "move_uci": "e5f3",
            "eval_before": {
                "fen": "4q1k1/8/8/4N3/8/8/8/4R1K1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 400}, "moves": ["e5f3", "e8e1", "f3e1"]}],
            },
            "eval_after": {
                "fen": "4q1k1/8/8/8/8/5N2/8/4R1K1 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 400}, "moves": ["e8e1", "f3e1"]}],
            },
            "motif_tag": "discovered_attack_rook_queen",
        },
        # Case 17: Discovered attack against Rook
        {
            "fen": "k7/7r/8/8/8/3P4/8/KB6 w - - 0 1",
            "move_uci": "d3d4",
            "eval_before": {
                "fen": "k7/7r/8/8/8/3P4/8/KB6 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["d3d4", "h7h8"]}],
            },
            "eval_after": {
                "fen": "k7/7r/8/8/3P4/8/8/KB6 b - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 300}, "moves": ["h7h8"]}],
            },
            "motif_tag": "discovered_attack_bishop_rook",
        },
        # Case 18: Discovered attack against undefended minor piece
        {
            "fen": "6k1/3n4/8/8/8/3B4/8/3Q2K1 w - - 0 1",
            "move_uci": "d3e4",
            "eval_before": {
                "fen": "6k1/3n4/8/8/8/3B4/8/3Q2K1 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 200}, "moves": ["d3e4", "d7f6"]}],
            },
            "eval_after": {
                "fen": "6k1/3n4/8/8/4B3/8/8/3Q2K1 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 200}, "moves": ["d7f6"]}],
            },
            "motif_tag": "discovered_attack_queen_minor",
        },
        # Case 19: Discovered attack against King from Knight move
        {
            "fen": "7k/8/8/8/8/2N5/8/B3K3 w - - 0 1",
            "move_uci": "c3e4",
            "eval_before": {
                "fen": "7k/8/8/8/8/2N5/8/B3K3 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 200}, "moves": ["c3e4", "h8g8"]}],
            },
            "eval_after": {
                "fen": "7k/8/8/8/4N3/8/8/B3K3 b - - 1 1",
                "pvs": [{"score": {"type": "cp", "value": 200}, "moves": ["h8g8"]}],
            },
            "motif_tag": "discovered_check_bishop_king_diagonal",
        },

        # --- FORCED MATE & MATE THREATS (20 to 24) ---
        # Case 20: Mover has forced mate in 1
        {
            "fen": "6k1/5ppp/8/8/8/8/8/4Q1K1 w - - 0 1",
            "move_uci": "e1e7",
            "eval_before": {
                "fen": "6k1/5ppp/8/8/8/8/8/4Q1K1 w - - 0 1",
                "pvs": [{"score": {"type": "mate", "value": 1}, "moves": ["e1e8"]}],
            },
            "eval_after": {
                "fen": "6k1/4Qppp/8/8/8/8/8/6K1 b - - 1 1",
                "pvs": [{"score": {"type": "mate", "value": 1}, "moves": ["h7h6", "e7e8"]}],
            },
            "motif_tag": "mate_mover_in_1",
        },
        # Case 21: Mover has forced mate in 2
        {
            "fen": "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
            "move_uci": "e1e7",
            "eval_before": {
                "fen": "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
                "pvs": [{"score": {"type": "mate", "value": 2}, "moves": ["e1e8"]}],
            },
            "eval_after": {
                "fen": "6k1/4Rppp/8/8/8/8/8/6K1 b - - 1 1",
                "pvs": [{"score": {"type": "mate", "value": 2}, "moves": ["g8f8", "e7a7", "f8e8", "a7a8"]}],
            },
            "motif_tag": "mate_mover_in_2",
        },
        # Case 22: Opponent threatens mate in 1
        {
            "fen": "4r1k1/8/8/8/8/8/4q3/6K1 w - - 0 1",
            "move_uci": "g1h1",
            "eval_before": {
                "fen": "4r1k1/8/8/8/8/8/4q3/6K1 w - - 0 1",
                "pvs": [{"score": {"type": "mate", "value": -1}, "moves": ["g1h1"]}],
            },
            "eval_after": {
                "fen": "4r1k1/8/8/8/8/8/4q3/7K b - - 1 1",
                "pvs": [{"score": {"type": "mate", "value": -1}, "moves": ["e2e1"]}],
            },
            "motif_tag": "mate_opponent_threat_in_1",
        },
        # Case 23: Opponent threatens mate in 2
        {
            "fen": "6k1/8/8/8/8/8/8/4K3 w - - 0 1",
            "move_uci": "e1d1",
            "eval_before": {
                "fen": "6k1/8/8/8/8/8/8/4K3 w - - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["e1e2"]}],
            },
            "eval_after": {
                "fen": "6k1/8/8/8/8/8/8/3K4 b - - 1 1",
                "pvs": [{"score": {"type": "mate", "value": -2}, "moves": ["g8f7"]}],
            },
            "motif_tag": "mate_opponent_threat_in_2",
        },
        # Case 24: Black mover has forced mate in 1
        {
            "fen": "4R1K1/8/8/8/8/8/4Q3/6k1 b - - 0 1",
            "move_uci": "g1h1",
            "eval_before": {
                "fen": "4R1K1/8/8/8/8/8/4Q3/6k1 b - - 0 1",
                "pvs": [{"score": {"type": "mate", "value": -1}, "moves": ["g1h1"]}],
            },
            "eval_after": {
                "fen": "4R1K1/8/8/8/8/8/4Q3/7k w - - 1 2",
                "pvs": [{"score": {"type": "mate", "value": -1}, "moves": ["e8e7"]}],
            },
            "motif_tag": "mate_black_mover_in_1",
        },

        # --- REFUTATION REASON ON BAD MOVES (25 to 29) ---
        # Case 25: Best move captures a pawn
        {
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "move_uci": "h2h3",
            "eval_before": {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "pvs": [{"score": {"type": "cp", "value": 40}, "moves": ["f3e5"]}],
            },
            "eval_after": {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N1P/PPPP1PP1/RNBQKB1R b KQkq - 0 3",
                "pvs": [{"score": {"type": "cp", "value": -30}, "moves": ["g8f6"]}],
            },
            "motif_tag": "refutation_captures_pawn",
        },
        # Case 26: Best move captures a queen
        {
            "fen": "r1b1kbnr/pppp1ppp/8/4q3/5B2/8/PPP2PPP/RN1Q1RK1 w kq - 0 1",
            "move_uci": "a2a3",
            "eval_before": {
                "fen": "r1b1kbnr/pppp1ppp/8/4q3/5B2/8/PPP2PPP/RN1Q1RK1 w kq - 0 1",
                "pvs": [{"score": {"type": "cp", "value": 600}, "moves": ["f4e5", "d7d6"]}],
            },
            "eval_after": {
                "fen": "r1b1kbnr/pppp1ppp/8/4q3/5B2/P7/1PP2PPP/RN1Q1RK1 b kq - 0 1",
                "pvs": [{"score": {"type": "cp", "value": -200}, "moves": ["e5f4"]}],
            },
            "motif_tag": "refutation_captures_queen",
        },
        # Case 27: Best move defends a hanging knight
        {
            "fen": "r1bqkb1r/pppp1ppp/2n5/8/3Pp3/5NP1/PPP2P1P/RNB1KB1R w KQkq - 0 4",
            "move_uci": "h2h3",
            "eval_before": {
                "fen": "r1bqkb1r/pppp1ppp/2n5/8/3Pp3/5NP1/PPP2P1P/RNB1KB1R w KQkq - 0 4",
                "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["b1d2", "e4f3"]}],
            },
            "eval_after": {
                "fen": "r1bqkb1r/pppp1ppp/2n5/8/3Pp3/5NP1/PPP2P1P/RNB1KB1R b KQkq - 0 4",
                "pvs": [{"score": {"type": "cp", "value": -250}, "moves": ["e4f3"]}],
            },
            "motif_tag": "refutation_defends_knight",
        },
        # Case 28: Best move defends a hanging bishop
        {
            "fen": "r1b1k1nr/pppp1ppp/2n5/4p3/1qB1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 5",
            "move_uci": "h2h3",
            "eval_before": {
                "fen": "r1b1k1nr/pppp1ppp/2n5/4p3/1qB1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 5",
                "pvs": [{"score": {"type": "cp", "value": 0}, "moves": ["b2b3", "b4e7"]}],
            },
            "eval_after": {
                "fen": "r1b1k1nr/pppp1ppp/2n5/4p3/1qB1P3/7P/PPPP1PP1/RNBQK1NR b KQkq - 0 5",
                "pvs": [{"score": {"type": "cp", "value": -300}, "moves": ["b4c4"]}],
            },
            "motif_tag": "refutation_defends_bishop",
        },
        # Case 29: Best move develops and prepares castling
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "move_uci": "a2a4",
            "eval_before": {
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["g1f3", "b8c6"]}],
            },
            "eval_after": {
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/P3P3/8/1PPP1PPP/RNBQKBNR b KQkq - 0 2",
                "pvs": [{"score": {"type": "cp", "value": -50}, "moves": ["g8f6", "b1c3"]}],
            },
            "motif_tag": "refutation_develops_castling",
        },
    ]

    golden_items: list[dict[str, Any]] = []
    levels = ["beginner", "club"]
    details = ["short", "long"]

    for idx, spec in enumerate(cases_spec):
        level = levels[idx % 2]
        detail = details[(idx // 2) % 2]
        task = "explain_move"
        fen = spec["fen"]
        move_uci = spec["move_uci"]
        eval_before = spec["eval_before"]
        eval_after = spec["eval_after"]

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

        item = {
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
            "motif_tag": spec["motif_tag"],
        }
        golden_items.append(item)

    return golden_items


def main() -> None:
    cases = build_tactical_cases()
    out_path = Path(__file__).resolve().parent / "golden_tactical_30.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(cases)} tactical golden cases at {out_path}.")


if __name__ == "__main__":
    main()
