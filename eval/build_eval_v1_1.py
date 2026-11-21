"""Build Phase 11 evaluation dev and test sets (devset-1.1 and testset-1.1).

This module refreshes devset-1.0 and testset-1.0 using v1.1 fact extraction.
It adds curated subtle positional regression cases and optional exported ratings.
It verifies zero contamination across training splits, devset-1.1, and testset-1.1.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chess
import chess.engine

from core.python.facts import _is_king_takes_friendly_rook
from core.python.prompt import rebuild_v1_1_record
from data.pipeline.rl_prompts import load_excluded_keys
from data.pipeline.splits import hash_game_id, normalize_fen
from eval.build_testset import extract_phase
from eval.concept_tags import classify_concept_tier

LOGGER = logging.getLogger(__name__)

DEFAULT_DEVSET_1_0 = Path("eval/devset-1.0.jsonl")
DEFAULT_TESTSET_1_0 = Path("eval/testset-1.0.jsonl")
DEFAULT_DEVSET_1_1 = Path("eval/devset-1.1.jsonl")
DEFAULT_TESTSET_1_1 = Path("eval/testset-1.1.jsonl")
DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_STOCKFISH_DEPTH = 12

ALL_TRAINING_SPLITS: tuple[str, ...] = (
    "sft_train",
    "sft_val",
    "rl_prompts",
    "rl_prompts_v2",
    "sft_v2_train",
    "teacher_prompts",
    "teacher_verified_non_thinking",
    "teacher_verified_thinking",
)


def find_stockfish_binary() -> str:
    """Locate the local Stockfish executable binary."""
    local_bin = (
        ROOT_DIR
        / "spikes"
        / "colab-stockfish"
        / "stockfish_bin"
        / "stockfish"
        / "stockfish-linux-x86-64-universal"
    )
    if local_bin.exists() and os.access(local_bin, os.X_OK):
        return str(local_bin)

    path_bin = shutil.which("stockfish")
    if path_bin:
        return path_bin

    raise FileNotFoundError("Stockfish binary not found.")


def _play_uci(moves: list[str]) -> str:
    """Play UCI moves from the starting position and return the final FEN."""
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board.fen()


def _analyze_board_with_engine(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    multipv: int = 3,
) -> dict[str, Any]:
    """Analyze a chess position and return White-perspective PVs."""
    if board.is_game_over():
        if board.is_checkmate():
            val = -1 if board.turn == chess.WHITE else 1
            return {
                "fen": board.fen(),
                "pvs": [{"score": {"type": "mate", "value": val}, "moves": []}],
            }
        return {
            "fen": board.fen(),
            "pvs": [{"score": {"type": "cp", "value": 0}, "moves": []}],
        }

    legal_count = board.legal_moves.count()
    actual_multipv = max(1, min(multipv, legal_count))
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=actual_multipv)
    if not isinstance(info, list):
        info = [info]

    pvs: list[dict[str, Any]] = []
    for entry in info:
        score = entry["score"].white()
        if score.is_mate():
            score_dict: dict[str, Any] = {"type": "mate", "value": int(score.mate() or 0)}
        else:
            score_dict = {"type": "cp", "value": int(score.score(mate_score=10000) or 0)}
        pv_moves = [m.uci() for m in entry.get("pv", [])[:8]]
        pvs.append({"score": score_dict, "moves": pv_moves})

    return {"fen": board.fen(), "pvs": pvs}


def get_curated_dev_specs() -> list[dict[str, Any]]:
    """Return ten curated positional and check-filter regression specs for devset-1.1."""
    return [
        {
            "id": "dev_151",
            "pool_id": "v11_dev_reg_01",
            "game_id": "v11_curated_dev_01",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3", "e7e5", "d4b3", "c8e6"]),
            "move_uci": "f2f3",
            "tag": "quiet_sicilian_f3_structure",
        },
        {
            "id": "dev_152",
            "pool_id": "v11_dev_reg_02",
            "game_id": "v11_curated_dev_02",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "d2d3", "b7b5", "a4b3", "e8g8", "c2c3", "d7d6"]),
            "move_uci": "c3c4",
            "tag": "ruy_closed_premature_c4_inaccuracy",
        },
        {
            "id": "dev_153",
            "pool_id": "v11_dev_reg_03",
            "game_id": "v11_curated_dev_03",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "e7e5", "g1f3", "e5d4", "f1c4", "b8c6"]),
            "move_uci": "h2h3",
            "tag": "qga_passive_h3_misses_castling",
        },
        {
            "id": "dev_154",
            "pool_id": "v11_dev_reg_04",
            "game_id": "v11_curated_dev_04",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7", "f1g2", "f8e7"]),
            "move_uci": "e1g1",
            "tag": "queen_indian_castles",
        },
        {
            "id": "dev_155",
            "pool_id": "v11_dev_reg_05",
            "game_id": "v11_curated_dev_05",
            "task": "explain_move",
            "level": "beginner",
            "detail": "long",
            "fen": _play_uci(["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4", "a2a4", "c8f5"]),
            "move_uci": "h2h3",
            "tag": "slav_passive_h3_misses_ne5",
        },
        {
            "id": "dev_156",
            "pool_id": "v11_dev_reg_06",
            "game_id": "v11_curated_dev_06",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6", "e4e5", "f6d7", "f1d3", "c7c5", "c2c3", "b8c6", "g1f3", "d8b6"]),
            "move_uci": "d4c5",
            "tag": "french_tarrasch_dxc5_inaccuracy",
        },
        {
            "id": "dev_157",
            "pool_id": "v11_dev_reg_07",
            "game_id": "v11_curated_dev_07",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": "2r3k1/pp3pp1/2p4p/8/8/2P4P/PP3PP1/R5K1 w - - 0 20",
            "move_uci": "a1d1",
            "tag": "endgame_rook_open_d_file",
        },
        {
            "id": "dev_158",
            "pool_id": "v11_dev_reg_08",
            "game_id": "v11_curated_dev_08",
            "task": "explain_move",
            "level": "club",
            "detail": "long",
            "fen": _play_uci(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "d2d3", "b7b5", "a4b3", "d7d6"]),
            "move_uci": "d3d4",
            "tag": "ruy_lopez_premature_d4",
        },
        {
            "id": "dev_159",
            "pool_id": "v11_dev_reg_09",
            "game_id": "v11_curated_dev_09",
            "task": "explain_position",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(
                [
                    "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4",
                    "f3d4", "g8f6", "b1c3", "g7g6", "c1e3", "f8g7",
                    "f2f3", "e8g8", "d1d2", "b8c6", "f1c4", "c8d7",
                    "e1c1", "a8c8",
                ]
            ),
            "move_uci": None,
            "tag": "dragon_position_plan",
        },
        {
            "id": "dev_160",
            "pool_id": "v11_dev_reg_10",
            "game_id": "v11_curated_dev_10",
            "task": "explain_position",
            "level": "club",
            "detail": "short",
            "fen": "r1bqkbnr/pppp1ppp/8/4p3/2BnP3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
            "move_uci": None,
            "tag": "noisy_check_suppression_pos",
        },
    ]


def get_curated_test_specs() -> list[dict[str, Any]]:
    """Return twenty curated positional and check-filter regression specs for testset-1.1."""
    return [
        {
            "id": "test_301",
            "pool_id": "v11_test_reg_01",
            "game_id": "v11_curated_test_01",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["d2d4", "g8f6", "g1f3", "e7e6", "c2c4", "f8b4", "c1d2", "d8e7", "g2g3", "e6e5", "f1g2"]),
            "move_uci": "e5d4",
            "tag": "bogo_6_exd4_vs_e4",
        },
        {
            "id": "test_302",
            "pool_id": "v11_test_reg_02",
            "game_id": "v11_curated_test_02",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3", "a7a6"]),
            "move_uci": "a2a4",
            "tag": "giuoco_piano_a4",
        },
        {
            "id": "test_303",
            "pool_id": "v11_test_reg_03",
            "game_id": "v11_curated_test_03",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8", "f1d3", "d7d5"]),
            "move_uci": "c4d5",
            "tag": "nimzo_cxd5_trade",
        },
        {
            "id": "test_304",
            "pool_id": "v11_test_reg_04",
            "game_id": "v11_curated_test_04",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["e2e4", "c7c5", "b1c3", "b8c6", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "a7a6"]),
            "move_uci": "d4c6",
            "tag": "sicilian_nxc6_recapture",
        },
        {
            "id": "test_305",
            "pool_id": "v11_test_reg_05",
            "game_id": "v11_curated_test_05",
            "task": "explain_move",
            "level": "club",
            "detail": "long",
            "fen": _play_uci(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e6", "f1e2", "a7a6", "e1g1", "f8e7", "f2f4", "e8g8", "c1e3"]),
            "move_uci": "b7b6",
            "tag": "sicilian_passive_b6_misses_e5_space",
        },
        {
            "id": "test_306",
            "pool_id": "v11_test_reg_06",
            "game_id": "v11_curated_test_06",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f6e4", "d2d4", "f8e7"]),
            "move_uci": "d4e5",
            "tag": "berlin_dxe5_capture",
        },
        {
            "id": "test_307",
            "pool_id": "v11_test_reg_07",
            "game_id": "v11_curated_test_07",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "f8b4", "f1g2", "e8g8"]),
            "move_uci": "a2a3",
            "tag": "english_a3_vs_bishop",
        },
        {
            "id": "test_308",
            "pool_id": "v11_test_reg_08",
            "game_id": "v11_curated_test_08",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6"]),
            "move_uci": "e2e4",
            "tag": "benoni_central_space_e4",
        },
        {
            "id": "test_309",
            "pool_id": "v11_test_reg_09",
            "game_id": "v11_curated_test_09",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3", "f8g7", "f1e2", "e8g8"]),
            "move_uci": "e1g1",
            "tag": "pirc_castles_safety",
        },
        {
            "id": "test_310",
            "pool_id": "v11_test_reg_10",
            "game_id": "v11_curated_test_10",
            "task": "explain_move",
            "level": "beginner",
            "detail": "long",
            "fen": _play_uci(["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "e7e6", "e2e3", "c7c5", "c2c3", "b8c6"]),
            "move_uci": "d4c5",
            "tag": "london_dxc5_releases_tension",
        },
        {
            "id": "test_311",
            "pool_id": "v11_test_reg_11",
            "game_id": "v11_curated_test_11",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "c8d7", "b5d7", "d8d7", "c2c4", "b8c6"]),
            "move_uci": "b1c3",
            "tag": "maroczy_nc3_central_control",
        },
        {
            "id": "test_312",
            "pool_id": "v11_test_reg_12",
            "game_id": "v11_curated_test_12",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8", "h2h3", "c6a5", "b3c2", "c7c5"]),
            "move_uci": "g2g3",
            "tag": "ruy_chigorin_g3_misses_d4_center",
        },
        {
            "id": "test_313",
            "pool_id": "v11_test_reg_13",
            "game_id": "v11_curated_test_13",
            "task": "explain_move",
            "level": "club",
            "detail": "long",
            "fen": _play_uci(["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6", "e4e5", "f6d7", "f1d3", "c7c5"]),
            "move_uci": "d4c5",
            "tag": "french_tarrasch_dxc5",
        },
        {
            "id": "test_314",
            "pool_id": "v11_test_reg_14",
            "game_id": "v11_curated_test_14",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5", "c2c4", "g8f6", "b1c3", "e7e6"]),
            "move_uci": "c4c5",
            "tag": "panov_c5_space_push",
        },
        {
            "id": "test_315",
            "pool_id": "v11_test_reg_15",
            "game_id": "v11_curated_test_15",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "e2e3", "c8f5", "b1c3", "e7e6", "f3h4", "f5g6"]),
            "move_uci": "c4c5",
            "tag": "slav_premature_c5_releases_tension",
        },
        {
            "id": "test_316",
            "pool_id": "v11_test_reg_16",
            "game_id": "v11_curated_test_16",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4", "c5b4", "c2c3", "b4a5"]),
            "move_uci": "d2d4",
            "tag": "evans_gambit_d4_center",
        },
        {
            "id": "test_317",
            "pool_id": "v11_test_reg_17",
            "game_id": "v11_curated_test_17",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4", "d5c3"]),
            "move_uci": "b2c3",
            "tag": "grunfeld_bxc3_center",
        },
        {
            "id": "test_318",
            "pool_id": "v11_test_reg_18",
            "game_id": "v11_curated_test_18",
            "task": "explain_position",
            "level": "beginner",
            "detail": "short",
            "fen": _play_uci(
                [
                    "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4",
                    "d1c2", "e8g8", "a2a3", "b4c3", "c2c3", "b7b6",
                ]
            ),
            "move_uci": None,
            "tag": "nimzo_classical_position",
        },
        {
            "id": "test_319",
            "pool_id": "v11_test_reg_19",
            "game_id": "v11_curated_test_19",
            "task": "explain_position",
            "level": "club",
            "detail": "short",
            "fen": _play_uci(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "b7b5"]),
            "move_uci": None,
            "tag": "ruy_closed_position",
        },
        {
            "id": "test_320",
            "pool_id": "v11_test_reg_20",
            "game_id": "v11_curated_test_20",
            "task": "explain_position",
            "level": "beginner",
            "detail": "long",
            "fen": _play_uci(
                [
                    "d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6",
                    "c4d5", "e6d5", "c1g5", "c7c6", "e2e3", "a7a6",
                    "f1d3", "h7h6", "g5h4",
                ]
            ),
            "move_uci": None,
            "tag": "carlsbad_structure_position",
        },
    ]


def refresh_record_v1_1(rec: dict[str, Any]) -> dict[str, Any]:
    """Rebuild facts, user_message, chat_prompt, and concept_tier using v1.1 core."""
    updated = rebuild_v1_1_record(rec)
    updated["phase"] = updated.get("phase") or extract_phase(str(updated["fen"]))
    updated["concept_tier"] = classify_concept_tier(updated)
    return updated


def build_curated_record(
    spec: dict[str, Any],
    analyzer: Callable[[chess.Board, int], dict[str, Any]],
    depth: int = DEFAULT_STOCKFISH_DEPTH,
) -> dict[str, Any]:
    """Analyze a curated regression spec with Stockfish and return a formatted item."""
    fen = str(spec["fen"])
    task = str(spec["task"])
    move_uci = spec.get("move_uci")
    spec_depth = int(spec.get("pinned_depth", depth))

    board = chess.Board(fen)
    eval_before = analyzer(board, spec_depth)
    eval_after: dict[str, Any] | None = None
    if task == "explain_move":
        if not move_uci:
            raise ValueError(f"Curated explain_move spec {spec['id']} missing move_uci")
        board_after = board.copy()
        board_after.push(chess.Move.from_uci(str(move_uci)))
        eval_after = analyzer(board_after, spec_depth)

    base_rec: dict[str, Any] = {
        "id": spec["id"],
        "pool_id": spec["pool_id"],
        "game_id": spec["game_id"],
        "fen": fen,
        "task": task,
        "level": spec["level"],
        "detail": spec["detail"],
        "move_uci": move_uci,
        "move_san": None,
        "verdict": None,
        "phase": extract_phase(fen),
        "white_elo": None,
        "black_elo": None,
        "avg_elo": None,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "source": "curated_v1_1",
        "redistributable": True,
        "regression_tag": spec.get("tag", ""),
    }
    return refresh_record_v1_1(base_rec)


def _resolve_move_uci(board: chess.Board, raw_move: str | None) -> str | None:
    """Resolve UCI or SAN move text against board into a legal UCI string."""
    if not raw_move:
        return None
    token = raw_move.strip()
    token = re.sub(r"^\d+\.+\s*", "", token)
    token = token.rstrip(".!?")
    if not token or token == "--" or token.lower() == "position":
        return None
    try:
        m = chess.Move.from_uci(token)
        if _is_king_takes_friendly_rook(board, m):
            return None
        if m in board.legal_moves:
            return m.uci()
    except ValueError:
        pass
    try:
        m = board.parse_san(token)
        if m in board.legal_moves:
            return m.uci()
    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        pass
    return None


def parse_exported_ratings_files(paths: list[Path | str]) -> list[dict[str, Any]]:
    """Parse thumbs-down feedback records from extension JSON exports.

    Deduplicate entries by normalized FEN and resolved UCI move.
    Skip malformed entries whose move text is neither a position marker nor a legal move.
    """
    parsed: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str | None]] = set()

    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Ratings export file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        entries = data.get("ratings", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("rating", "")).lower() != "down":
                continue
            fen = str(entry.get("fen", "")).strip()
            if not fen:
                continue
            try:
                board = chess.Board(fen)
            except ValueError:
                continue

            raw_move = entry.get("move")
            raw_move_str = str(raw_move).strip() if raw_move is not None else ""
            is_position_marker = not raw_move_str or raw_move_str == "--" or raw_move_str.lower() == "position"
            move_uci = _resolve_move_uci(board, raw_move_str)
            if not is_position_marker and move_uci is None:
                LOGGER.warning("Skipping malformed rating entry with illegal move %r on %s", raw_move_str, fen)
                continue

            norm_fen = normalize_fen(fen)
            key = (norm_fen, move_uci)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            level = str(entry.get("level", "club")).lower()
            if level not in ("beginner", "club"):
                level = "club"
            detail = str(entry.get("detail", "short")).lower()
            if detail not in ("short", "long"):
                detail = "short"

            game_id = str(entry.get("game_id") or entry.get("gameId") or "").strip()
            parsed.append({
                "fen": board.fen(),
                "norm_fen": norm_fen,
                "move_uci": move_uci,
                "task": "explain_move" if move_uci else "explain_position",
                "level": level,
                "detail": detail,
                "rating_id": str(entry.get("id", "")),
                "game_id": game_id,
            })

    return parsed


def verify_v1_1_zero_contamination(
    dev_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
) -> None:
    """Verify zero game ID or FEN overlap against training splits and across dev/test."""
    excluded_games, excluded_fens = load_excluded_keys(
        split_dir,
        split_names=ALL_TRAINING_SPLITS,
    )

    dev_fens: set[str] = set()
    dev_games: set[str] = set()
    for item in dev_records:
        gid = str(item.get("game_id", ""))
        n_fen = normalize_fen(str(item.get("fen", "")))
        if gid and gid in excluded_games:
            raise ValueError(f"Dev v1.1 game_id contamination: {gid}")
        if n_fen in excluded_fens:
            raise ValueError(f"Dev v1.1 FEN contamination with training splits: {n_fen}")
        if n_fen in dev_fens:
            raise ValueError(f"Duplicate FEN in devset-1.1: {n_fen}")
        dev_fens.add(n_fen)
        if gid:
            dev_games.add(gid)

    test_fens: set[str] = set()
    for item in test_records:
        gid = str(item.get("game_id", ""))
        n_fen = normalize_fen(str(item.get("fen", "")))
        if gid and gid in excluded_games:
            raise ValueError(f"Test v1.1 game_id contamination: {gid}")
        if gid and gid in dev_games:
            raise ValueError(f"Game ID contamination between devset-1.1 and testset-1.1: {gid}")
        if n_fen in excluded_fens:
            raise ValueError(f"Test v1.1 FEN contamination with training splits: {n_fen}")
        if n_fen in dev_fens:
            raise ValueError(f"Contamination between devset-1.1 and testset-1.1: {n_fen}")
        if n_fen in test_fens:
            raise ValueError(f"Duplicate FEN in testset-1.1: {n_fen}")
        test_fens.add(n_fen)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records from path."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_eval_v1_1(
    dev_in_path: Path | str = DEFAULT_DEVSET_1_0,
    test_in_path: Path | str = DEFAULT_TESTSET_1_0,
    dev_out_path: Path | str = DEFAULT_DEVSET_1_1,
    test_out_path: Path | str = DEFAULT_TESTSET_1_1,
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
    ratings_files: list[Path | str] | None = None,
    stockfish_path: str | None = None,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    analyzer: Callable[[chess.Board, int], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build devset-1.1.jsonl and testset-1.1.jsonl from v1.0 sets, curated cases, and ratings."""
    base_dev = [refresh_record_v1_1(r) for r in _load_jsonl(Path(dev_in_path))]
    base_test = [refresh_record_v1_1(r) for r in _load_jsonl(Path(test_in_path))]

    engine: chess.engine.SimpleEngine | None = None
    if analyzer is None:
        sf_bin = stockfish_path or find_stockfish_binary()
        engine = chess.engine.SimpleEngine.popen_uci(sf_bin)
        active_analyzer: Callable[[chess.Board, int], dict[str, Any]] = (
            lambda b, d: _analyze_board_with_engine(engine, b, depth=d, multipv=3)
        )
    else:
        active_analyzer = analyzer

    try:
        curated_dev = [
            build_curated_record(spec, active_analyzer, depth=depth)
            for spec in get_curated_dev_specs()
        ]
        curated_test = [
            build_curated_record(spec, active_analyzer, depth=depth)
            for spec in get_curated_test_specs()
        ]

        dev_records = base_dev + curated_dev
        test_records = base_test + curated_test

        if ratings_files:
            excluded_games, excluded_fens = load_excluded_keys(
                split_dir,
                split_names=ALL_TRAINING_SPLITS,
            )
            used_fens = set(excluded_fens)
            dev_games = {str(r.get("game_id", "")) for r in dev_records if r.get("game_id")}
            test_games = {str(r.get("game_id", "")) for r in test_records if r.get("game_id")}
            for rec in dev_records + test_records:
                used_fens.add(normalize_fen(str(rec["fen"])))

            rating_items = parse_exported_ratings_files(ratings_files)
            r_dev_idx = 1
            r_test_idx = 1
            for r_item in rating_items:
                n_fen = r_item["norm_fen"]
                raw_gid = str(r_item.get("game_id") or "")
                if raw_gid and raw_gid in excluded_games:
                    continue
                if n_fen in used_fens:
                    continue
                used_fens.add(n_fen)

                routing_key = raw_gid or n_fen
                to_dev = hash_game_id(routing_key, seed=11) < 0.35
                if raw_gid:
                    if raw_gid in dev_games:
                        to_dev = True
                    elif raw_gid in test_games:
                        to_dev = False
                rec_id = f"dev_r_{r_dev_idx:03d}" if to_dev else f"test_r_{r_test_idx:03d}"
                if to_dev:
                    r_dev_idx += 1
                else:
                    r_test_idx += 1

                gid = raw_gid or f"rating_{rec_id}"
                if to_dev:
                    dev_games.add(gid)
                else:
                    test_games.add(gid)

                spec = {
                    "id": rec_id,
                    "pool_id": f"rating_{r_item['rating_id'] or rec_id}",
                    "game_id": gid,
                    "task": r_item["task"],
                    "level": r_item["level"],
                    "detail": r_item["detail"],
                    "fen": r_item["fen"],
                    "move_uci": r_item["move_uci"],
                    "tag": "soft_launch_thumbs_down",
                }
                built = build_curated_record(spec, active_analyzer, depth=depth)
                built["source"] = "soft_launch_ratings"
                if to_dev:
                    dev_records.append(built)
                else:
                    test_records.append(built)
    finally:
        if engine is not None:
            engine.quit()

    verify_v1_1_zero_contamination(dev_records, test_records, split_dir=split_dir)

    _write_jsonl(Path(dev_out_path), dev_records)
    _write_jsonl(Path(test_out_path), test_records)
    return dev_records, test_records


def main() -> None:
    """Parse CLI arguments and build v1.1 evaluation datasets."""
    parser = argparse.ArgumentParser(description="Build Second v1.1 dev and test sets.")
    parser.add_argument("--dev-in", default=str(DEFAULT_DEVSET_1_0))
    parser.add_argument("--test-in", default=str(DEFAULT_TESTSET_1_0))
    parser.add_argument("--dev-out", default=str(DEFAULT_DEVSET_1_1))
    parser.add_argument("--test-out", default=str(DEFAULT_TESTSET_1_1))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--ratings-files", nargs="*", default=[])
    parser.add_argument("--stockfish-path", default=None)
    parser.add_argument("--depth", type=int, default=DEFAULT_STOCKFISH_DEPTH)
    args = parser.parse_args()

    dev_recs, test_recs = build_eval_v1_1(
        dev_in_path=args.dev_in,
        test_in_path=args.test_in,
        dev_out_path=args.dev_out,
        test_out_path=args.test_out,
        split_dir=args.split_dir,
        ratings_files=args.ratings_files or None,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
    )
    print(f"Built {len(dev_recs)} dev items at {args.dev_out} and {len(test_recs)} test items at {args.test_out}.")


if __name__ == "__main__":
    main()
