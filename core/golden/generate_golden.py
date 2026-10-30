"""Generate the golden dataset for the Second chess assistant.

This script creates 200 golden test cases.
It analyzes positions with Stockfish.
It produces 100 percent byte parity between Python and TypeScript.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure benchmark positions directory is in sys.path
BENCHMARK_DIR = PROJECT_ROOT / "spikes" / "colab-stockfish"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import chess
import chess.engine
from positions import BENCHMARK_POSITIONS

from core.python import (
    build_user_message,
    classify_verdict,
    format_qwen_chat_prompt,
)
from core.python.facts import (
    compute_material_fact,
    compute_phase_fact,
    win_chance_side,
)


def find_stockfish_binary() -> str:
    """Find the Stockfish executable binary.

    Check the local spikes directory first.
    If not found, search the system PATH.
    """
    local_bin = (
        PROJECT_ROOT
        / "spikes"
        / "colab-stockfish"
        / "stockfish_bin"
        / "stockfish"
        / "stockfish-linux-x86-64-universal"
    )
    if local_bin.exists() and os.access(local_bin, os.X_OK):
        return str(local_bin)

    import shutil
    path_bin = shutil.which("stockfish")
    if path_bin:
        return path_bin

    raise FileNotFoundError("Stockfish binary not found.")


def analyze_board(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int = 10,
    multipv: int = 3,
) -> dict[str, Any]:
    """Run Stockfish analysis on a chess position.

    Convert scores to White perspective.
    Return evaluation dictionary with FEN and variation lines.
    """
    if board.is_game_over():
        if board.is_checkmate():
            # If the side to move has checkmate on the board, the other side won.
            val = -1 if board.turn == chess.WHITE else 1
            return {
                "fen": board.fen(),
                "pvs": [{"score": {"type": "mate", "value": val}, "moves": []}],
            }
        return {
            "fen": board.fen(),
            "pvs": [{"score": {"type": "cp", "value": 0}, "moves": []}],
        }

    actual_multipv = min(multipv, len(list(board.legal_moves)))
    if actual_multipv == 0:
        actual_multipv = 1

    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=actual_multipv)
    if not isinstance(info, list):
        info = [info]

    pvs: list[dict[str, Any]] = []
    for entry in info:
        score = entry["score"].white()
        if score.is_mate():
            s_dict = {"type": "mate", "value": score.mate()}
        else:
            s_dict = {"type": "cp", "value": score.score() or 0}

        moves = [m.uci() for m in entry.get("pv", [])]
        pvs.append({"score": s_dict, "moves": moves})

    return {"fen": board.fen(), "pvs": pvs}


def get_material_template_fens() -> list[str]:
    """Return positions covering all seventeen material balance templates."""
    return [
        "8/8/8/8/8/8/8/4K2k w - - 0 1",              # 1. equal
        "8/8/8/8/8/8/4P3/4K2k w - - 0 1",          # 2. up a pawn
        "8/8/8/8/8/8/2P1P3/4K2k w - - 0 1",        # 3. up two pawns
        "8/8/8/8/8/8/8/2N1K2k w - - 0 1",          # 4. up a knight
        "8/8/8/8/8/8/8/2B1K2k w - - 0 1",          # 5. up a bishop
        "8/8/8/8/8/8/8/2R1K2k w - - 0 1",          # 6. up a rook
        "8/8/8/8/8/8/8/2Q1K2k w - - 0 1",          # 7. up a queen
        "8/8/8/8/8/8/8/2R1K1nk w - - 0 1",          # 8. up the exchange
        "8/8/8/8/8/8/2pp4/2B1K2k w - - 0 1",       # 9. up a pawn in material
        "8/8/8/8/8/8/2p5/2N1K2k w - - 0 1",        # 10. up two pawns in material
        "8/8/8/8/8/8/2pp4/2R1K2k w - - 0 1",       # 11. up a minor piece in material
        "8/8/8/8/8/8/4P3/2B1K2k w - - 0 1",        # 12. up a minor piece and a pawn in material
        "8/8/8/8/8/8/2p5/2BBK2k w - - 0 1",        # 13. up a rook in material
        "8/8/8/8/8/8/8/2BBK2k w - - 0 1",          # 14. up two minor pieces in material
        "8/8/8/8/8/8/2P1P3/2R1K2k w - - 0 1",      # 15. up a rook and two pawns in material
        "8/8/8/8/8/8/8/2RBK2k w - - 0 1",          # 16. up a rook and a minor piece in material
        "8/8/8/8/8/8/4P3/2Q1K2k w - - 0 1",        # 17. up a queen in material
    ]


def get_tactical_and_special_fens() -> list[dict[str, Any]]:
    """Return targeted positions for castling, promotions, en passant, and only moves."""
    return [
        # Castling cases
        {
            "fen": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "move_uci": "e1g1",
            "target_verdict": None,
        },
        {
            "fen": "r3k2r/pppq1ppp/2np1n2/2b1p3/4P3/1PNP1N2/P1P2PPP/R3K2R w KQkq - 0 8",
            "move_uci": "e1c1",
            "target_verdict": None,
        },
        {
            "fen": "r1bqk2r/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
            "move_uci": "e8g8",
            "target_verdict": None,
        },
        {
            "fen": "r3k2r/pppq1ppp/2np1n2/2b1p3/4P3/1PNP1N2/P1P2PPP/R1BQ1RK1 b kq - 0 8",
            "move_uci": "e8c8",
            "target_verdict": None,
        },
        # En passant cases
        {
            "fen": "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "move_uci": "e5d6",
            "target_verdict": None,
        },
        {
            "fen": "rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 2",
            "move_uci": "e4f3",
            "target_verdict": None,
        },
        {
            "fen": "8/8/8/kpP4R/8/8/8/4K3 w - b6 0 1",
            "move_uci": "c5b6",
            "target_verdict": None,
        },
        # Pawn promotions
        {
            "fen": "8/4P3/8/8/8/8/8/4K2k w - - 0 1",
            "move_uci": "e7e8q",
            "target_verdict": None,
        },
        {
            "fen": "8/2P5/8/8/8/8/8/4K2k w - - 0 1",
            "move_uci": "c7c8n",
            "target_verdict": None,
        },
        {
            "fen": "4k2K/8/8/8/8/8/2p5/8 b - - 0 1",
            "move_uci": "c2c1q",
            "target_verdict": None,
        },
        {
            "fen": "4k2K/8/8/8/8/8/4p3/8 b - - 0 1",
            "move_uci": "e2e1r",
            "target_verdict": None,
        },
        {
            "fen": "4k2K/8/8/8/8/8/6p1/8 b - - 0 1",
            "move_uci": "g2g1b",
            "target_verdict": None,
        },
        # Uncastled king at move >= 10
        {
            "fen": "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPP3PP/R2QK2R w KQ - 1 10",
            "move_uci": "c4b3",
            "target_verdict": None,
        },
        {
            "fen": "r1bqk2r/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPP3PP/R2Q1RK1 b kq - 1 10",
            "move_uci": "e8g8",
            "target_verdict": None,
        },
        # Forced Only Move tactical positions
        {
            "fen": "4k3/8/8/8/8/8/4q3/4R1K1 w - - 0 1",
            "move_uci": "e1e2",
            "target_verdict": "Only move",
        },
        {
            "fen": "4k3/8/8/8/8/8/4r3/4Q1K1 w - - 0 1",
            "move_uci": "e1e2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/4k3/8/4p3/4K3 w - - 0 1",
            "move_uci": "e1e2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/3k4/8/3p4/3K4 w - - 0 1",
            "move_uci": "d1d2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/2k5/8/2p5/2K5 w - - 0 1",
            "move_uci": "c1c2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/1k6/8/1p6/1K6 w - - 0 1",
            "move_uci": "b1b2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/k7/8/p7/K7 w - - 0 1",
            "move_uci": "a1a2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/7k/6p1/6K1 w - - 0 1",
            "move_uci": "g1f2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/6k1/5p2/5K2 w - - 0 1",
            "move_uci": "f1e2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/k7/1p6/K7 w - - 0 1",
            "move_uci": "a1b1",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/1k6/2p5/2K5 w - - 0 1",
            "move_uci": "c1d2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/2k5/3p4/3K4 w - - 0 1",
            "move_uci": "d1e2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/3k4/4p3/4K3 w - - 0 1",
            "move_uci": "e1f2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/4k3/5p2/5K2 w - - 0 1",
            "move_uci": "f1g2",
            "target_verdict": "Only move",
        },
        {
            "fen": "7k/7R/8/8/8/8/8/7K b - - 0 1",
            "move_uci": "h8h7",
            "target_verdict": "Only move",
        },
        {
            "fen": "6k1/6R1/8/8/8/8/8/6K1 b - - 0 1",
            "move_uci": "g8g7",
            "target_verdict": "Only move",
        },
        {
            "fen": "5k2/5R2/8/8/8/8/8/5K2 b - - 0 1",
            "move_uci": "f8f7",
            "target_verdict": "Only move",
        },
        {
            "fen": "4k3/4R3/8/8/8/8/8/4K3 b - - 0 1",
            "move_uci": "e8e7",
            "target_verdict": "Only move",
        },
        {
            "fen": "3k4/3R4/8/8/8/8/8/3K4 b - - 0 1",
            "move_uci": "d8d7",
            "target_verdict": "Only move",
        },
        {
            "fen": "2k5/2R5/8/8/8/8/8/2K5 b - - 0 1",
            "move_uci": "c8c7",
            "target_verdict": "Only move",
        },
        {
            "fen": "1k6/1R6/8/8/8/8/8/1K6 b - - 0 1",
            "move_uci": "b8b7",
            "target_verdict": "Only move",
        },
        {
            "fen": "k7/R7/8/8/8/8/8/K7 b - - 0 1",
            "move_uci": "a8a7",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/8/1r6/1K5k w - - 0 1",
            "move_uci": "b1b2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/8/r7/K6k w - - 0 1",
            "move_uci": "a1a2",
            "target_verdict": "Only move",
        },
        {
            "fen": "8/8/8/8/8/8/5r2/5K1k w - - 0 1",
            "move_uci": "f1f2",
            "target_verdict": "Only move",
        },
        # Material win in line (Case 2: reply does not capture, ply 2 captures piece)
        {
            "fen": "r1bqkb1r/pp1p1ppp/2n5/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 5",
            "move_uci": "f3e5",
            "target_verdict": None,
            "eval_after": {
                "fen": "r1bqkb1r/pp1p1ppp/2n5/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 5",
                "pvs": [
                    {
                        "score": {"type": "cp", "value": -150},
                        "moves": ["d8a5", "b1c3", "a5e5"],
                    }
                ],
            },
        },
        {
            "fen": "rnbqkb1r/pppp1ppp/8/8/4n3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 2",
            "move_uci": "d7d5",
            "target_verdict": None,
            "eval_after": {
                "fen": "rnbqkb1r/ppp2ppp/8/3p4/4n3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
                "pvs": [
                    {
                        "score": {"type": "cp", "value": 150},
                        "moves": ["d1e2", "d8e7", "e2e4"],
                    }
                ],
            },
        },
        {
            "fen": "r1bqk2r/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2QK2R b KQkq - 2 8",
            "move_uci": "e8g8",
            "target_verdict": None,
        },
        {
            "fen": "r2q1rk1/pp1b1ppp/2n1pn2/2pp4/2PP4/1PN1PN2/PB2BPPP/R2Q1RK1 b - - 0 9",
            "move_uci": "c5d4",
            "target_verdict": None,
        },
    ]


def main() -> None:
    """Run generation and write golden_200.json."""
    print("Finding Stockfish binary...")
    sf_path = find_stockfish_binary()
    print(f"Using Stockfish binary at: {sf_path}")

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": 4, "Hash": 128})

    print("Collecting candidate positions...")
    # Gather initial pool
    all_fens: list[str] = []
    seen_fens: set[str] = set()

    def add_fen(fen_str: str) -> bool:
        norm = chess.Board(fen_str).fen()
        if norm not in seen_fens:
            seen_fens.add(norm)
            all_fens.append(norm)
            return True
        return False

    # 1. 100 benchmark positions
    for f in BENCHMARK_POSITIONS:
        add_fen(f)

    # 2. 17 material template positions
    for f in get_material_template_fens():
        add_fen(f)

    # 3. Targeted tactical positions
    special_cases = get_tactical_and_special_fens()
    for item in special_cases:
        add_fen(item["fen"])

    # 4. Step forward moves from benchmark positions until we have ample unique positions
    for fen_str in list(all_fens):
        b = chess.Board(fen_str)
        for m in b.legal_moves:
            b2 = b.copy()
            b2.push(m)
            if not b2.is_game_over():
                add_fen(b2.fen())
            if len(all_fens) >= 300:
                break
        if len(all_fens) >= 300:
            break

    print(f"Collected {len(all_fens)} unique candidate positions.")

    # Goal: exactly 150 explain_move cases (25 of each of the 6 verdicts)
    # and exactly 50 explain_position cases.
    verdict_targets = {
        "Best": 25,
        "Only move": 25,
        "Good": 25,
        "Inaccuracy": 25,
        "Mistake": 25,
        "Blunder": 25,
    }
    verdict_counts = {v: 0 for v in verdict_targets}

    explain_move_cases: list[dict[str, Any]] = []
    used_fens: set[str] = set()

    # First, process special targeted moves
    print("Evaluating special targeted cases...")
    for item in special_cases:
        fen_norm = chess.Board(item["fen"]).fen()
        if fen_norm in used_fens:
            continue
        move_uci = item["move_uci"]
        board = chess.Board(fen_norm)
        move_obj = chess.Move.from_uci(move_uci)
        if move_obj not in board.legal_moves:
            continue

        eval_before = analyze_board(engine, board, depth=10, multipv=3)
        if "eval_after" in item and item["eval_after"] is not None:
            eval_after = item["eval_after"]
        else:
            b_after = board.copy()
            b_after.push(move_obj)
            eval_after = analyze_board(engine, b_after, depth=10, multipv=1)

        v = classify_verdict(eval_before, eval_after, move_uci)
        if verdict_counts[v] < verdict_targets[v]:
            verdict_counts[v] += 1
            used_fens.add(fen_norm)
            explain_move_cases.append({
                "task": "explain_move",
                "fen": fen_norm,
                "move_uci": move_uci,
                "eval_before": eval_before,
                "eval_after": eval_after,
                "verdict": v,
            })

    print(f"Special cases added: {len(explain_move_cases)}. Verdict counts: {verdict_counts}")

    # Next, search candidate positions to fill remaining verdict quotas
    print("Searching positions to fill verdict quotas...")
    for fen_norm in all_fens:
        if fen_norm in used_fens:
            continue
        if all(verdict_counts[v] >= verdict_targets[v] for v in verdict_targets):
            break

        board = chess.Board(fen_norm)
        if board.is_game_over() or len(list(board.legal_moves)) == 0:
            continue

        eval_before = analyze_board(engine, board, depth=10, multipv=3)
        pvs = eval_before.get("pvs", [])
        if not pvs or not pvs[0].get("moves"):
            continue

        best_move_uci = pvs[0]["moves"][0]
        is_white = (board.turn == chess.WHITE)
        wc_before = win_chance_side(pvs[0]["score"], is_white)

        # Check if best_move is Only move
        if verdict_counts["Only move"] < verdict_targets["Only move"]:
            is_candidate_only_move = False
            if board.legal_moves.count() == 1:
                is_candidate_only_move = True
            elif len(pvs) >= 2 and "score" in pvs[1]:
                wc_sec = win_chance_side(pvs[1]["score"], is_white)
                if (wc_before - wc_sec) >= 15.0:
                    is_candidate_only_move = True

            if is_candidate_only_move:
                b_after = board.copy()
                b_after.push(chess.Move.from_uci(best_move_uci))
                eval_after = analyze_board(engine, b_after, depth=10, multipv=1)
                v = classify_verdict(eval_before, eval_after, best_move_uci)
                if v == "Only move":
                    verdict_counts["Only move"] += 1
                    used_fens.add(fen_norm)
                    explain_move_cases.append({
                        "task": "explain_move",
                        "fen": fen_norm,
                        "move_uci": best_move_uci,
                        "eval_before": eval_before,
                        "eval_after": eval_after,
                        "verdict": v,
                    })
                    continue

        # Check if best_move qualifies as Best
        if verdict_counts["Best"] < verdict_targets["Best"]:
            b_after = board.copy()
            b_after.push(chess.Move.from_uci(best_move_uci))
            eval_after = analyze_board(engine, b_after, depth=10, multipv=1)
            v = classify_verdict(eval_before, eval_after, best_move_uci)
            if v == "Best":
                verdict_counts["Best"] += 1
                used_fens.add(fen_norm)
                explain_move_cases.append({
                    "task": "explain_move",
                    "fen": fen_norm,
                    "move_uci": best_move_uci,
                    "eval_before": eval_before,
                    "eval_after": eval_after,
                    "verdict": v,
                })
                continue

        # Check other legal moves for Good, Inaccuracy, Mistake, Blunder
        needed_verdicts = [v for v in ["Good", "Inaccuracy", "Mistake", "Blunder"] if verdict_counts[v] < verdict_targets[v]]
        if not needed_verdicts:
            continue

        chosen_move = None
        chosen_eval_after = None
        chosen_v = None

        for m in board.legal_moves:
            m_uci = m.uci()
            if m_uci == best_move_uci:
                continue
            b_after = board.copy()
            b_after.push(m)
            eval_after = analyze_board(engine, b_after, depth=10, multipv=1)
            v = classify_verdict(eval_before, eval_after, m_uci)
            if v in needed_verdicts:
                chosen_move = m_uci
                chosen_eval_after = eval_after
                chosen_v = v
                break

        if chosen_move and chosen_eval_after and chosen_v:
            verdict_counts[chosen_v] += 1
            used_fens.add(fen_norm)
            explain_move_cases.append({
                "task": "explain_move",
                "fen": fen_norm,
                "move_uci": chosen_move,
                "eval_before": eval_before,
                "eval_after": chosen_eval_after,
                "verdict": chosen_v,
            })

    print(f"Total explain_move cases built: {len(explain_move_cases)}. Verdict counts: {verdict_counts}")

    # Build 50 explain_position cases from unused positions
    print("Building 50 explain_position cases...")
    explain_pos_cases: list[dict[str, Any]] = []

    # Include material template positions in explain_position cases if unused
    for f in get_material_template_fens():
        norm = chess.Board(f).fen()
        if norm not in used_fens:
            b = chess.Board(norm)
            eval_b = analyze_board(engine, b, depth=10, multipv=3)
            used_fens.add(norm)
            explain_pos_cases.append({
                "task": "explain_position",
                "fen": norm,
                "move_uci": None,
                "eval_before": eval_b,
                "eval_after": None,
            })
            if len(explain_pos_cases) >= 50:
                break

    for fen_norm in all_fens:
        if len(explain_pos_cases) >= 50:
            break
        if fen_norm in used_fens:
            continue

        b = chess.Board(fen_norm)
        if b.is_game_over() or len(list(b.legal_moves)) == 0:
            continue

        eval_b = analyze_board(engine, b, depth=10, multipv=3)
        used_fens.add(fen_norm)
        explain_pos_cases.append({
            "task": "explain_position",
            "fen": fen_norm,
            "move_uci": None,
            "eval_before": eval_b,
            "eval_after": None,
        })

    print(f"Total explain_position cases built: {len(explain_pos_cases)}")

    engine.quit()

    # Combine into 200 cases
    total_cases_raw = explain_move_cases + explain_pos_cases
    assert len(total_cases_raw) == 200, f"Expected 200 cases, got {len(total_cases_raw)}"

    # Assign alternating levels and details
    # 100 beginner, 100 club
    # 100 short, 100 long
    golden_items: list[dict[str, Any]] = []
    levels = ["beginner", "club"]
    details = ["short", "long"]

    for idx, c in enumerate(total_cases_raw):
        level = levels[idx % 2]
        detail = details[(idx // 2) % 2]
        task = c["task"]
        fen = c["fen"]
        move_uci = c["move_uci"]
        eval_before = c["eval_before"]
        eval_after = c["eval_after"]

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
        }
        golden_items.append(item)

    out_path = PROJECT_ROOT / "core" / "golden" / "golden_200.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden_items, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully wrote {len(golden_items)} items to {out_path}!")

    # Verify requirements
    print("\n--- Verifying Requirements ---")
    fens = [c["fen"] for c in golden_items]
    print(f"Total cases: {len(golden_items)}")
    print(f"Unique FENs: {len(set(fens))} (Expected: 200)")

    tasks = [c["task"] for c in golden_items]
    print(f"Task counts: explain_move={tasks.count('explain_move')}, explain_position={tasks.count('explain_position')}")

    levels_count = [c["level"] for c in golden_items]
    print(f"Level counts: beginner={levels_count.count('beginner')}, club={levels_count.count('club')}")

    details_count = [c["detail"] for c in golden_items]
    print(f"Detail counts: short={details_count.count('short')}, long={details_count.count('long')}")

    print("Verdicts in explain_move:", verdict_counts)


if __name__ == "__main__":
    main()
