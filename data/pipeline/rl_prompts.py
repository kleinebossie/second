"""Reinforcement learning prompt set generator module.

This module generates prompt sets for reinforcement learning from CC0 Lichess games.
It guarantees that no position or game overlaps with evaluation splits.
It maintains the target distribution of tasks, verdicts, levels, and detail lengths.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import chess
import pandas as pd

from core.python import (
    build_user_message,
    classify_verdict,
    extract_facts,
    format_qwen_chat_prompt,
)
try:
    from .analyse import (
        StockfishAnalyzer,
        StockfishCache,
    )
except ImportError:
    from data.pipeline.analyse import (  # type: ignore
        StockfishAnalyzer,
        StockfishCache,
    )

LOGGER = logging.getLogger(__name__)
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_PUBLIC_DIR = Path("data/processed/public")


def load_excluded_keys(
    split_dir: Path | str,
) -> tuple[set[str], set[str]]:
    """Load game identifiers and FEN positions from validation and test splits.

    Read sft_val and sft_test_pool files if they exist.
    Return a tuple containing excluded game identifiers and excluded FEN strings.
    """
    p_dir = Path(split_dir)
    excluded_games: set[str] = set()
    excluded_fens: set[str] = set()

    for name in ("sft_val", "sft_test_pool"):
        jsonl_path = p_dir / f"{name}.jsonl"
        parquet_path = p_dir / f"{name}.parquet"

        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if item.get("game_id"):
                            excluded_games.add(str(item["game_id"]))
                        if item.get("fen"):
                            excluded_fens.add(str(item["fen"]))
        elif parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if "game_id" in df.columns:
                excluded_games.update(df["game_id"].dropna().astype(str).tolist())
            if "fen" in df.columns:
                excluded_fens.update(df["fen"].dropna().astype(str).tolist())

    LOGGER.info(
        "Loaded exclusions: %d game IDs and %d FENs",
        len(excluded_games),
        len(excluded_fens),
    )
    return excluded_games, excluded_fens


def build_rl_prompts_from_evaluated(
    evaluated_candidates: list[dict[str, Any]],
    target_count: int,
    excluded_games: set[str] | None = None,
    excluded_fens: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble balanced RL prompt records from evaluated candidates.

    Exclude positions or games that appear in validation or test splits.
    Balance tasks and verdicts:
    - 45 percent explain_move with Inaccuracy, Mistake, or Blunder.
    - 35 percent explain_move with Good, Best, or Only move.
    - 20 percent explain_position.
    Balance levels: 50 percent beginner, 50 percent club.
    Balance details: 80 percent short, 20 percent long.
    """
    ex_games = excluded_games or set()
    ex_fens = excluded_fens or set()

    bad_move_bucket: list[dict[str, Any]] = []
    good_move_bucket: list[dict[str, Any]] = []
    position_bucket: list[dict[str, Any]] = []

    bad_verdicts = {"Inaccuracy", "Mistake", "Blunder"}
    good_verdicts = {"Good", "Best", "Only move"}

    for item in evaluated_candidates:
        gid = str(item.get("game_id", ""))
        fen = str(item.get("fen", ""))

        if gid and gid in ex_games:
            continue
        if fen in ex_fens:
            continue

        eval_before = item.get("eval_before")
        if not eval_before:
            continue

        # Check candidate move
        move_uci = item.get("move_uci")
        eval_after = item.get("eval_after")
        verdict = item.get("verdict")

        if move_uci and eval_after and not verdict:
            verdict = classify_verdict(eval_before, eval_after, move_uci)

        # Place candidate into matching bucket
        if verdict in bad_verdicts and move_uci and eval_after:
            bad_move_bucket.append({
                "game_id": gid,
                "fen": fen,
                "task": "explain_move",
                "move_uci": move_uci,
                "verdict": verdict,
                "eval_before": eval_before,
                "eval_after": eval_after,
            })
        elif verdict in good_verdicts and move_uci and eval_after:
            good_move_bucket.append({
                "game_id": gid,
                "fen": fen,
                "task": "explain_move",
                "move_uci": move_uci,
                "verdict": verdict,
                "eval_before": eval_before,
                "eval_after": eval_after,
            })

        # Every valid position can also serve as an explain_position candidate
        position_bucket.append({
            "game_id": gid,
            "fen": fen,
            "task": "explain_position",
            "move_uci": None,
            "verdict": None,
            "eval_before": eval_before,
            "eval_after": None,
        })

    # If any bucket was short, synthesize candidate moves from eval_before PVs
    target_bad = int(target_count * 0.45)
    target_good = int(target_count * 0.35)
    target_pos = target_count - target_bad - target_good

    if len(good_move_bucket) < target_good:
        seen_good = {(r["fen"], r["move_uci"]) for r in good_move_bucket}
        for item in evaluated_candidates:
            if len(good_move_bucket) >= target_good:
                break
            gid = str(item.get("game_id", ""))
            fen = str(item.get("fen", ""))
            if (gid and gid in ex_games) or (fen in ex_fens):
                continue
            eval_before = item.get("eval_before")
            if not eval_before or not eval_before.get("pvs"):
                continue
            pvs = eval_before["pvs"]
            if pvs[0].get("moves"):
                best_uci = pvs[0]["moves"][0]
                if (fen, best_uci) in seen_good:
                    continue
                # Construct eval_after from the continuation of pvs[0]
                board = chess.Board(fen)
                try:
                    m_obj = chess.Move.from_uci(best_uci)
                    if m_obj not in board.legal_moves:
                        continue
                    board.push(m_obj)
                except ValueError:
                    continue
                synth_after = {
                    "fen": board.fen(),
                    "pvs": [
                        {
                            "score": pvs[0]["score"],
                            "moves": pvs[0]["moves"][1:],
                        }
                    ],
                }
                v = classify_verdict(eval_before, synth_after, best_uci)
                if v in good_verdicts:
                    seen_good.add((fen, best_uci))
                    good_move_bucket.append({
                        "game_id": gid,
                        "fen": fen,
                        "task": "explain_move",
                        "move_uci": best_uci,
                        "verdict": v,
                        "eval_before": eval_before,
                        "eval_after": synth_after,
                    })

    if len(bad_move_bucket) < target_bad:
        seen_bad = {(r["fen"], r["move_uci"]) for r in bad_move_bucket}
        for item in evaluated_candidates:
            if len(bad_move_bucket) >= target_bad:
                break
            gid = str(item.get("game_id", ""))
            fen = str(item.get("fen", ""))
            if (gid and gid in ex_games) or (fen in ex_fens):
                continue
            eval_before = item.get("eval_before")
            if not eval_before or len(eval_before.get("pvs", [])) < 2:
                continue
            pvs = eval_before["pvs"]
            for alt_pv in pvs[1:]:
                if not alt_pv.get("moves"):
                    continue
                alt_uci = alt_pv["moves"][0]
                if (fen, alt_uci) in seen_bad:
                    continue
                board = chess.Board(fen)
                try:
                    m_obj = chess.Move.from_uci(alt_uci)
                    if m_obj not in board.legal_moves:
                        continue
                    board.push(m_obj)
                except ValueError:
                    continue
                synth_after = {
                    "fen": board.fen(),
                    "pvs": [
                        {
                            "score": alt_pv["score"],
                            "moves": alt_pv["moves"][1:],
                        }
                    ],
                }
                v = classify_verdict(eval_before, synth_after, alt_uci)
                if v in bad_verdicts:
                    seen_bad.add((fen, alt_uci))
                    bad_move_bucket.append({
                        "game_id": gid,
                        "fen": fen,
                        "task": "explain_move",
                        "move_uci": alt_uci,
                        "verdict": v,
                        "eval_before": eval_before,
                        "eval_after": synth_after,
                    })
                    break

    selected_raw: list[dict[str, Any]] = []
    selected_raw.extend(bad_move_bucket[:target_bad])
    selected_raw.extend(good_move_bucket[:target_good])
    selected_raw.extend(position_bucket[:target_pos])

    # If any bucket was short, backfill from remaining candidates
    remaining_needed = target_count - len(selected_raw)
    if remaining_needed > 0:
        seen_keys = {(r["fen"], r.get("move_uci")) for r in selected_raw}
        for pool in (position_bucket, good_move_bucket, bad_move_bucket):
            for cand in pool:
                key = (cand["fen"], cand.get("move_uci"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    selected_raw.append(cand)
                    if len(selected_raw) >= target_count:
                        break
            if len(selected_raw) >= target_count:
                break

    # Format prompt records with balanced levels and details
    final_prompts: list[dict[str, Any]] = []
    for idx, item in enumerate(selected_raw):
        # 50% beginner, 50% club
        level = "beginner" if (idx % 2 == 0) else "club"
        # 80% short, 20% long (4 short, 1 long)
        detail = "long" if (idx % 5 == 4) else "short"

        task = item["task"]
        fen = item["fen"]
        move_uci = item["move_uci"]
        eval_before = item["eval_before"]
        eval_after = item["eval_after"]
        verdict = item["verdict"]

        facts = extract_facts(
            task=task,
            fen=fen,
            eval_before=eval_before,
            eval_after=eval_after,
            move_uci=move_uci,
        )

        user_message = build_user_message(
            task=task,
            level=level,
            detail=detail,
            fen=fen,
            eval_before=eval_before,
            eval_after=eval_after,
            move_uci=move_uci,
        )
        chat_prompt = format_qwen_chat_prompt(user_message)

        record = {
            "id": f"rl_{idx:06d}",
            "game_id": item["game_id"],
            "fen": fen,
            "task": task,
            "level": level,
            "detail": detail,
            "move_uci": move_uci,
            "verdict": verdict,
            "facts": facts,
            "user_message": user_message,
            "chat_prompt": chat_prompt,
            "source": "lichess_games",
            "redistributable": True,
            "eval_before": eval_before,
            "eval_after": eval_after,
        }
        final_prompts.append(record)

    return final_prompts


def generate_rl_prompts(
    raw_candidates: list[dict[str, Any]],
    target_count: int = 20000,
    analyzer: StockfishAnalyzer | None = None,
    stockfish_path: str | Path | None = None,
    cache_path: Path | str | None = None,
    depth: int = 12,
    num_workers: int = 4,
    split_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> list[dict[str, Any]]:
    """Generate the full reinforcement learning prompt dataset.

    Filter candidates against validation and test split positions.
    Evaluate missing positions with Stockfish.
    Balance tasks, verdicts, player levels, and detail lengths.
    Return formatted prompt records.
    """
    excluded_games, excluded_fens = load_excluded_keys(split_dir)

    # Filter out excluded items upfront
    valid_candidates: list[dict[str, Any]] = []
    for cand in raw_candidates:
        gid = str(cand.get("game_id", ""))
        fen = str(cand.get("fen", ""))
        if (gid and gid in excluded_games) or (fen in excluded_fens):
            continue
        valid_candidates.append(cand)

    if not valid_candidates:
        LOGGER.warning("No valid candidate positions remain after exclusions.")
        return []

    # If evaluations are missing, run Stockfish analysis
    needs_eval = any("eval_before" not in c for c in valid_candidates)
    if needs_eval:
        if analyzer is None:
            analyzer = StockfishAnalyzer(
                stockfish_path=stockfish_path,
                cache_path=cache_path,
                depth=depth,
                num_workers=num_workers,
            )

        # Cap candidates analyzed to avoid redundant engine work beyond target_count
        max_candidates_to_eval = min(len(valid_candidates), max(int(target_count * 1.15), target_count + 100))
        eval_subset = valid_candidates[:max_candidates_to_eval]

        # First evaluate eval_before at MultiPV=3 for the subset
        before_fens: list[tuple[str, int]] = [(cand["fen"], 3) for cand in eval_subset]
        analyzer.analyze_fens(before_fens, flush_interval=500)

        # Identify which candidates need an extra eval_after call
        # Note: if played move is already in pvs[0..2] of eval_before, we can derive its score and continuation directly!
        after_fens: list[tuple[str, int]] = []
        for cand in eval_subset:
            eb = analyzer.cache.get(cand["fen"], min_depth=depth, min_multipv=3)
            cand["eval_before"] = eb
            move_uci = cand.get("move_uci")
            if not move_uci or not eb:
                continue

            matched_pv = None
            for pv in eb.get("pvs", []):
                if pv.get("moves") and pv["moves"][0] == move_uci:
                    matched_pv = pv
                    break

            board = chess.Board(cand["fen"])
            board.push(chess.Move.from_uci(move_uci))
            fen_after = board.fen()

            if matched_pv is not None:
                cand["eval_after"] = {
                    "fen": fen_after,
                    "pvs": [
                        {
                            "score": matched_pv["score"],
                            "moves": matched_pv["moves"][1:],
                        }
                    ],
                }
            else:
                cached_after = analyzer.cache.get(fen_after, min_depth=depth, min_multipv=1)
                if cached_after is not None:
                    cand["eval_after"] = cached_after
                else:
                    after_fens.append((fen_after, 1))

        if after_fens:
            analyzer.analyze_fens(after_fens, flush_interval=500)
            for cand in eval_subset:
                if "eval_after" not in cand and cand.get("move_uci"):
                    board = chess.Board(cand["fen"])
                    board.push(chess.Move.from_uci(cand["move_uci"]))
                    cand["eval_after"] = analyzer.cache.get(board.fen(), min_depth=depth, min_multipv=1)

        valid_candidates = eval_subset

    prompts = build_rl_prompts_from_evaluated(
        valid_candidates,
        target_count=target_count,
        excluded_games=excluded_games,
        excluded_fens=excluded_fens,
    )
    return prompts


def save_rl_prompts(
    prompts: list[dict[str, Any]],
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    public_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Save reinforcement learning prompts to JSONL and Parquet files.

    Save full prompt set to the output directory.
    Save public prompt set to the public directory.
    All RL prompts originate from CC0 Lichess games and are redistributable.
    """
    out_path = Path(output_dir)
    pub_path = Path(public_dir) if public_dir else out_path / "public"

    out_path.mkdir(parents=True, exist_ok=True)
    pub_path.mkdir(parents=True, exist_ok=True)

    jsonl_file = out_path / "rl_prompts.jsonl"
    parquet_file = out_path / "rl_prompts.parquet"
    pub_jsonl = pub_path / "rl_prompts.jsonl"
    pub_parquet = pub_path / "rl_prompts.parquet"

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    # Mirror to public jsonl
    shutil.copy2(jsonl_file, pub_jsonl)

    if prompts:
        df = pd.DataFrame(prompts)
        df.to_parquet(parquet_file, index=False)
        df.to_parquet(pub_parquet, index=False)
    else:
        empty_df = pd.DataFrame()
        empty_df.to_parquet(parquet_file, index=False)
        empty_df.to_parquet(pub_parquet, index=False)

    stats = {
        "count": len(prompts),
        "tasks": {},
        "levels": {},
        "details": {},
        "verdicts": {},
    }
    for p in prompts:
        stats["tasks"][p["task"]] = stats["tasks"].get(p["task"], 0) + 1
        stats["levels"][p["level"]] = stats["levels"].get(p["level"], 0) + 1
        stats["details"][p["detail"]] = stats["details"].get(p["detail"], 0) + 1
        if p.get("verdict"):
            stats["verdicts"][p["verdict"]] = stats["verdicts"].get(p["verdict"], 0) + 1

    LOGGER.info("Saved %d RL prompts to %s and %s", len(prompts), jsonl_file, pub_jsonl)
    return stats


def main() -> None:
    """Command line entry point for generating RL prompts."""
    parser = argparse.ArgumentParser(description="Generate reinforcement learning prompts from Lichess games.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file of candidate games or positions.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Output directory.")
    parser.add_argument("--public-dir", type=str, default=str(DEFAULT_PUBLIC_DIR), help="Public export directory.")
    parser.add_argument("--split-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Directory with SFT splits to exclude.")
    parser.add_argument("--target-count", type=int, default=20000, help="Number of prompts to generate.")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish executable.")
    parser.add_argument("--cache-path", type=str, default=None, help="Stockfish cache Parquet path.")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth.")
    parser.add_argument("--workers", type=int, default=4, help="Stockfish worker count.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    candidates: list[dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    prompts = generate_rl_prompts(
        candidates,
        target_count=args.target_count,
        stockfish_path=args.stockfish_path,
        cache_path=args.cache_path,
        depth=args.depth,
        num_workers=args.workers,
        split_dir=args.split_dir,
    )
    save_rl_prompts(prompts, output_dir=args.output_dir, public_dir=args.public_dir)


if __name__ == "__main__":
    main()
