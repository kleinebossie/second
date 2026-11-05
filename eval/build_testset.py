"""Frozen test set builder module.

This module builds the frozen Phase 4 evaluation test set.
It selects positions deterministically from the evaluation pool.
It validates ratings, phases, levels, and details.
It guarantees zero contamination against training and reinforcement learning data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chess

from core.python.facts import compute_phase_fact
from data.pipeline.rl_prompts import format_prompt_record, load_excluded_keys
from data.pipeline.splits import hash_game_id, normalize_fen
from rewards.api import parse_prompt

LOGGER = logging.getLogger(__name__)

DEFAULT_EVAL_POOL_PATH = Path("data/processed/eval_pool.jsonl")
DEFAULT_RAW_GAMES_PATH = Path("data/raw/lichess_games.jsonl")
DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_OUTPUT_PATH = Path("eval/testset-1.0.jsonl")

DEFAULT_TARGET_COUNT = 300
DEFAULT_SEED = 42

BAD_VERDICTS: frozenset[str] = frozenset({"Inaccuracy", "Mistake", "Blunder"})
GOOD_VERDICTS: frozenset[str] = frozenset({"Good", "Best", "Only move"})

TARGET_SHORT_BAD = 120
TARGET_SHORT_GOOD = 90
TARGET_SHORT_POS = 45

TARGET_LONG_BAD = 21
TARGET_LONG_GOOD = 16
TARGET_LONG_POS = 8

TARGET_TOTAL_BAD = TARGET_SHORT_BAD + TARGET_LONG_BAD  # 141
TARGET_TOTAL_GOOD = TARGET_SHORT_GOOD + TARGET_LONG_GOOD  # 106
TARGET_TOTAL_POS = TARGET_SHORT_POS + TARGET_LONG_POS  # 53


def load_game_ratings(raw_games_path: Path | str) -> dict[str, dict[str, int | None]]:
    """Load player ratings for games from raw Lichess records.

    If the raw games file does not exist, raise FileNotFoundError.
    Return a dictionary that maps game IDs to white and black Elo ratings.
    """
    path = Path(raw_games_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw games file not found: {path}")

    ratings: dict[str, dict[str, int | None]] = {}
    with open(path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            gid = entry.get("game_id")
            if not gid or gid in ratings:
                continue

            white_val = entry.get("white_elo")
            black_val = entry.get("black_elo")
            w_elo = int(white_val) if white_val is not None else None
            b_elo = int(black_val) if black_val is not None else None
            ratings[str(gid)] = {"white_elo": w_elo, "black_elo": b_elo}

    return ratings


def extract_phase(fen: str) -> str:
    """Identify the game phase for a chess position FEN.

    Extract the phase from compute_phase_fact.
    Return 'opening', 'middlegame', or 'endgame'.
    """
    try:
        board = chess.Board(fen)
        phase_fact = compute_phase_fact(board)
        if "opening" in phase_fact:
            return "opening"
        if "endgame" in phase_fact:
            return "endgame"
        return "middlegame"
    except Exception:
        return "middlegame"


def select_bucket_items(
    candidates: list[dict[str, Any]],
    target_count: int,
    seed: int,
    used_games: set[str],
    used_fens: set[str],
) -> list[dict[str, Any]]:
    """Select candidate records deterministically across game phases.

    Sort candidates using a deterministic hash key.
    Group candidates into opening, middlegame, and endgame lists.
    Select candidates by cycling through phases.
    Prefer unselected games and unique positions.
    Return the selected candidate records list.
    """
    if target_count <= 0 or not candidates:
        return []

    def sort_key(item: dict[str, Any]) -> float:
        gid = str(item.get("game_id", ""))
        fen = str(item.get("fen", ""))
        m_uci = str(item.get("move_uci", ""))
        token = f"{gid}_{fen}_{m_uci}"
        return hash_game_id(token, seed=seed)

    sorted_candidates = sorted(candidates, key=sort_key)
    phases = ["opening", "middlegame", "endgame"]
    by_phase: dict[str, list[dict[str, Any]]] = {p: [] for p in phases}
    for item in sorted_candidates:
        phase = item.get("phase", "middlegame")
        if phase not in by_phase:
            phase = "middlegame"
        by_phase[phase].append(item)

    selected: list[dict[str, Any]] = []

    # Pass 1: Select candidates with unseen game ID and unseen normalized FEN
    phase_idx = 0
    stagnant_steps = 0
    while len(selected) < target_count and stagnant_steps < len(phases):
        phase = phases[phase_idx % len(phases)]
        phase_idx += 1
        pool = by_phase[phase]

        found_idx = -1
        for idx, cand in enumerate(pool):
            gid = str(cand.get("game_id", ""))
            norm_fen = normalize_fen(str(cand.get("fen", "")))
            if gid not in used_games and norm_fen not in used_fens:
                found_idx = idx
                break

        if found_idx >= 0:
            picked = pool.pop(found_idx)
            selected.append(picked)
            gid = str(picked.get("game_id", ""))
            norm_fen = normalize_fen(str(picked.get("fen", "")))
            if gid:
                used_games.add(gid)
            if norm_fen:
                used_fens.add(norm_fen)
            stagnant_steps = 0
        else:
            stagnant_steps += 1

    # Pass 2: If target is not reached, allow seen game IDs with unseen FENs
    if len(selected) < target_count:
        stagnant_steps = 0
        while len(selected) < target_count and stagnant_steps < len(phases):
            phase = phases[phase_idx % len(phases)]
            phase_idx += 1
            pool = by_phase[phase]

            found_idx = -1
            for idx, cand in enumerate(pool):
                norm_fen = normalize_fen(str(cand.get("fen", "")))
                if norm_fen not in used_fens:
                    found_idx = idx
                    break

            if found_idx >= 0:
                picked = pool.pop(found_idx)
                selected.append(picked)
                gid = str(picked.get("game_id", ""))
                norm_fen = normalize_fen(str(picked.get("fen", "")))
                if gid:
                    used_games.add(gid)
                if norm_fen:
                    used_fens.add(norm_fen)
                stagnant_steps = 0
            else:
                stagnant_steps += 1

    return selected


def _select_uniform_indices(total: int, count: int) -> set[int]:
    """Select count indices at uniform intervals across a range of length total."""
    if total <= 0 or count <= 0:
        return set()
    target = min(count, total)
    step = total / target
    indices = {int((i + 0.5) * step) for i in range(target)}
    idx = 0
    while len(indices) < target:
        if idx not in indices:
            indices.add(idx)
        idx += 1
    return indices


def partition_and_label_bucket(
    items: list[dict[str, Any]],
    num_long: int,
    beg_long: int,
    beg_short: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition items into short and long detail subsets and assign levels.

    Select long items at uniform intervals across the candidate list.
    Assign beginner and club levels at uniform intervals in each detail subset.
    Return a tuple with the short items list and the long items list.
    """
    total = len(items)
    if total == 0:
        return [], []

    long_indices = _select_uniform_indices(total, num_long)
    long_items = [dict(items[i]) for i in sorted(long_indices)]
    short_items = [dict(items[i]) for i in range(total) if i not in long_indices]

    beg_long_indices = _select_uniform_indices(len(long_items), beg_long)
    for i, it in enumerate(long_items):
        it["detail"] = "long"
        it["level"] = "beginner" if i in beg_long_indices else "club"

    beg_short_indices = _select_uniform_indices(len(short_items), beg_short)
    for i, it in enumerate(short_items):
        it["detail"] = "short"
        it["level"] = "beginner" if i in beg_short_indices else "club"

    return short_items, long_items


def verify_zero_contamination(
    records: list[dict[str, Any]],
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
) -> None:
    """Verify zero overlap between test set and training splits.

    If any game ID or normalized FEN matches an SFT or RL split, raise ValueError.
    """
    excluded_games, excluded_fens = load_excluded_keys(
        split_dir,
        split_names=("sft_train", "sft_val", "rl_prompts"),
    )

    contaminated_games: list[str] = []
    contaminated_fens: list[str] = []

    for item in records:
        gid = str(item.get("game_id", ""))
        fen = str(item.get("fen", ""))
        norm_fen = normalize_fen(fen)

        if gid and gid in excluded_games:
            contaminated_games.append(gid)
        if norm_fen and norm_fen in excluded_fens:
            contaminated_fens.append(norm_fen)

    if contaminated_games or contaminated_fens:
        msg = (
            f"The builder detected contamination: {len(contaminated_games)} game IDs "
            f"and {len(contaminated_fens)} FENs overlap with training splits."
        )
        raise ValueError(msg)


def build_testset(
    eval_pool_path: Path | str = DEFAULT_EVAL_POOL_PATH,
    raw_games_path: Path | str = DEFAULT_RAW_GAMES_PATH,
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Build the complete frozen test set from the evaluation pool.

    Enrich evaluation pool records with Elo ratings and game phases.
    Select positions deterministically according to target distributions.
    Rebuild user messages and chat prompts with prompt formatting functions.
    Verify prompt parsing and zero contamination against training data.
    Return 300 formatted test records.
    """
    pool_file = Path(eval_pool_path)
    if not pool_file.exists():
        raise FileNotFoundError(f"Evaluation pool file not found: {pool_file}")

    ratings = load_game_ratings(raw_games_path)

    raw_pool: list[dict[str, Any]] = []
    with open(pool_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_pool.append(json.loads(line))

    # Enrich pool items with Elo ratings and phase
    enriched_pool: list[dict[str, Any]] = []
    for item in raw_pool:
        gid = str(item.get("game_id", ""))
        game_ratings = ratings.get(gid, {})
        w_elo = game_ratings.get("white_elo")
        b_elo = game_ratings.get("black_elo")

        if w_elo is not None and b_elo is not None:
            avg_elo = (float(w_elo) + float(b_elo)) / 2.0
        else:
            continue

        if not (800.0 <= avg_elo <= 2000.0):
            continue

        enriched = dict(item)
        enriched["white_elo"] = w_elo
        enriched["black_elo"] = b_elo
        enriched["avg_elo"] = avg_elo
        enriched["phase"] = extract_phase(str(item["fen"]))
        enriched_pool.append(enriched)

    # Separate candidates into 3 categories
    bad_candidates = [
        it for it in enriched_pool
        if it.get("task") == "explain_move" and it.get("verdict") in BAD_VERDICTS
    ]
    good_candidates = [
        it for it in enriched_pool
        if it.get("task") == "explain_move" and it.get("verdict") in GOOD_VERDICTS
    ]
    pos_candidates = [
        it for it in enriched_pool
        if it.get("task") == "explain_position"
    ]

    used_games: set[str] = set()
    used_fens: set[str] = set()

    selected_bad = select_bucket_items(
        bad_candidates,
        TARGET_TOTAL_BAD,
        seed=seed,
        used_games=used_games,
        used_fens=used_fens,
    )
    selected_good = select_bucket_items(
        good_candidates,
        TARGET_TOTAL_GOOD,
        seed=seed,
        used_games=used_games,
        used_fens=used_fens,
    )
    selected_pos = select_bucket_items(
        pos_candidates,
        TARGET_TOTAL_POS,
        seed=seed,
        used_games=used_games,
        used_fens=used_fens,
    )

    if len(selected_bad) != TARGET_TOTAL_BAD:
        raise ValueError(f"Expected {TARGET_TOTAL_BAD} bad move items, got {len(selected_bad)}")
    if len(selected_good) != TARGET_TOTAL_GOOD:
        raise ValueError(f"Expected {TARGET_TOTAL_GOOD} good move items, got {len(selected_good)}")
    if len(selected_pos) != TARGET_TOTAL_POS:
        raise ValueError(f"Expected {TARGET_TOTAL_POS} position items, got {len(selected_pos)}")

    # Partition buckets into short and long, beginner and club
    bad_short, bad_long = partition_and_label_bucket(
        selected_bad,
        num_long=TARGET_LONG_BAD,
        beg_long=11,
        beg_short=60,
    )
    good_short, good_long = partition_and_label_bucket(
        selected_good,
        num_long=TARGET_LONG_GOOD,
        beg_long=8,
        beg_short=45,
    )
    pos_short, pos_long = partition_and_label_bucket(
        selected_pos,
        num_long=TARGET_LONG_POS,
        beg_long=4,
        beg_short=22,
    )

    # Interleave items across buckets for uniform distribution
    bucket_lists = [bad_short, good_short, pos_short, bad_long, good_long, pos_long]
    max_len = max(len(bucket) for bucket in bucket_lists)
    interleaved: list[dict[str, Any]] = []
    for step_idx in range(max_len):
        for bucket in bucket_lists:
            if step_idx < len(bucket):
                interleaved.append(bucket[step_idx])

    if len(interleaved) != target_count:
        raise ValueError(f"Expected {target_count} total items, got {len(interleaved)}")

    final_records: list[dict[str, Any]] = []
    for idx, item in enumerate(interleaved):
        test_id = f"test_{idx + 1:03d}"
        pool_id = str(item.get("id", ""))
        fen = str(item["fen"])
        task = str(item["task"])
        level = str(item["level"])
        detail = str(item["detail"])
        move_uci = item.get("move_uci")
        verdict = item.get("verdict")
        eval_before = item.get("eval_before", {})
        eval_after = item.get("eval_after")

        board = chess.Board(fen)
        if task == "explain_move":
            if not move_uci:
                raise ValueError(f"Missing move_uci for {test_id}")
            move_obj = chess.Move.from_uci(str(move_uci))
            move_san: str | None = board.san(move_obj)
        else:
            move_san = None

        prompt_record = format_prompt_record(
            item_id=test_id,
            game_id=str(item["game_id"]),
            fen=fen,
            task=task,
            level=level,
            detail=detail,
            eval_before=eval_before,
            eval_after=eval_after,
            move_uci=move_uci,
            verdict=verdict,
            source="lichess_games",
            redistributable=True,
        )

        record: dict[str, Any] = {
            "id": test_id,
            "pool_id": pool_id,
            "game_id": str(item["game_id"]),
            "fen": fen,
            "task": task,
            "level": level,
            "detail": detail,
            "move_uci": move_uci,
            "move_san": move_san,
            "verdict": verdict,
            "phase": str(item["phase"]),
            "white_elo": item["white_elo"],
            "black_elo": item["black_elo"],
            "avg_elo": item["avg_elo"],
            "facts": prompt_record["facts"],
            "user_message": prompt_record["user_message"],
            "chat_prompt": prompt_record["chat_prompt"],
            "eval_before": eval_before,
            "eval_after": eval_after,
            "source": "lichess_games",
            "redistributable": True,
        }

        # Quality check: verify parser processes user message and chat prompt
        parsed_user = parse_prompt(record["user_message"])
        parsed_chat = parse_prompt(record["chat_prompt"])
        if parsed_user.task != task or parsed_chat.task != task:
            raise ValueError(f"Prompt parser task mismatch in {test_id}")
        if parsed_user.level != level or parsed_chat.level != level:
            raise ValueError(f"Prompt parser level mismatch in {test_id}")
        if parsed_user.detail != detail or parsed_chat.detail != detail:
            raise ValueError(f"Prompt parser detail mismatch in {test_id}")

        final_records.append(record)

    verify_zero_contamination(final_records, split_dir=split_dir)
    return final_records


def save_testset(records: list[dict[str, Any]], output_path: Path | str = DEFAULT_OUTPUT_PATH) -> None:
    """Save test set records to a JSONL file.

    If parent directories do not exist, create them.
    Write each record as a JSON line.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    LOGGER.info("Saved %d records to %s", len(records), out_file)


def main() -> None:
    """Run the test set builder from the command line."""
    parser = argparse.ArgumentParser(description="Build frozen test set for Second evaluation.")
    parser.add_argument("--eval-pool", type=str, default=str(DEFAULT_EVAL_POOL_PATH), help="Path to eval pool JSONL.")
    parser.add_argument("--raw-games", type=str, default=str(DEFAULT_RAW_GAMES_PATH), help="Path to raw games JSONL.")
    parser.add_argument("--split-dir", type=str, default=str(DEFAULT_SPLIT_DIR), help="Directory of SFT splits.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic hashing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    records = build_testset(
        eval_pool_path=args.eval_pool,
        raw_games_path=args.raw_games,
        split_dir=args.split_dir,
        seed=args.seed,
    )
    save_testset(records, output_path=args.output)


if __name__ == "__main__":
    main()
