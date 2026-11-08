"""Stratified development set builder module.

This module builds the frozen 150-item dev set for Phase 6.5.
It samples records deterministically from uncontaminated test pools.
It validates player levels, detail lengths, verdicts, and phases.
It guarantees zero game and position contamination against training and evaluation sets.
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

from core.python.facts import compute_phase_fact, extract_facts
from core.python.prompt import build_user_message, format_qwen_chat_prompt
from data.pipeline.rl_prompts import format_prompt_record, load_excluded_keys
from data.pipeline.splits import hash_game_id, normalize_fen
from eval.build_testset import (
    BAD_VERDICTS,
    GOOD_VERDICTS,
    _select_uniform_indices,
    extract_phase,
    select_bucket_items,
)
from eval.concept_tags import classify_concept_tier

LOGGER = logging.getLogger(__name__)

DEFAULT_SFT_TEST_POOL = Path("data/processed/sft_test_pool.jsonl")
DEFAULT_EVAL_POOL = Path("data/processed/eval_pool.jsonl")
DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_TESTSET_PATH = Path("eval/testset-1.0.jsonl")
DEFAULT_OUTPUT_PATH = Path("eval/devset-1.0.jsonl")

DEFAULT_TARGET_COUNT = 150
DEFAULT_SEED = 42

DEV_SHORT_BAD = 60
DEV_SHORT_GOOD = 45
DEV_SHORT_POS = 22

DEV_LONG_BAD = 11
DEV_LONG_GOOD = 8
DEV_LONG_POS = 4

DEV_TOTAL_BAD = DEV_SHORT_BAD + DEV_LONG_BAD    # 71
DEV_TOTAL_GOOD = DEV_SHORT_GOOD + DEV_LONG_GOOD  # 53
DEV_TOTAL_POS = DEV_SHORT_POS + DEV_LONG_POS    # 26


def verify_devset_zero_contamination(
    records: list[dict[str, Any]],
    testset_path: Path | str = DEFAULT_TESTSET_PATH,
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
) -> None:
    """Verify zero overlap with testset and training splits.

    If any game ID or normalized FEN matches an existing split, raise ValueError.
    """
    excluded_games, excluded_fens = load_excluded_keys(
        split_dir,
        split_names=("sft_train", "sft_val", "rl_prompts"),
    )

    t_path = Path(testset_path)
    if t_path.exists():
        with open(t_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    entry = json.loads(line_str)
                    gid = str(entry.get("game_id", ""))
                    fen = str(entry.get("fen", ""))
                    if gid:
                        excluded_games.add(gid)
                    if fen:
                        excluded_fens.add(normalize_fen(fen))

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
            f"Contamination detected in dev set: {len(contaminated_games)} game IDs "
            f"and {len(contaminated_fens)} FENs overlap with excluded splits."
        )
        raise ValueError(msg)


def partition_dev_bucket(
    items: list[dict[str, Any]],
    num_long: int,
    beg_long: int,
    beg_short: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition items into short and long subsets and assign levels.

    Return a tuple of short items list and long items list.
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


def build_devset(
    sft_test_pool_path: Path | str = DEFAULT_SFT_TEST_POOL,
    eval_pool_path: Path | str = DEFAULT_EVAL_POOL,
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
    testset_path: Path | str = DEFAULT_TESTSET_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Build the 150-item stratified development set.

    Select items across bad moves, good moves, and position explanations.
    Ensure exact level balance and zero data leakage.
    Return the list of created records.
    """
    # 1. Gather all excluded keys
    excluded_games, excluded_fens = load_excluded_keys(
        split_dir,
        split_names=("sft_train", "sft_val", "rl_prompts"),
    )
    t_path = Path(testset_path)
    if t_path.exists():
        with open(t_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    t_rec = json.loads(s)
                    if t_rec.get("game_id"):
                        excluded_games.add(str(t_rec["game_id"]))
                    if t_rec.get("fen"):
                        excluded_fens.add(normalize_fen(str(t_rec["fen"])))

    # 2. Load candidate pools
    all_raw_candidates: list[dict[str, Any]] = []
    for p_path in (Path(sft_test_pool_path), Path(eval_pool_path)):
        if not p_path.exists():
            continue
        with open(p_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    all_raw_candidates.append(json.loads(s))

    # Filter candidates by exclusions
    clean_candidates: list[dict[str, Any]] = []
    seen_cand_fens: set[str] = set()
    for cand in all_raw_candidates:
        gid = str(cand.get("game_id", ""))
        fen = str(cand.get("fen", ""))
        n_fen = normalize_fen(fen)
        if gid in excluded_games or n_fen in excluded_fens:
            continue
        if n_fen in seen_cand_fens:
            continue
        seen_cand_fens.add(n_fen)

        enriched = dict(cand)
        enriched["phase"] = extract_phase(fen)
        clean_candidates.append(enriched)

    # Separate candidates into buckets
    bad_candidates = [
        it for it in clean_candidates
        if it.get("task") == "explain_move" and it.get("verdict") in BAD_VERDICTS
    ]
    good_candidates = [
        it for it in clean_candidates
        if it.get("task") == "explain_move" and it.get("verdict") in GOOD_VERDICTS
    ]

    # For explain_position, derive candidate positions from clean candidate pool
    pos_candidates: list[dict[str, Any]] = []
    pos_used_fens: set[str] = set()
    for it in clean_candidates:
        # Avoid using same FEN as selected move items later
        n_fen = normalize_fen(str(it["fen"]))
        if n_fen not in pos_used_fens:
            pos_item = dict(it)
            pos_item["task"] = "explain_position"
            pos_item["move_uci"] = None
            pos_item["move_san"] = None
            pos_item["verdict"] = None
            pos_item["eval_after"] = None
            pos_candidates.append(pos_item)
            pos_used_fens.add(n_fen)

    used_games: set[str] = set()
    used_fens: set[str] = set()

    selected_bad = select_bucket_items(
        bad_candidates,
        DEV_TOTAL_BAD,
        seed=seed,
        used_games=used_games,
        used_fens=used_fens,
    )
    selected_good = select_bucket_items(
        good_candidates,
        DEV_TOTAL_GOOD,
        seed=seed,
        used_games=used_games,
        used_fens=used_fens,
    )

    # Exclude games and fens used by bad and good for position items
    pos_available = [
        it for it in pos_candidates
        if normalize_fen(str(it["fen"])) not in used_fens
    ]
    selected_pos = select_bucket_items(
        pos_available,
        DEV_TOTAL_POS,
        seed=seed,
        used_games=used_games,
        used_fens=used_fens,
    )

    if len(selected_bad) != DEV_TOTAL_BAD:
        raise ValueError(f"Expected {DEV_TOTAL_BAD} bad items, got {len(selected_bad)}")
    if len(selected_good) != DEV_TOTAL_GOOD:
        raise ValueError(f"Expected {DEV_TOTAL_GOOD} good items, got {len(selected_good)}")
    if len(selected_pos) != DEV_TOTAL_POS:
        raise ValueError(f"Expected {DEV_TOTAL_POS} pos items, got {len(selected_pos)}")

    # Partition and assign beginner / club levels
    # Total beginner = 30 + 6 + 22 + 4 + 11 + 2 = 75
    # Total club = 30 + 5 + 23 + 4 + 11 + 2 = 75
    bad_short, bad_long = partition_dev_bucket(
        selected_bad,
        num_long=DEV_LONG_BAD,
        beg_long=6,
        beg_short=30,
    )
    good_short, good_long = partition_dev_bucket(
        selected_good,
        num_long=DEV_LONG_GOOD,
        beg_long=4,
        beg_short=22,
    )
    pos_short, pos_long = partition_dev_bucket(
        selected_pos,
        num_long=DEV_LONG_POS,
        beg_long=2,
        beg_short=11,
    )

    bucket_lists = [bad_short, good_short, pos_short, bad_long, good_long, pos_long]
    max_len = max(len(b) for b in bucket_lists)
    interleaved: list[dict[str, Any]] = []
    for step_idx in range(max_len):
        for b in bucket_lists:
            if step_idx < len(b):
                interleaved.append(b[step_idx])

    if len(interleaved) != DEFAULT_TARGET_COUNT:
        raise ValueError(f"Expected {DEFAULT_TARGET_COUNT} items, got {len(interleaved)}")

    final_records: list[dict[str, Any]] = []
    for idx, item in enumerate(interleaved):
        dev_id = f"dev_{idx + 1:03d}"
        pool_id = str(item.get("id", item.get("pool_id", f"pool_{idx:04d}")))
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
                raise ValueError(f"Missing move_uci for {dev_id}")
            move_obj = chess.Move.from_uci(str(move_uci))
            move_san: str | None = board.san(move_obj)
        else:
            move_san = None

        user_msg = build_user_message(
            task=task,
            level=level,
            detail=detail,
            fen=fen,
            eval_before=eval_before,
            eval_after=eval_after,
            move_uci=str(move_uci) if move_uci else None,
        )
        chat_prompt = format_qwen_chat_prompt(user_msg)
        computed_facts = extract_facts(
            task=task,
            fen=fen,
            eval_before=eval_before,
            eval_after=eval_after,
            move_uci=str(move_uci) if move_uci else None,
        )

        record: dict[str, Any] = {
            "id": dev_id,
            "pool_id": pool_id,
            "game_id": str(item["game_id"]),
            "fen": fen,
            "task": task,
            "level": level,
            "detail": detail,
            "move_uci": str(move_uci) if move_uci else None,
            "move_san": move_san,
            "verdict": verdict,
            "phase": item["phase"],
            "white_elo": item.get("white_elo"),
            "black_elo": item.get("black_elo"),
            "avg_elo": item.get("avg_elo"),
            "facts": computed_facts,
            "user_message": user_msg,
            "chat_prompt": chat_prompt,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "source": item.get("source", "lichess_studies"),
            "redistributable": item.get("redistributable", True),
        }
        record["concept_tier"] = classify_concept_tier(record)
        final_records.append(record)

    verify_devset_zero_contamination(final_records, testset_path=testset_path, split_dir=split_dir)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOGGER.info("Successfully built dev set at %s (%d items)", out_p, len(final_records))
    return final_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_devset()
