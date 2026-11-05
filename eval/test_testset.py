"""Unit and integration tests for Phase 4.1 test set and builder.

This module validates eval/testset-1.0.jsonl and eval/build_testset.py.
It verifies dataset size, task distributions, player levels, and detail lengths.
It ensures zero data contamination against training and RL splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import pytest

from core.python.prompt import format_qwen_chat_prompt
from data.pipeline.rl_prompts import load_excluded_keys
from data.pipeline.splits import normalize_fen
from eval.build_testset import (
    BAD_VERDICTS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RAW_GAMES_PATH,
    DEFAULT_SPLIT_DIR,
    GOOD_VERDICTS,
    extract_phase,
    load_game_ratings,
    partition_and_label_bucket,
    select_bucket_items,
    verify_zero_contamination,
)
from rewards.api import parse_prompt

DEFAULT_OUTPUT_PATH = Path("eval/testset-1.0.jsonl")
DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_RAW_GAMES_PATH = Path("data/raw/lichess_games.jsonl")
TESTSET_PATH = DEFAULT_OUTPUT_PATH


@pytest.fixture(scope="module")
def testset_records() -> list[dict[str, Any]]:
    """Load records from the frozen test set file.

    If the test set file does not exist, fail the test immediately.
    Return a list of parsed record dictionaries.
    """
    assert TESTSET_PATH.exists(), f"Frozen testset file not found at {TESTSET_PATH}"
    records: list[dict[str, Any]] = []
    with open(TESTSET_PATH, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _has_local_splits() -> bool:
    """Return True if local data/processed split files exist."""
    return (DEFAULT_SPLIT_DIR / "sft_train.jsonl").exists()


def _has_local_raw_games() -> bool:
    """Return True if local data/raw/lichess_games.jsonl exists."""
    return DEFAULT_RAW_GAMES_PATH.exists()


def test_testset_item_count(testset_records: list[dict[str, Any]]) -> None:
    """Verify that the test set contains exactly 300 items."""
    assert len(testset_records) == 300


def test_testset_mix_distribution(testset_records: list[dict[str, Any]]) -> None:
    """Verify the exact distribution across tasks, verdicts, and detail modes.

    The frozen test set must contain:
    - 120 short bad moves (Inaccuracy, Mistake, or Blunder).
    - 90 short good moves (Good, Best, or Only move).
    - 45 short position explanations.
    - 45 long detail items across categories.
    """
    short_bad = [
        r for r in testset_records
        if r["task"] == "explain_move" and r["detail"] == "short" and r["verdict"] in BAD_VERDICTS
    ]
    short_good = [
        r for r in testset_records
        if r["task"] == "explain_move" and r["detail"] == "short" and r["verdict"] in GOOD_VERDICTS
    ]
    short_pos = [
        r for r in testset_records
        if r["task"] == "explain_position" and r["detail"] == "short" and r["verdict"] is None
    ]
    long_items = [r for r in testset_records if r["detail"] == "long"]

    assert len(short_bad) == 120
    assert len(short_good) == 90
    assert len(short_pos) == 45
    assert len(long_items) == 45

    long_bad = [
        r for r in long_items
        if r["task"] == "explain_move" and r["verdict"] in BAD_VERDICTS
    ]
    long_good = [
        r for r in long_items
        if r["task"] == "explain_move" and r["verdict"] in GOOD_VERDICTS
    ]
    long_pos = [
        r for r in long_items
        if r["task"] == "explain_position" and r["verdict"] is None
    ]

    assert len(long_bad) == 21
    assert len(long_good) == 16
    assert len(long_pos) == 8

    # Verify overall totals across short and long
    total_bad = len(short_bad) + len(long_bad)
    total_good = len(short_good) + len(long_good)
    total_pos = len(short_pos) + len(long_pos)

    assert total_bad == 141
    assert total_good == 106
    assert total_pos == 53
    assert total_bad + total_good + total_pos == 300


def test_testset_levels_balance(testset_records: list[dict[str, Any]]) -> None:
    """Verify player level balance across the test set and within buckets.

    The test set must contain exactly 150 beginner and 150 club items.
    Each of the four buckets must maintain balanced levels.
    """
    beg_total = sum(1 for r in testset_records if r["level"] == "beginner")
    club_total = sum(1 for r in testset_records if r["level"] == "club")

    assert beg_total == 150
    assert club_total == 150

    short_bad = [
        r for r in testset_records
        if r["task"] == "explain_move" and r["detail"] == "short" and r["verdict"] in BAD_VERDICTS
    ]
    short_good = [
        r for r in testset_records
        if r["task"] == "explain_move" and r["detail"] == "short" and r["verdict"] in GOOD_VERDICTS
    ]
    short_pos = [
        r for r in testset_records
        if r["task"] == "explain_position" and r["detail"] == "short"
    ]
    long_items = [r for r in testset_records if r["detail"] == "long"]

    assert sum(1 for r in short_bad if r["level"] == "beginner") == 60
    assert sum(1 for r in short_bad if r["level"] == "club") == 60

    assert sum(1 for r in short_good if r["level"] == "beginner") == 45
    assert sum(1 for r in short_good if r["level"] == "club") == 45

    assert sum(1 for r in short_pos if r["level"] == "beginner") == 22
    assert sum(1 for r in short_pos if r["level"] == "club") == 23

    assert sum(1 for r in long_items if r["level"] == "beginner") == 23
    assert sum(1 for r in long_items if r["level"] == "club") == 22


def test_testset_phase_representation(testset_records: list[dict[str, Any]]) -> None:
    """Verify representation for opening, middlegame, and endgame phases.

    Each phase must have a substantial presence in the test set.
    """
    phase_counts: dict[str, int] = {}
    for r in testset_records:
        ph = r["phase"]
        phase_counts[ph] = phase_counts.get(ph, 0) + 1

    assert "opening" in phase_counts
    assert "middlegame" in phase_counts
    assert "endgame" in phase_counts

    assert phase_counts["opening"] >= 50
    assert phase_counts["middlegame"] >= 50
    assert phase_counts["endgame"] >= 50


def test_testset_elo_ratings_range(testset_records: list[dict[str, Any]]) -> None:
    """Verify that every record has valid Elo ratings within 800 to 2000."""
    for r in testset_records:
        assert "white_elo" in r and r["white_elo"] is not None
        assert "black_elo" in r and r["black_elo"] is not None
        assert "avg_elo" in r and r["avg_elo"] is not None

        w_elo = r["white_elo"]
        b_elo = r["black_elo"]
        avg_elo = r["avg_elo"]

        assert isinstance(w_elo, int)
        assert isinstance(b_elo, int)
        assert isinstance(avg_elo, (int, float))

        expected_avg = (w_elo + b_elo) / 2.0
        assert abs(avg_elo - expected_avg) < 1e-6
        assert 800.0 <= avg_elo <= 2000.0


def test_testset_prompt_formatting_and_parsing(testset_records: list[dict[str, Any]]) -> None:
    """Verify prompt formatting and ensure rewards.parse_prompt parses all items."""
    for r in testset_records:
        user_msg = r["user_message"]
        chat_prompt = r["chat_prompt"]

        # Check Qwen chat prompt template structure
        expected_chat = format_qwen_chat_prompt(user_msg)
        assert chat_prompt == expected_chat

        lines = user_msg.splitlines()
        assert lines[0] == f"task: {r['task']}"
        assert lines[1] == f"level: {r['level']}"
        assert lines[2] == f"detail: {r['detail']}"

        # Parse user message
        parsed_user = parse_prompt(user_msg)
        assert parsed_user.task == r["task"]
        assert parsed_user.level == r["level"]
        assert parsed_user.detail == r["detail"]
        assert parsed_user.fen == r["fen"]

        # Parse chat prompt
        parsed_chat = parse_prompt(chat_prompt)
        assert parsed_chat.task == r["task"]
        assert parsed_chat.level == r["level"]
        assert parsed_chat.detail == r["detail"]
        assert parsed_chat.fen == r["fen"]

        # Check task-specific fields
        if r["task"] == "explain_move":
            assert parsed_user.move_played_san == r["move_san"]
            assert parsed_user.verdict == r["verdict"]
            assert parsed_chat.move_played_san == r["move_san"]
            assert parsed_chat.verdict == r["verdict"]
        else:
            assert parsed_user.move_played_san is None
            assert parsed_user.verdict is None
            assert parsed_chat.move_played_san is None
            assert parsed_chat.verdict is None


def test_testset_zero_contamination(
    testset_records: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Verify zero overlap with SFT train, SFT val, and RL prompt sets.

    When local split files exist, verify against the full local splits.
    Always verify that verify_zero_contamination detects overlapping games and FENs.
    """
    if _has_local_splits():
        verify_zero_contamination(testset_records, split_dir=DEFAULT_SPLIT_DIR)

        excluded_games, excluded_fens = load_excluded_keys(
            DEFAULT_SPLIT_DIR,
            split_names=("sft_train", "sft_val", "rl_prompts"),
        )

        test_games = {str(r["game_id"]) for r in testset_records if r.get("game_id")}
        test_fens = {normalize_fen(str(r["fen"])) for r in testset_records if r.get("fen")}

        assert len(test_games & excluded_games) == 0
        assert len(test_fens & excluded_fens) == 0

    # Also test verify_zero_contamination with a synthetic split directory
    disjoint_split = tmp_path / "sft_train.jsonl"
    disjoint_split.write_text(
        json.dumps({"game_id": "non_overlapping_game", "fen": "8/8/8/8/8/8/4K3/4k3 w - - 0 1"}) + "\n",
        encoding="utf-8",
    )
    verify_zero_contamination(testset_records[:5], split_dir=tmp_path)

    overlapping_split = tmp_path / "rl_prompts.jsonl"
    overlapping_split.write_text(
        json.dumps({"game_id": testset_records[0]["game_id"], "fen": testset_records[0]["fen"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="detected contamination"):
        verify_zero_contamination(testset_records[:5], split_dir=tmp_path)


def test_testset_schema_and_integrity(testset_records: list[dict[str, Any]]) -> None:
    """Verify record schema keys, identifiers, and move validity."""
    expected_keys = {
        "id", "pool_id", "game_id", "fen", "task", "level", "detail",
        "move_uci", "move_san", "verdict", "phase", "white_elo", "black_elo",
        "avg_elo", "facts", "user_message", "chat_prompt", "eval_before",
        "eval_after", "source", "redistributable",
    }

    seen_ids: set[str] = set()
    seen_fens: set[str] = set()

    for idx, r in enumerate(testset_records):
        assert set(r.keys()) == expected_keys
        expected_id = f"test_{idx + 1:03d}"
        assert r["id"] == expected_id
        seen_ids.add(r["id"])

        assert r["pool_id"].startswith("eval_")
        assert r["source"] == "lichess_games"
        assert r["redistributable"] is True

        norm_fen = normalize_fen(r["fen"])
        assert norm_fen not in seen_fens, f"Duplicate position found: {r['fen']}"
        seen_fens.add(norm_fen)

        board = chess.Board(r["fen"])
        if r["task"] == "explain_move":
            move_obj = chess.Move.from_uci(r["move_uci"])
            assert move_obj in board.legal_moves
            assert board.san(move_obj) == r["move_san"]
        else:
            assert r["move_uci"] is None
            assert r["move_san"] is None
            assert r["verdict"] is None

    assert len(seen_ids) == 300
    assert len(seen_fens) == 300


def test_testset_game_diversity(testset_records: list[dict[str, Any]]) -> None:
    """Verify that test set positions come from many distinct games."""
    unique_games = {str(r["game_id"]) for r in testset_records}
    assert len(unique_games) >= 250


def test_extract_phase_unit() -> None:
    """Verify game phase extraction on known chess positions."""
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert extract_phase(start_fen) == "opening"

    end_fen = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
    assert extract_phase(end_fen) == "endgame"


def test_load_game_ratings_unit(tmp_path: Path) -> None:
    """Verify loading game ratings from raw Lichess game records."""
    if _has_local_raw_games():
        ratings = load_game_ratings(DEFAULT_RAW_GAMES_PATH)
        assert len(ratings) > 1000
        first_gid = next(iter(ratings))
        assert "white_elo" in ratings[first_gid]
        assert "black_elo" in ratings[first_gid]

    sample_raw = tmp_path / "games.jsonl"
    sample_raw.write_text(
        json.dumps({"game_id": "g_test_1", "white_elo": 1450, "black_elo": 1550}) + "\n",
        encoding="utf-8",
    )
    parsed = load_game_ratings(sample_raw)
    assert parsed["g_test_1"] == {"white_elo": 1450, "black_elo": 1550}


def test_partition_and_label_bucket_unit() -> None:
    """Verify the partition and labeling function on mock records."""
    mock_items = [{"id": i, "phase": "middlegame"} for i in range(10)]
    short_items, long_items = partition_and_label_bucket(
        mock_items,
        num_long=2,
        beg_long=1,
        beg_short=4,
    )
    assert len(short_items) == 8
    assert len(long_items) == 2

    assert sum(1 for it in long_items if it["level"] == "beginner") == 1
    assert sum(1 for it in long_items if it["level"] == "club") == 1

    assert sum(1 for it in short_items if it["level"] == "beginner") == 4
    assert sum(1 for it in short_items if it["level"] == "club") == 4


def test_select_bucket_items_unit() -> None:
    """Verify deterministic selection cycling across phases."""
    candidates = [
        {"game_id": "g1", "fen": "f1", "phase": "opening"},
        {"game_id": "g2", "fen": "f2", "phase": "middlegame"},
        {"game_id": "g3", "fen": "f3", "phase": "endgame"},
        {"game_id": "g4", "fen": "f4", "phase": "opening"},
    ]
    used_games: set[str] = set()
    used_fens: set[str] = set()
    selected = select_bucket_items(
        candidates,
        target_count=3,
        seed=42,
        used_games=used_games,
        used_fens=used_fens,
    )
    assert len(selected) == 3
    assert len(used_games) == 3
