"""Tests for Phase 11 v1.1 evaluation datasets and builder.

This module validates eval/devset-1.1.jsonl, eval/testset-1.1.jsonl, and build_eval_v1_1.
It verifies item counts, level balance, v1.1 fact freshness, zero contamination, and rating ingestion.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import pytest

from core.python.facts import extract_facts
from data.pipeline.splits import normalize_fen
from eval.build_eval_v1_1 import (
    ALL_TRAINING_SPLITS,
    DEFAULT_DEVSET_1_1,
    DEFAULT_SPLIT_DIR,
    DEFAULT_TESTSET_1_1,
    _resolve_move_uci,
    build_eval_v1_1,
    parse_exported_ratings_files,
    verify_v1_1_zero_contamination,
)
from eval.concept_tags import CONCEPT_TIERS
from rewards.checks import check_piece_on_square


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from path."""
    assert path.exists(), f"Missing eval file: {path}"
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


@pytest.fixture(scope="module")
def devset_v1_1() -> list[dict[str, Any]]:
    """Load devset-1.1.jsonl records."""
    return _load_jsonl(DEFAULT_DEVSET_1_1)


@pytest.fixture(scope="module")
def testset_v1_1() -> list[dict[str, Any]]:
    """Load testset-1.1.jsonl records."""
    return _load_jsonl(DEFAULT_TESTSET_1_1)


def test_eval_v1_1_counts_and_level_balance(
    devset_v1_1: list[dict[str, Any]],
    testset_v1_1: list[dict[str, Any]],
) -> None:
    """Verify total counts and exact 50/50 beginner-club balance in v1.1 sets."""
    assert len(devset_v1_1) == 160
    assert len(testset_v1_1) == 320

    assert sum(1 for r in devset_v1_1 if r["level"] == "beginner") == 80
    assert sum(1 for r in devset_v1_1 if r["level"] == "club") == 80
    assert sum(1 for r in testset_v1_1 if r["level"] == "beginner") == 160
    assert sum(1 for r in testset_v1_1 if r["level"] == "club") == 160


def test_eval_v1_1_concept_tiers_and_regression_cases(
    devset_v1_1: list[dict[str, Any]],
    testset_v1_1: list[dict[str, Any]],
) -> None:
    """Verify valid concept tiers and presence of the 6... exd4 positional inaccuracy case."""
    for rec in devset_v1_1 + testset_v1_1:
        assert rec.get("concept_tier") in CONCEPT_TIERS

    exd4_items = [
        r for r in testset_v1_1 if r.get("regression_tag") == "bogo_6_exd4_vs_e4"
    ]
    assert len(exd4_items) == 1
    item = exd4_items[0]
    assert item["move_san"] == "exd4"
    assert item["verdict"] == "Inaccuracy"
    assert item["concept_tier"] == "tier_4_positional"
    assert any("centralizes the knight" in f for f in item["facts"])
    assert any("gains space" in f for f in item["facts"])

    bad_verdicts = {"Inaccuracy", "Mistake", "Blunder"}
    curated_dev = [r for r in devset_v1_1 if r.get("source") == "curated_v1_1"]
    curated_test = [r for r in testset_v1_1 if r.get("source") == "curated_v1_1"]
    assert len(curated_dev) == 10
    assert len(curated_test) == 20
    assert sum(1 for r in curated_dev if r.get("verdict") in bad_verdicts) >= 4
    assert sum(1 for r in curated_test if r.get("verdict") in bad_verdicts) >= 5

    # Specific positional cases must classify as tier_4_positional and curated positions stay balanced
    dev_by_id = {r["id"]: r for r in devset_v1_1}
    test_by_id = {r["id"]: r for r in testset_v1_1}
    assert dev_by_id["dev_152"]["concept_tier"] == "tier_4_positional"
    for tid in ("test_301", "test_311", "test_320"):
        assert test_by_id[tid]["concept_tier"] == "tier_4_positional"

    assert test_by_id["test_091"]["concept_tier"] == "tier_1_mechanics"
    assert test_by_id["test_199"]["concept_tier"] == "tier_3_calculation"
    assert test_by_id["test_295"]["concept_tier"] == "tier_3_calculation"

    dev_155_facts = " ".join(dev_by_id["dev_155"]["facts"])
    assert "wins a pawn." in dev_155_facts
    assert "keeps a pawn advantage" not in dev_155_facts
    assert "White leads in piece development." in test_by_id["test_320"]["facts"]

    piece_vals = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    for balanced_rec in (dev_by_id["dev_159"], test_by_id["test_318"]):
        b = chess.Board(balanced_rec["fen"])
        w_m = sum(piece_vals[p.piece_type] for p in b.piece_map().values() if p.color == chess.WHITE)
        b_m = sum(piece_vals[p.piece_type] for p in b.piece_map().values() if p.color == chess.BLACK)
        assert w_m == b_m, f"Expected balanced material on {balanced_rec['id']}, got {w_m}:{b_m}"


def test_eval_v1_1_zero_contamination(
    devset_v1_1: list[dict[str, Any]],
    testset_v1_1: list[dict[str, Any]],
) -> None:
    """Verify zero game ID or FEN overlap across all 8 splits and between dev and test."""
    assert "sft_v2_train" in ALL_TRAINING_SPLITS
    assert "teacher_prompts" in ALL_TRAINING_SPLITS
    verify_v1_1_zero_contamination(
        dev_records=devset_v1_1,
        test_records=testset_v1_1,
        split_dir=DEFAULT_SPLIT_DIR,
    )

    sft_v2_path = DEFAULT_SPLIT_DIR / "sft_v2_train.jsonl"
    if sft_v2_path.exists():
        first_rec = json.loads(sft_v2_path.read_text(encoding="utf-8").splitlines()[0])
        with pytest.raises(ValueError):
            verify_v1_1_zero_contamination(
                dev_records=[{"id": "dup", "game_id": "x", "fen": first_rec["fen"]}],
                test_records=[],
                split_dir=DEFAULT_SPLIT_DIR,
            )


def test_eval_v1_1_prompt_freshness(
    devset_v1_1: list[dict[str, Any]],
    testset_v1_1: list[dict[str, Any]],
) -> None:
    """Verify every v1.1 item facts list matches current v1.1 extract_facts and passes Rule 9."""
    for rec in devset_v1_1 + testset_v1_1:
        fresh_facts = extract_facts(
            task=rec["task"],
            fen=rec["fen"],
            eval_before=rec["eval_before"],
            eval_after=rec.get("eval_after"),
            move_uci=rec.get("move_uci"),
        )
        assert rec["facts"] == fresh_facts
        assert "Facts:" in rec["user_message"]
        assert rec["chat_prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        rule9 = check_piece_on_square(" ".join(fresh_facts), rec["fen"], rec.get("move_uci"))
        assert rule9.passed, f"Rule 9 violation on {rec['id']}: {rule9.reason}"


def test_ratings_ingestion_from_exported_json(tmp_path: Path) -> None:
    """Verify exported extension thumbs-down ratings parse and route without Stockfish subprocess."""
    fen_down = "r2q1rk1/1p2ppbp/p1np1np1/2p5/3PP3/2N1BP2/PPPQ2PP/2KR1B1R b - - 2 9"
    fen_pos = "r2q1rk1/1p2ppbp/p1np1np1/2p5/3PP3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 10"
    fen_up = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    export_payload = {
        "version": "1.0",
        "exportedAt": "2026-11-20T09:00:00Z",
        "count": 5,
        "ratings": [
            {
                "id": "r_1",
                "timestamp": 1700000001,
                "fen": fen_down,
                "move": "Qa5",
                "rating": "down",
                "prompt": "...",
                "output": "Blocks coordination.",
                "level": "club",
                "detail": "short",
                "model": "Second-1.0-2B",
            },
            {
                "id": "r_2",
                "timestamp": 1700000002,
                "fen": fen_down,
                "move": "d8a5",
                "rating": "down",
                "prompt": "...",
                "output": "Duplicate entry.",
                "level": "club",
                "detail": "short",
                "model": "Second-1.0-2B",
            },
            {
                "id": "r_3",
                "timestamp": 1700000003,
                "fen": fen_pos,
                "move": "position",
                "rating": "down",
                "prompt": "...",
                "output": "Vague position.",
                "level": "beginner",
                "detail": "short",
                "model": "Second-1.0-2B",
            },
            {
                "id": "r_4",
                "timestamp": 1700000004,
                "fen": fen_pos,
                "move": "Qxh8",
                "rating": "down",
                "prompt": "...",
                "output": "Malformed move.",
                "level": "club",
                "detail": "short",
                "model": "Second-1.0-2B",
            },
            {
                "id": "r_5",
                "timestamp": 1700000005,
                "fen": fen_up,
                "move": "c5",
                "rating": "up",
                "prompt": "...",
                "output": "Good.",
                "level": "beginner",
                "detail": "short",
                "model": "Second-1.0-2B",
            },
        ],
    }
    ratings_file = tmp_path / "second_ratings_test.json"
    ratings_file.write_text(json.dumps(export_payload), encoding="utf-8")

    parsed = parse_exported_ratings_files([ratings_file])
    assert len(parsed) == 2
    assert parsed[0]["move_uci"] == "d8a5"
    assert parsed[0]["task"] == "explain_move"
    assert parsed[1]["move_uci"] is None
    assert parsed[1]["task"] == "explain_position"

    # Build a small v1.1 pair using mocked Stockfish analyzer
    sample_dev_in = tmp_path / "dev_in.jsonl"
    sample_test_in = tmp_path / "test_in.jsonl"
    sample_dev_in.write_text(
        DEFAULT_DEVSET_1_1.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    sample_test_in.write_text(
        DEFAULT_TESTSET_1_1.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )

    def mock_analyzer(board: chess.Board, _depth: int) -> dict[str, Any]:
        moves = [next(iter(board.legal_moves)).uci()] if board.legal_moves.count() > 0 else []
        return {"fen": board.fen(), "pvs": [{"score": {"type": "cp", "value": 25}, "moves": moves}]}

    dev_out = tmp_path / "dev_out.jsonl"
    test_out = tmp_path / "test_out.jsonl"
    dev_recs, test_recs = build_eval_v1_1(
        dev_in_path=sample_dev_in,
        test_in_path=sample_test_in,
        dev_out_path=dev_out,
        test_out_path=test_out,
        split_dir=DEFAULT_SPLIT_DIR,
        ratings_files=[ratings_file],
        analyzer=mock_analyzer,
    )
    all_recs = dev_recs + test_recs
    rating_recs = [r for r in all_recs if r.get("source") == "soft_launch_ratings"]
    assert len(rating_recs) == 2
    assert {normalize_fen(r["fen"]) for r in rating_recs} == {
        normalize_fen(fen_down),
        normalize_fen(fen_pos),
    }

    # Verify SAN number/annotation stripping and Chess960 castling UCI rejection on standard FENs
    scholars_board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    assert _resolve_move_uci(scholars_board, "14. Qxf7+!") == "h5f7"
    std_castle_board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert _resolve_move_uci(std_castle_board, "e1g1") == "e1g1"
    assert _resolve_move_uci(std_castle_board, "e1h1") is None

    shared_gid_payload = {
        "ratings": [
            {
                "id": "sg_1",
                "game_id": "shared_game_xyz",
                "fen": "r2q1rk1/1p2ppbp/p1np1np1/2p5/3PP3/2N1BP2/PPPQ2PP/2KR1B1R b - - 2 9",
                "move": "Qa5",
                "rating": "down",
            },
            {
                "id": "sg_2",
                "game_id": "shared_game_xyz",
                "fen": "r2q1rk1/1p2ppbp/p1np1np1/2p5/3PP3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 10",
                "move": "Kb1",
                "rating": "down",
            },
        ]
    }
    sg_file = tmp_path / "shared_gid.json"
    sg_file.write_text(json.dumps(shared_gid_payload), encoding="utf-8")
    sg_dev, sg_test = build_eval_v1_1(
        dev_in_path=sample_dev_in,
        test_in_path=sample_test_in,
        dev_out_path=tmp_path / "sg_dev.jsonl",
        test_out_path=tmp_path / "sg_test.jsonl",
        split_dir=DEFAULT_SPLIT_DIR,
        ratings_files=[sg_file],
        analyzer=mock_analyzer,
    )
    sg_in_dev = [r for r in sg_dev if r.get("source") == "soft_launch_ratings"]
    sg_in_test = [r for r in sg_test if r.get("source") == "soft_launch_ratings"]
    assert (len(sg_in_dev) == 2 and len(sg_in_test) == 0) or (
        len(sg_in_dev) == 0 and len(sg_in_test) == 2
    )

    # Verify excluded_games in ratings is skipped and dev/test game_id overlap fails verification
    from data.pipeline.rl_prompts import load_excluded_keys

    excl_games, _ = load_excluded_keys(DEFAULT_SPLIT_DIR, split_names=ALL_TRAINING_SPLITS)
    leaked_gid = next(iter(excl_games))
    fresh_gid_file = tmp_path / "excluded_gid_ratings.json"
    fresh_gid_file.write_text(
        json.dumps({
            "ratings": [
                {
                    "id": "leaked_1",
                    "game_id": leaked_gid,
                    "fen": "r1bqk2r/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
                    "move": "position",
                    "rating": "down",
                }
            ]
        }),
        encoding="utf-8",
    )
    ex_dev, ex_test = build_eval_v1_1(
        dev_in_path=sample_dev_in,
        test_in_path=sample_test_in,
        dev_out_path=tmp_path / "ex_dev.jsonl",
        test_out_path=tmp_path / "ex_test.jsonl",
        split_dir=DEFAULT_SPLIT_DIR,
        ratings_files=[fresh_gid_file],
        analyzer=mock_analyzer,
    )
    assert all(r.get("source") != "soft_launch_ratings" for r in ex_dev + ex_test)

    with pytest.raises(ValueError, match="Game ID contamination"):
        verify_v1_1_zero_contamination(
            dev_records=[{"game_id": "dup_gid_1", "fen": "r2q1rk1/1p2ppbp/p1np1np1/2p5/3PP3/2N1BP2/PPPQ2PP/2KR1B1R b - - 2 9"}],
            test_records=[{"game_id": "dup_gid_1", "fen": "r2q1rk1/1p2ppbp/p1np1np1/2p5/3PP3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 10"}],
            split_dir=tmp_path,
        )
