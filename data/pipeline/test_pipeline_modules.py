"""Unit tests for Phase 2 data pipeline modules.

This module verifies:
- Stockfish binary discovery and engine worker pools.
- Parquet evaluation cache reading, writing, and updates.
- Position evaluation and record factual enrichment.
- Game-level dataset partitioning and collision resolution.
- Reinforcement learning prompt generation and target distribution.
- Dataset export, Hugging Face preparation, and Drive synchronization.
- End-to-end pipeline execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import chess
import pandas as pd
import pytest

from data.pipeline.analyse import (
    StockfishAnalyzer,
    StockfishCache,
    StockfishPool,
    analyse_records,
    find_stockfish_binary,
)
from data.pipeline.publish import (
    prepare_hf_export,
    sync_to_drive,
    upload_to_hf,
)
from data.pipeline.rl_prompts import (
    build_rl_prompts_from_evaluated,
    generate_rl_prompts,
    load_excluded_keys,
    save_rl_prompts,
)
from data.pipeline.run_pipeline import (
    run_pipeline,
)
from data.pipeline.splits import (
    hash_game_id,
    normalize_fen,
    save_splits,
    split_by_game,
)


@pytest.fixture(scope="module")
def stockfish_path() -> str:
    """Provide verified Stockfish executable path for tests."""
    return find_stockfish_binary()


def test_find_stockfish_binary(stockfish_path: str) -> None:
    """Verify that the system finds an executable Stockfish binary."""
    assert Path(stockfish_path).exists()
    assert os.access(stockfish_path, os.X_OK)

    with pytest.raises(FileNotFoundError):
        find_stockfish_binary("/path/that/does/not/exist/stockfish_fake")


def test_stockfish_cache_roundtrip(tmp_path: Path) -> None:
    """Verify Parquet evaluation cache storage and retrieval."""
    cache_file = tmp_path / "cache.parquet"
    cache = StockfishCache(cache_path=cache_file)
    assert len(cache) == 0

    fen = chess.STARTING_FEN
    pvs = [
        {"score": {"type": "cp", "value": 20}, "moves": ["e2e4", "e7e5"]},
        {"score": {"type": "cp", "value": 15}, "moves": ["d2d4", "d7d5"]},
    ]
    cache.put(fen, depth=12, multipv=2, pvs=pvs)
    assert len(cache) == 1

    # Test retrieval with sufficient and insufficient limits
    assert cache.get(fen, min_depth=10, min_multipv=2) is not None
    assert cache.get(fen, min_depth=15, min_multipv=2) is None
    assert cache.get(fen, min_depth=10, min_multipv=3) is None

    # Test multipv slicing
    retrieved = cache.get(fen, min_depth=10, min_multipv=1)
    assert retrieved is not None
    assert len(retrieved["pvs"]) == 1

    # Save to disk and reload in a new instance
    cache.save()
    assert cache_file.exists()

    cache_reloaded = StockfishCache(cache_path=cache_file)
    assert len(cache_reloaded) == 1
    reloaded_eval = cache_reloaded.get(fen, min_depth=12, min_multipv=2)
    assert reloaded_eval is not None
    assert reloaded_eval["pvs"][0]["score"]["value"] == 20


def test_stockfish_pool_and_analyzer(stockfish_path: str, tmp_path: Path) -> None:
    """Verify concurrent position analysis with Stockfish worker pool."""
    with StockfishPool(stockfish_path, num_workers=2) as pool:
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        res = pool.analyze_fen(fen, depth=6, multipv=2)
        assert res["fen"] == fen
        assert len(res["pvs"]) == 2
        assert "score" in res["pvs"][0]
        assert "moves" in res["pvs"][0]

    # Test game over position (checkmate)
    mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    analyzer = StockfishAnalyzer(
        stockfish_path=stockfish_path,
        cache_path=tmp_path / "cache.parquet",
        depth=6,
        num_workers=2,
    )
    analyzer.analyze_fens([(mate_fen, 1)])
    mate_eval = analyzer.cache.get(mate_fen, min_depth=6, min_multipv=1)
    assert mate_eval is not None
    assert mate_eval["pvs"][0]["score"]["type"] == "mate"
    # White is checkmated, so score for White is negative
    assert mate_eval["pvs"][0]["score"]["value"] < 0


def test_analyse_records(stockfish_path: str, tmp_path: Path) -> None:
    """Verify record enrichment with evaluations, facts, and prompt text."""
    cache = StockfishCache(cache_path=tmp_path / "eval_cache.parquet")
    records = [
        {
            "game_id": "game_1",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "move_uci": "f3g5",
        },
        {
            "game_id": "game_2",
            "task": "explain_position",
            "level": "beginner",
            "detail": "short",
            "fen": "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
        },
    ]

    enriched = analyse_records(
        records,
        cache=cache,
        stockfish_path=stockfish_path,
        depth=6,
        num_workers=2,
    )
    assert len(enriched) == 2

    # Verify explain_move fields
    move_rec = enriched[0]
    assert move_rec["task"] == "explain_move"
    assert move_rec["verdict"] in ("Best", "Good", "Inaccuracy", "Mistake", "Blunder", "Only move")
    assert isinstance(move_rec["facts"], list)
    assert len(move_rec["facts"]) > 0
    assert "Move played: Ng5." in move_rec["user_message"]
    assert "<|im_start|>system" in move_rec["chat_prompt"]
    assert move_rec["eval_before"] is not None
    assert move_rec["eval_after"] is not None

    # Verify explain_position fields
    pos_rec = enriched[1]
    assert pos_rec["task"] == "explain_position"
    assert "Move played" not in pos_rec["user_message"]
    assert "Verdict" not in pos_rec["user_message"]
    assert pos_rec["eval_before"] is not None
    assert pos_rec["eval_after"] is None


def test_normalize_fen() -> None:
    """Verify normalization of FEN string to its four-field board state."""
    full_fen_1 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    full_fen_2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 5 12"
    norm_1 = normalize_fen(full_fen_1)
    norm_2 = normalize_fen(full_fen_2)
    assert norm_1 == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    assert norm_1 == norm_2


def test_split_by_game_and_collision_resolution(tmp_path: Path) -> None:
    """Verify game-level partitioning and cross-split position collision resolution."""
    # Create 100 mock games
    records = []
    for g in range(100):
        gid = f"game_{g:03d}"
        for p in range(3):
            records.append({
                "game_id": gid,
                "fen": f"fen_unique_{g}_{p}",
                "redistributable": (g % 2 == 0),
            })

    # Add shared opening FENs between games
    common_opening = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    records.append({"game_id": "game_001", "fen": common_opening, "redistributable": True})
    records.append({"game_id": "game_050", "fen": common_opening, "redistributable": True})
    records.append({"game_id": "game_099", "fen": common_opening, "redistributable": True})

    splits = split_by_game(records, val_ratio=0.10, test_ratio=0.20, seed=123)

    train_games = {r["game_id"] for r in splits["train"]}
    val_games = {r["game_id"] for r in splits["val"]}
    test_games = {r["game_id"] for r in splits["test_pool"]}

    # Verify game isolation: no game appears in multiple splits
    assert not (train_games & val_games)
    assert not (train_games & test_games)
    assert not (val_games & test_games)

    # Verify position isolation: no FEN appears in multiple splits
    train_fens = {r["fen"] for r in splits["train"]}
    val_fens = {r["fen"] for r in splits["val"]}
    test_fens = {r["fen"] for r in splits["test_pool"]}

    assert not (train_fens & val_fens)
    assert not (train_fens & test_fens)
    assert not (val_fens & test_fens)

    # Verify that the shared common opening FEN appears in at most one split
    splits_with_common = [
        name
        for name, fset in (("train", train_fens), ("val", val_fens), ("test_pool", test_fens))
        if common_opening in fset
    ]
    assert len(splits_with_common) == 1

    # Verify that positions with different move clocks but identical board states collide
    fen_clock_0 = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    fen_clock_5 = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 5 10"
    records_clock = [
        {"game_id": "game_clock_a", "fen": fen_clock_0, "redistributable": True},
        {"game_id": "game_clock_b", "fen": fen_clock_5, "redistributable": True},
    ]
    splits_clock = split_by_game(records_clock, val_ratio=0.50, test_ratio=0.0, seed=42)
    norm_train = {normalize_fen(r["fen"]) for r in splits_clock["train"]}
    norm_val = {normalize_fen(r["fen"]) for r in splits_clock["val"]}
    assert not (norm_train & norm_val)

    # Test saving splits to disk
    out_dir = tmp_path / "processed"
    pub_dir = tmp_path / "processed" / "public"
    stats = save_splits(splits, output_dir=out_dir, public_dir=pub_dir)

    total_kept = sum(len(v) for v in splits.values())
    assert stats["total_records"] == total_kept
    assert (out_dir / "sft_train.jsonl").exists()
    assert (out_dir / "sft_train.parquet").exists()
    assert (pub_dir / "sft_train.parquet").exists()

    # Public files must only contain redistributable records
    pub_df = pd.read_parquet(pub_dir / "sft_train.parquet")
    if not pub_df.empty:
        assert (pub_df["redistributable"] == True).all()  # noqa: E712


def test_rl_prompt_generator(tmp_path: Path) -> None:
    """Verify RL prompt generator distribution, exclusions, and Qwen formatting."""
    eval_before = {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "pvs": [
            {"score": {"type": "cp", "value": 30}, "moves": ["d2d3", "f8c5"]},
            {"score": {"type": "cp", "value": 20}, "moves": ["f3g5", "d7d5"]},
            {"score": {"type": "cp", "value": 0}, "moves": ["b1c3", "f8c5"]},
        ],
    }
    eval_after_good = {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
        "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["f8c5"]}],
    }
    eval_after_bad = {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 5 5",
        "pvs": [{"score": {"type": "cp", "value": -350}, "moves": ["f6e4"]}],
    }

    candidates = []
    # Create 30 bad move candidates
    for i in range(30):
        candidates.append({
            "game_id": f"game_bad_{i}",
            "fen": eval_before["fen"],
            "move_uci": "f3g1",
            "verdict": "Blunder",
            "eval_before": eval_before,
            "eval_after": eval_after_bad,
        })
    # Create 25 good move candidates
    for i in range(25):
        candidates.append({
            "game_id": f"game_good_{i}",
            "fen": eval_before["fen"],
            "move_uci": "d2d3",
            "verdict": "Best",
            "eval_before": eval_before,
            "eval_after": eval_after_good,
        })
    # Create 15 position candidates
    for i in range(15):
        candidates.append({
            "game_id": f"game_pos_{i}",
            "fen": eval_before["fen"],
            "eval_before": eval_before,
        })

    # Test exclusion filtering
    excluded_games = {"game_bad_0", "game_good_0"}
    excluded_fens = {"fen_excluded_1"}

    prompts = build_rl_prompts_from_evaluated(
        candidates,
        target_count=20,
        excluded_games=excluded_games,
        excluded_fens=excluded_fens,
    )

    assert len(prompts) == 20
    for p in prompts:
        assert p["game_id"] not in excluded_games
        assert p["source"] == "lichess_games"
        assert p["redistributable"] is True
        assert "<|im_start|>system" in p["chat_prompt"]
        assert "<think>" in p["chat_prompt"]

    # Verify level balance (50% beginner, 50% club)
    beginner_count = sum(1 for p in prompts if p["level"] == "beginner")
    club_count = sum(1 for p in prompts if p["level"] == "club")
    assert beginner_count == 10
    assert club_count == 10

    # Verify detail balance (80% short, 20% long)
    short_count = sum(1 for p in prompts if p["detail"] == "short")
    long_count = sum(1 for p in prompts if p["detail"] == "long")
    assert short_count == 16
    assert long_count == 4

    # Save to disk and verify Parquet and JSONL
    out_dir = tmp_path / "processed"
    save_stats = save_rl_prompts(prompts, output_dir=out_dir)
    assert save_stats["count"] == 20
    assert (out_dir / "rl_prompts.parquet").exists()
    assert (out_dir / "public" / "rl_prompts.parquet").exists()


def test_publish_and_drive_sync(tmp_path: Path) -> None:
    """Verify Hugging Face export package structure and Google Drive sync."""
    pub_dir = tmp_path / "processed" / "public"
    pub_dir.mkdir(parents=True, exist_ok=True)

    # Create mock public dataset files
    (pub_dir / "sft_train.parquet").write_bytes(b"mock_parquet_data")
    (pub_dir / "rl_prompts.parquet").write_bytes(b"mock_parquet_data")

    # Create mock non-redistributable file in processed parent
    secret_file = tmp_path / "processed" / "proprietary_commentary.jsonl"
    secret_file.write_text("proprietary text", encoding="utf-8")

    exp_dir = tmp_path / "export"
    prepared_path = prepare_hf_export(public_dir=pub_dir, export_dir=exp_dir)

    assert prepared_path.exists()
    assert (prepared_path / "sft_train.parquet").exists()
    assert (prepared_path / "rl_prompts.parquet").exists()
    assert (prepared_path / "README.md").exists()
    assert (prepared_path / "REBUILD.md").exists()
    assert (prepared_path / ".gitattributes").exists()

    # Verify that proprietary commentary was NOT leaked to export
    assert not (prepared_path / "proprietary_commentary.jsonl").exists()

    # Test upload status reporting without token
    upload_status = upload_to_hf(export_dir=prepared_path, token="invalid_test_token_123")
    assert not upload_status["uploaded"]
    assert "reason" in upload_status

    # Test Drive synchronization
    drive_dir = tmp_path / "google_drive" / "second" / "data"
    cache_file = tmp_path / "cache.parquet"
    cache_file.write_bytes(b"cache_bytes")

    sync_status = sync_to_drive(
        source_dir=pub_dir,
        drive_dir=drive_dir,
        cache_path=cache_file,
    )
    assert sync_status["synced"] is True
    assert sync_status["files_copied"] >= 2
    assert (drive_dir / "sft_train.parquet").exists()
    assert (drive_dir / "cache" / "cache.parquet").exists()


def test_run_pipeline_end_to_end(stockfish_path: str, tmp_path: Path) -> None:
    """Verify full end-to-end pipeline execution with mock inputs."""
    sft_file = tmp_path / "raw_sft.jsonl"
    sft_records = [
        {
            "game_id": "game_100",
            "fen": "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
            "move_uci": "e2e4",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "comment": "White pushes the pawn to control the center.",
            "source": "mate",
            "redistributable": True,
        },
        {
            "game_id": "game_101",
            "fen": "8/8/8/8/8/8/8/4K2k w - - 0 1",
            "task": "explain_position",
            "level": "beginner",
            "detail": "short",
            "comment": "Equal king and pawn endgame.",
            "source": "gameknot",
            "redistributable": False,
        },
    ]
    with open(sft_file, "w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "processed"
    pub_dir = tmp_path / "processed" / "public"
    cache_file = tmp_path / "cache.parquet"
    drive_dir = tmp_path / "drive_backup"

    report = run_pipeline(
        sft_input_path=sft_file,
        rl_input_path=None,
        output_dir=out_dir,
        public_dir=pub_dir,
        cache_path=cache_file,
        drive_dir=drive_dir,
        stockfish_path=stockfish_path,
        depth=6,
        num_workers=2,
        upload_hf=False,
        sync_drive=True,
    )

    assert report["sft_records_in"] == 2
    assert report["cache_size"] > 0
    assert (out_dir / "sft_train.parquet").exists()
    assert (pub_dir / "sft_train.parquet").exists()
    assert (drive_dir / "sft_train.parquet").exists()
    assert Path(report["export_directory"]).resolve() == (out_dir / "hf_export").resolve()
    assert (out_dir / "hf_export").exists()
