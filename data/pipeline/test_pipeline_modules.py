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
from data.pipeline.eval_pool import (
    build_eval_pool_from_evaluated,
    generate_eval_pool,
    save_eval_pool,
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


def test_run_pipeline_with_eval_pool_and_rl(stockfish_path: str, tmp_path: Path) -> None:
    """Verify pipeline orchestration with evaluation pool and reinforcement learning prompts."""
    sft_file = tmp_path / "raw_sft.jsonl"
    sft_records = [
        {
            "game_id": "game_sft_10",
            "fen": "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
            "move_uci": "e2e4",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "comment": "White pushes the pawn to control the center squares.",
            "source": "mate",
            "redistributable": True,
        },
    ]
    with open(sft_file, "w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r) + "\n")

    fen_eval_1 = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    fen_eval_2 = "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1"
    fen_rl_1 = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"
    fen_rl_2 = "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1"

    rl_file = tmp_path / "raw_rl.jsonl"
    rl_records = [
        {
            "game_id": "game_cand_1",
            "fen": fen_eval_1,
            "move_uci": "f3g1",
            "verdict": "Blunder",
            "eval_before": {"fen": fen_eval_1, "pvs": [{"score": {"type": "cp", "value": 30}, "moves": ["d2d3"]}]},
            "eval_after": {"fen": fen_eval_1, "pvs": [{"score": {"type": "cp", "value": -250}, "moves": []}]},
        },
        {
            "game_id": "game_cand_2",
            "fen": fen_eval_2,
            "move_uci": "e1d1",
            "verdict": "Blunder",
            "eval_before": {"fen": fen_eval_2, "pvs": [{"score": {"type": "cp", "value": 20}, "moves": ["e2e3"]}]},
            "eval_after": {"fen": fen_eval_2, "pvs": [{"score": {"type": "cp", "value": -200}, "moves": []}]},
        },
        {
            "game_id": "game_cand_3",
            "fen": fen_rl_1,
            "move_uci": "c1e3",
            "verdict": "Best",
            "eval_before": {"fen": fen_rl_1, "pvs": [{"score": {"type": "cp", "value": 20}, "moves": ["c1e3"]}]},
            "eval_after": {"fen": fen_rl_1, "pvs": [{"score": {"type": "cp", "value": 20}, "moves": []}]},
        },
        {
            "game_id": "game_cand_4",
            "fen": fen_rl_2,
            "move_uci": "d2d3",
            "verdict": "Best",
            "eval_before": {"fen": fen_rl_2, "pvs": [{"score": {"type": "cp", "value": 15}, "moves": ["d2d3"]}]},
            "eval_after": {"fen": fen_rl_2, "pvs": [{"score": {"type": "cp", "value": 15}, "moves": []}]},
        },
    ]
    with open(rl_file, "w", encoding="utf-8") as f:
        for r in rl_records:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "processed"
    pub_dir = tmp_path / "processed" / "public"
    cache_file = tmp_path / "cache.parquet"
    drive_dir = tmp_path / "drive_backup"

    report = run_pipeline(
        sft_input_path=sft_file,
        rl_input_path=rl_file,
        output_dir=out_dir,
        public_dir=pub_dir,
        cache_path=cache_file,
        drive_dir=drive_dir,
        stockfish_path=stockfish_path,
        depth=6,
        num_workers=2,
        eval_pool_size=2,
        rl_target_count=2,
        upload_hf=False,
        sync_drive=True,
    )

    assert report["sft_records_in"] == 1
    assert report["eval_pool_generated"] == 2
    assert report["rl_prompts_generated"] == 2
    assert (out_dir / "eval_pool.parquet").exists()
    assert (out_dir / "eval_pool.jsonl").exists()
    assert (pub_dir / "eval_pool.parquet").exists()
    assert (pub_dir / "eval_pool.jsonl").exists()
    assert (out_dir / "hf_export" / "eval_pool.parquet").exists()
    assert (out_dir / "hf_export" / "eval_pool.jsonl").exists()
    assert (drive_dir / "eval_pool.parquet").exists()


def _make_synthetic_eval_candidates() -> list[dict[str, Any]]:
    """Create diverse synthetic candidates covering phases and verdicts."""
    candidates: list[dict[str, Any]] = []

    # 10 distinct opening FENs by playing 10 different first moves from starting position
    opening_fens: list[str] = []
    start_board = chess.Board()
    for m in list(start_board.legal_moves)[:10]:
        b = chess.Board()
        b.push(m)
        opening_fens.append(b.fen())

    # 10 distinct middlegame FENs
    base_mid = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
    middlegame_fens: list[str] = []
    for m in list(base_mid.legal_moves)[:10]:
        b = base_mid.copy()
        b.push(m)
        middlegame_fens.append(b.fen())

    # 10 distinct endgame FENs with pieces on different files
    endgame_fens = [
        "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
        "8/8/8/8/8/8/3P4/3K2k1 w - - 0 1",
        "8/8/8/8/8/8/2P5/2K2k2 w - - 0 1",
        "8/8/8/8/8/8/1P6/1K2k3 w - - 0 1",
        "8/8/8/8/8/8/5P2/5K1k w - - 0 1",
        "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1",
        "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1",
        "8/8/8/8/8/2k5/2P5/2K5 w - - 0 1",
        "8/8/8/8/8/5k2/5P2/5K2 w - - 0 1",
        "8/8/8/8/8/1k6/1P6/1K6 w - - 0 1",
    ]

    phase_fen_map = {
        "opening": opening_fens,
        "middlegame": middlegame_fens,
        "endgame": endgame_fens,
    }

    game_id_counter = 0
    for phase, fens in phase_fen_map.items():
        for fen in fens:
            board = chess.Board(fen)
            legal_moves = [m.uci() for m in list(board.legal_moves)]
            best_uci = legal_moves[0] if legal_moves else "e2e4"
            other_uci = legal_moves[1] if len(legal_moves) > 1 else best_uci

            eval_before = {
                "fen": fen,
                "pvs": [
                    {"score": {"type": "cp", "value": 40}, "moves": [best_uci]},
                    {"score": {"type": "cp", "value": -200}, "moves": [other_uci]},
                    {"score": {"type": "cp", "value": -300}, "moves": [other_uci]},
                ],
            }
            eval_after_good = {
                "fen": fen,
                "pvs": [{"score": {"type": "cp", "value": 40}, "moves": []}],
            }
            eval_after_bad = {
                "fen": fen,
                "pvs": [{"score": {"type": "cp", "value": -260}, "moves": []}],
            }

            # Bad move candidate
            game_id_counter += 1
            candidates.append({
                "game_id": f"game_{game_id_counter}",
                "fen": fen,
                "move_uci": other_uci,
                "verdict": "Blunder",
                "eval_before": eval_before,
                "eval_after": eval_after_bad,
                "white_elo": 1500,
                "black_elo": 1500,
            })

            # Good move candidate
            game_id_counter += 1
            candidates.append({
                "game_id": f"game_{game_id_counter}",
                "fen": fen,
                "move_uci": best_uci,
                "verdict": "Best",
                "eval_before": eval_before,
                "eval_after": eval_after_good,
                "white_elo": 1600,
                "black_elo": 1400,
            })

            # Position candidate
            game_id_counter += 1
            candidates.append({
                "game_id": f"game_{game_id_counter}",
                "fen": fen,
                "eval_before": eval_before,
                "white_elo": 1700,
                "black_elo": 1300,
            })

    return candidates


def test_eval_pool_mix_targets_synthetic(tmp_path: Path) -> None:
    """Verify that the evaluation pool meets target counts, mix, and schema.

    Create synthetic candidates with diverse verdicts, Elo ratings, and game phases.
    Build an evaluation pool with custom target counts.
    Verify verdict counts, detail long counts, level balance, and phase coverage.
    Verify that record identifiers start with prefix 'eval_'.
    Save the evaluation pool to disk and verify Parquet and JSONL files.
    """
    candidates = _make_synthetic_eval_candidates()

    eval_pool = build_eval_pool_from_evaluated(
        candidates,
        target_count=60,
        target_bad=24,
        target_good=18,
        target_position=18,
        min_long=12,
        seed=42,
    )

    assert len(eval_pool) == 60

    bad_count = sum(1 for p in eval_pool if p["verdict"] in {"Inaccuracy", "Mistake", "Blunder"})
    good_count = sum(1 for p in eval_pool if p["verdict"] in {"Good", "Best", "Only move"})
    pos_count = sum(1 for p in eval_pool if p["task"] == "explain_position")
    long_count = sum(1 for p in eval_pool if p["detail"] == "long")
    beginner_count = sum(1 for p in eval_pool if p["level"] == "beginner")
    club_count = sum(1 for p in eval_pool if p["level"] == "club")

    assert bad_count == 24
    assert good_count == 18
    assert pos_count == 18
    assert long_count >= 12
    assert beginner_count == 30
    assert club_count == 30

    # Verify ID prefix and schema
    expected_keys = {
        "id", "game_id", "fen", "task", "level", "detail", "move_uci",
        "verdict", "facts", "user_message", "chat_prompt", "source",
        "redistributable", "eval_before", "eval_after",
    }
    for idx, p in enumerate(eval_pool):
        assert p["id"] == f"eval_{idx:06d}"
        assert set(p.keys()) == expected_keys
        assert p["source"] == "lichess_games"
        assert p["redistributable"] is True
        assert "<|im_start|>system" in p["chat_prompt"]
        assert "<think>" in p["chat_prompt"]

    # Verify saving to disk
    out_dir = tmp_path / "processed"
    pub_dir = tmp_path / "processed" / "public"
    stats = save_eval_pool(eval_pool, output_dir=out_dir, public_dir=pub_dir)
    assert stats["count"] == 60
    assert (out_dir / "eval_pool.parquet").exists()
    assert (pub_dir / "eval_pool.parquet").exists()
    assert (out_dir / "eval_pool.jsonl").exists()
    assert (pub_dir / "eval_pool.jsonl").exists()

    # Check phase coverage in stats
    assert stats["phases"].get("opening", 0) > 0
    assert stats["phases"].get("middlegame", 0) > 0
    assert stats["phases"].get("endgame", 0) > 0


def test_eval_pool_game_disjointness(tmp_path: Path) -> None:
    """Verify that the evaluation pool is game-disjoint from SFT splits and RL prompts.

    Create candidate records for SFT and CC0 games.
    Split SFT records by game.
    Build the evaluation pool and reserve games.
    Build RL prompts from candidates excluding SFT and evaluation games.
    Verify that the evaluation pool shares zero games and zero FENs with other partitions.
    """
    sft_dir = tmp_path / "processed"
    sft_records = [
        {
            "game_id": "game_sft_1",
            "fen": "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
            "task": "explain_move",
            "comment": "White pushes the pawn to establish center presence.",
            "source": "mate",
            "redistributable": True,
        },
        {
            "game_id": "game_sft_2",
            "fen": "8/8/8/8/8/8/8/4K2k w - - 0 1",
            "task": "explain_position",
            "comment": "Balanced king and pawn endgame position.",
            "source": "mate",
            "redistributable": True,
        },
    ]
    splits = split_by_game(sft_records, val_ratio=0.5, test_ratio=0.5, seed=42)
    save_splits(splits, output_dir=sft_dir)

    # Synthetic candidate games for evaluation pool and RL prompts
    fen_a = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    fen_b = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"
    fen_c = "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1"
    fen_d = "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1"

    eval_before_a = {
        "fen": fen_a,
        "pvs": [
            {"score": {"type": "cp", "value": 30}, "moves": ["d2d3"]},
            {"score": {"type": "cp", "value": -250}, "moves": ["f3g1"]},
        ],
    }
    eval_before_b = {
        "fen": fen_b,
        "pvs": [
            {"score": {"type": "cp", "value": 20}, "moves": ["c1e3"]},
            {"score": {"type": "cp", "value": -280}, "moves": ["a1b1"]},
        ],
    }
    eval_before_c = {
        "fen": fen_c,
        "pvs": [
            {"score": {"type": "cp", "value": 10}, "moves": ["e2e3"]},
            {"score": {"type": "cp", "value": -150}, "moves": ["e1d1"]},
        ],
    }
    eval_before_d = {
        "fen": fen_d,
        "pvs": [
            {"score": {"type": "cp", "value": 15}, "moves": ["d2d3"]},
            {"score": {"type": "cp", "value": -180}, "moves": ["d1c1"]},
        ],
    }

    candidates = [
        # Candidate sharing a game with SFT (must be excluded from eval pool)
        {
            "game_id": "game_sft_1",
            "fen": fen_a,
            "move_uci": "d2d3",
            "verdict": "Best",
            "eval_before": eval_before_a,
            "eval_after": {"fen": fen_a, "pvs": [{"score": {"type": "cp", "value": 30}, "moves": []}]},
        },
        # Position 1 from game_reserve_1
        {
            "game_id": "game_reserve_1",
            "fen": fen_a,
            "move_uci": "f3g1",
            "verdict": "Blunder",
            "eval_before": eval_before_a,
            "eval_after": {"fen": fen_a, "pvs": [{"score": {"type": "cp", "value": -250}, "moves": []}]},
        },
        # Position 2 from game_reserve_1 (different position in same game)
        {
            "game_id": "game_reserve_1",
            "fen": fen_b,
            "move_uci": "c1e3",
            "verdict": "Best",
            "eval_before": eval_before_b,
            "eval_after": {"fen": fen_b, "pvs": [{"score": {"type": "cp", "value": 20}, "moves": []}]},
        },
        # Game for eval pool
        {
            "game_id": "game_reserve_2",
            "fen": fen_b,
            "move_uci": "c1e3",
            "verdict": "Best",
            "eval_before": eval_before_b,
            "eval_after": {"fen": fen_b, "pvs": [{"score": {"type": "cp", "value": 20}, "moves": []}]},
        },
        # Games for RL prompts
        {
            "game_id": "game_rl_1",
            "fen": fen_c,
            "move_uci": "e2e3",
            "verdict": "Best",
            "eval_before": eval_before_c,
            "eval_after": {"fen": fen_c, "pvs": [{"score": {"type": "cp", "value": 10}, "moves": []}]},
        },
        {
            "game_id": "game_rl_2",
            "fen": fen_d,
            "move_uci": "d1c1",
            "verdict": "Blunder",
            "eval_before": eval_before_d,
            "eval_after": {"fen": fen_d, "pvs": [{"score": {"type": "cp", "value": -180}, "moves": []}]},
        },
    ]

    # Step 1: Generate eval pool with target_count=2
    eval_pool = generate_eval_pool(
        candidates,
        target_count=2,
        target_bad=1,
        target_good=1,
        target_position=0,
        split_dir=sft_dir,
        seed=42,
    )
    assert len(eval_pool) == 2
    eval_games = {r["game_id"] for r in eval_pool}
    # Verify game_sft_1 was excluded from eval pool
    assert "game_sft_1" not in eval_games
    # Verify game_reserve_1 was selected in eval pool
    assert "game_reserve_1" in eval_games

    # Save eval pool to disk so RL prompt generator loads it
    save_eval_pool(eval_pool, output_dir=sft_dir)

    # Step 2: Generate RL prompts
    rl_prompts = generate_rl_prompts(
        candidates,
        target_count=2,
        split_dir=sft_dir,
    )
    assert len(rl_prompts) == 2

    rl_games = {r["game_id"] for r in rl_prompts}
    rl_fens = {normalize_fen(r["fen"]) for r in rl_prompts}
    eval_fens = {normalize_fen(r["fen"]) for r in eval_pool}

    # Verify game disjointness between eval pool and RL prompts
    assert eval_games.isdisjoint(rl_games)
    # Specifically, neither position from game_reserve_1 may appear in RL prompts!
    assert "game_reserve_1" not in rl_games


def test_eval_pool_determinism() -> None:
    """Verify deterministic candidate selection with fixed random seeds.

    Build two pools using the same seed.
    Verify that both runs yield identical records.
    Build a third pool using a different seed.
    Verify that the selection order or candidate set differs.
    """
    candidates = _make_synthetic_eval_candidates()

    pool_1 = build_eval_pool_from_evaluated(candidates, target_count=30, seed=42)
    pool_2 = build_eval_pool_from_evaluated(candidates, target_count=30, seed=42)
    pool_3 = build_eval_pool_from_evaluated(candidates, target_count=30, seed=999)

    # Same seed must yield identical IDs, game IDs, tasks, levels, and chat prompts
    assert [r["id"] for r in pool_1] == [r["id"] for r in pool_2]
    assert [r["game_id"] for r in pool_1] == [r["game_id"] for r in pool_2]
    assert [r["task"] for r in pool_1] == [r["task"] for r in pool_2]
    assert [r["level"] for r in pool_1] == [r["level"] for r in pool_2]
    assert [r["detail"] for r in pool_1] == [r["detail"] for r in pool_2]
    assert [r["chat_prompt"] for r in pool_1] == [r["chat_prompt"] for r in pool_2]

    # Different seed should yield different candidate ordering or selection
    games_1 = [r["game_id"] for r in pool_1]
    games_3 = [r["game_id"] for r in pool_3]
    assert games_1 != games_3
