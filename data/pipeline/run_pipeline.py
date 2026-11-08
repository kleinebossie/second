"""End-to-end data pipeline orchestrator module.

This module coordinates the complete Phase 2 data pipeline:
- Ingestion of raw commentary records.
- Quality filtering and audience level labeling.
- Stockfish evaluation and Parquet caching.
- Game-level dataset partitioning.
- Reinforcement learning prompt generation.
- Dataset export and synchronization.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import chess

from .analyse import (
    StockfishAnalyzer,
    StockfishCache,
    analyse_records,
    find_stockfish_binary,
)
from .eval_pool import (
    generate_eval_pool,
    save_eval_pool,
)
from .label_levels import (
    label_level,
)
from .publish import (
    prepare_hf_export,
    sync_to_drive,
    upload_to_hf,
)
from .rl_prompts import (
    generate_rl_prompts,
    save_rl_prompts,
)
from .splits import (
    save_splits,
    split_by_game,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_PUBLIC_DIR = Path("data/processed/public")


def run_pipeline(
    sft_input_path: Path | str | None = None,
    rl_input_path: Path | str | None = None,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    public_dir: Path | str = DEFAULT_PUBLIC_DIR,
    export_dir: Path | str | None = None,
    cache_path: Path | str | None = None,
    drive_dir: Path | str | None = None,
    stockfish_path: str | Path | None = None,
    depth: int = 12,
    num_workers: int = 4,
    eval_pool_size: int = 600,
    rl_target_count: int = 20000,
    seed: int = 42,
    filter_sft: bool = True,
    upload_hf: bool = False,
    sync_drive: bool = False,
) -> dict[str, Any]:
    """Execute the end-to-end data pipeline.

    Verify the Stockfish executable binary.
    Load and process supervised fine-tuning records.
    Evaluate positions and populate facts.
    Split records by game into train, validation, and test pool partitions.
    Generate the evaluation pool to reserve games and evaluation positions.
    Generate reinforcement learning prompt sets excluding evaluation games and positions.
    Prepare export files and synchronize to storage targets.
    Return execution summary metrics.
    """
    sf_bin = find_stockfish_binary(stockfish_path)
    LOGGER.info("Using Stockfish binary: %s", sf_bin)

    out_path = Path(output_dir)
    pub_path = Path(public_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pub_path.mkdir(parents=True, exist_ok=True)

    cache = StockfishCache(cache_path=cache_path, drive_dir=drive_dir)
    report: dict[str, Any] = {
        "sft_records_in": 0,
        "sft_splits": {},
        "eval_pool_generated": 0,
        "rl_prompts_generated": 0,
        "cache_size": 0,
    }

    # Step 1: Process SFT commentary data
    sft_records: list[dict[str, Any]] = []
    if sft_input_path and Path(sft_input_path).exists():
        with open(sft_input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sft_records.append(json.loads(line))
        LOGGER.info("Loaded %d raw SFT records from %s", len(sft_records), sft_input_path)

    report["sft_records_in"] = len(sft_records)

    if sft_records:
        LOGGER.info("Pre-filtering SFT records before Stockfish analysis")
        from .filter_comments import (
            FilterStats,
            assign_detail_and_truncate,
            pre_filter_comment,
        )

        filter_stats = FilterStats()
        pre_seen_fps: set[str] = set()
        pre_seen_shingles: list[set[tuple[str, ...]]] = []
        pre_filtered_sft: list[tuple[dict[str, Any], str]] = []

        if filter_sft:
            for rec in sft_records:
                comment = rec.get("comment", "")
                fen = rec.get("fen", "")
                move_uci = rec.get("move_uci")

                passed, reason, cleaned = pre_filter_comment(
                    text=comment,
                    fen=fen,
                    move_uci=move_uci,
                    seen_fingerprints=pre_seen_fps,
                    seen_shingles=pre_seen_shingles,
                    stats=filter_stats,
                )
                if passed:
                    pre_filtered_sft.append((rec, cleaned))
        else:
            for rec in sft_records:
                pre_filtered_sft.append((rec, rec.get("comment", "")))

        LOGGER.info(
            "Analyzing %d pre-filtered SFT records (from %d raw) with Stockfish at depth %d",
            len(pre_filtered_sft),
            len(sft_records),
            depth,
        )
        records_to_analyse = [r for r, _ in pre_filtered_sft]
        enriched_sft = analyse_records(
            records_to_analyse,
            cache=cache,
            stockfish_path=sf_bin,
            depth=depth,
            num_workers=num_workers,
            drive_dir=drive_dir,
        )

        if filter_sft:
            from rewards.checks import check_legal_moves, check_verdict_tone

            LOGGER.info("Completing move legality, verdict-tone, and level labeling on SFT records")
            filtered_sft: list[dict[str, Any]] = []
            for rec, (_, cleaned) in zip(enriched_sft, pre_filtered_sft):
                fen = rec.get("fen", "")
                move_uci = rec.get("move_uci")
                verdict = rec.get("verdict", "Good")
                facts = rec.get("facts")

                if not check_legal_moves(text=cleaned, fen=fen, move_uci=move_uci, facts=facts).passed:
                    filter_stats.illegal_move_or_wrong_piece += 1
                    continue

                if not check_verdict_tone(text=cleaned, verdict=verdict).passed:
                    filter_stats.tone_disagrees_with_verdict += 1
                    continue

                filter_stats.kept += 1

                detail_items = assign_detail_and_truncate(cleaned)
                for item in detail_items:
                    if item["detail"] == "short":
                        filter_stats.kept_short += 1
                    elif item["detail"] == "long":
                        filter_stats.kept_long += 1
                    new_rec = dict(rec)
                    new_rec["comment"] = item["text"]
                    new_rec["detail"] = item["detail"]
                    new_rec["level"] = label_level(item["text"])
                    from core.python import build_user_message, format_qwen_chat_prompt
                    new_rec["user_message"] = build_user_message(
                        task=new_rec.get("task", "explain_move"),
                        level=new_rec["level"],
                        detail=new_rec["detail"],
                        fen=new_rec["fen"],
                        eval_before=new_rec["eval_before"],
                        eval_after=new_rec.get("eval_after"),
                        move_uci=new_rec.get("move_uci"),
                    )
                    new_rec["chat_prompt"] = format_qwen_chat_prompt(new_rec["user_message"])
                    filtered_sft.append(new_rec)
            enriched_sft = filtered_sft
            report["filter_stats"] = filter_stats.to_dict()

        LOGGER.info("Splitting %d SFT records by game", len(enriched_sft))
        splits = split_by_game(enriched_sft)
        split_stats = save_splits(splits, output_dir=out_path, public_dir=pub_path)
        report["sft_splits"] = split_stats

    # Load candidate games if provided
    rl_candidates: list[dict[str, Any]] = []
    if rl_input_path and Path(rl_input_path).exists():
        with open(rl_input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rl_candidates.append(json.loads(line))
        LOGGER.info("Loaded %d raw RL candidates from %s", len(rl_candidates), rl_input_path)

    # Step 2: Process evaluation pool from CC0 games (reserves games)
    if rl_candidates:
        LOGGER.info("Generating %d eval pool items from candidates", eval_pool_size)
        eval_items = generate_eval_pool(
            rl_candidates,
            target_count=eval_pool_size,
            stockfish_path=sf_bin,
            cache_path=cache.cache_path,
            depth=depth,
            num_workers=num_workers,
            split_dir=out_path,
            seed=seed,
        )
        eval_stats = save_eval_pool(eval_items, output_dir=out_path, public_dir=pub_path)
        report["eval_pool_generated"] = eval_stats["count"]

    # Step 3: Process RL prompts from CC0 games (excluding SFT val/test and eval pool)
    if rl_candidates:
        LOGGER.info("Generating %d RL prompts from candidates", rl_target_count)
        rl_prompts = generate_rl_prompts(
            rl_candidates,
            target_count=rl_target_count,
            stockfish_path=sf_bin,
            cache_path=cache.cache_path,
            depth=depth,
            num_workers=num_workers,
            split_dir=out_path,
        )
        rl_stats = save_rl_prompts(rl_prompts, output_dir=out_path, public_dir=pub_path)
        report["rl_prompts_generated"] = rl_stats["count"]

    # Step 4: Prepare Hugging Face export package
    exp_path = Path(export_dir) if export_dir is not None else out_path / "hf_export"
    exp_dir = prepare_hf_export(public_dir=pub_path, export_dir=exp_path)
    report["export_directory"] = str(exp_dir)

    # Step 5: Optional Hugging Face upload
    if upload_hf:
        LOGGER.info("Uploading dataset to Hugging Face Hub")
        hf_status = upload_to_hf(export_dir=exp_dir)
        report["hf_upload"] = hf_status

    # Step 6: Optional Google Drive sync
    if sync_drive or drive_dir:
        LOGGER.info("Synchronizing datasets to Google Drive")
        sync_status = sync_to_drive(
            source_dir=out_path,
            drive_dir=drive_dir,
            cache_path=cache.cache_path,
        )
        report["drive_sync"] = sync_status

    report["cache_size"] = len(cache)
    LOGGER.info("Pipeline execution finished successfully.")
    return report


def main() -> None:
    """Command line entry point for orchestrating the data pipeline."""
    parser = argparse.ArgumentParser(description="Run the Second chess assistant data pipeline.")
    parser.add_argument("--sft-input", type=str, default=None, help="Input JSONL file of human SFT commentary.")
    parser.add_argument("--rl-input", type=str, default=None, help="Input JSONL file of CC0 Lichess game positions.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Output directory for full splits.")
    parser.add_argument("--public-dir", type=str, default=str(DEFAULT_PUBLIC_DIR), help="Output directory for public splits.")
    parser.add_argument("--export-dir", type=str, default=None, help="Output directory for Hugging Face export package.")
    parser.add_argument("--cache-path", type=str, default=None, help="Path to Stockfish Parquet cache file.")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive sync directory.")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish executable.")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth.")
    parser.add_argument("--workers", type=int, default=4, help="Stockfish worker count.")
    parser.add_argument("--eval-pool-size", type=int, default=600, help="Target number of evaluation pool items.")
    parser.add_argument("--rl-target-count", type=int, default=20000, help="Target number of RL prompts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic hashing.")
    parser.add_argument("--upload-hf", action="store_true", help="Upload public files to Hugging Face Hub.")
    parser.add_argument("--sync-drive", action="store_true", help="Synchronize files to Google Drive.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    report = run_pipeline(
        sft_input_path=args.sft_input,
        rl_input_path=args.rl_input,
        output_dir=args.output_dir,
        public_dir=args.public_dir,
        export_dir=args.export_dir,
        cache_path=args.cache_path,
        drive_dir=args.drive_dir,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        num_workers=args.workers,
        eval_pool_size=args.eval_pool_size,
        rl_target_count=args.rl_target_count,
        seed=args.seed,
        upload_hf=args.upload_hf,
        sync_drive=args.sync_drive,
    )
    print("\n=== Data Pipeline Execution Report ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
