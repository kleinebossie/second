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
from .filter_comments import (
    FilterStats,
    assign_detail_and_truncate,
    filter_comment,
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
    cache_path: Path | str | None = None,
    drive_dir: Path | str | None = None,
    stockfish_path: str | Path | None = None,
    depth: int = 12,
    num_workers: int = 4,
    rl_target_count: int = 20000,
    filter_sft: bool = True,
    upload_hf: bool = False,
    sync_drive: bool = False,
) -> dict[str, Any]:
    """Execute the end-to-end data pipeline.

    Verify the Stockfish executable binary.
    Load and process supervised fine-tuning records.
    Evaluate positions and populate facts.
    Split records by game into train, validation, and test pool partitions.
    Generate reinforcement learning prompt sets.
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
        filter_stats = FilterStats()
        seen_fps: set[str] = set()
        pre_filtered_sft: list[tuple[dict[str, Any], str]] = []

        if filter_sft:
            from langdetect import detect as _detect_lang
            from .filter_comments import (
                HANDLE_RE,
                LINK_RE,
                clean_comment,
                compute_fingerprint,
                has_chess_content,
                has_offensive_content,
            )
            from rewards.checks import check_legal_moves, check_piece_on_square

            for rec in sft_records:
                filter_stats.total_input += 1
                comment = rec.get("comment", "")
                fen = rec.get("fen", "")
                move_uci = rec.get("move_uci")

                cleaned = clean_comment(comment)
                if len(cleaned.split()) < 8:
                    filter_stats.cleaned_empty_or_short += 1
                    continue

                try:
                    if _detect_lang(cleaned) != "en":
                        filter_stats.not_english += 1
                        continue
                except Exception:
                    filter_stats.not_english += 1
                    continue

                if not has_chess_content(cleaned):
                    filter_stats.no_chess_content += 1
                    continue

                try:
                    b_check = chess.Board(fen)
                    if not b_check.is_valid():
                        filter_stats.illegal_move_or_wrong_piece += 1
                        continue
                    if move_uci:
                        m_check = chess.Move.from_uci(move_uci)
                        if m_check not in b_check.legal_moves:
                            filter_stats.illegal_move_or_wrong_piece += 1
                            continue
                except Exception:
                    filter_stats.illegal_move_or_wrong_piece += 1
                    continue

                if not check_piece_on_square(text=cleaned, fen=fen, move_uci=move_uci).passed:
                    filter_stats.illegal_move_or_wrong_piece += 1
                    continue

                if LINK_RE.search(cleaned) or HANDLE_RE.search(cleaned) or has_offensive_content(cleaned):
                    filter_stats.links_usernames_offensive += 1
                    continue

                fp = compute_fingerprint(cleaned)
                if fp in seen_fps:
                    filter_stats.duplicate += 1
                    continue

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

                fp = compute_fingerprint(cleaned)
                if fp in seen_fps:
                    filter_stats.duplicate += 1
                    continue
                seen_fps.add(fp)
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

    # Step 2: Process RL prompts from CC0 games
    rl_candidates: list[dict[str, Any]] = []
    if rl_input_path and Path(rl_input_path).exists():
        with open(rl_input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rl_candidates.append(json.loads(line))
        LOGGER.info("Loaded %d raw RL candidates from %s", len(rl_candidates), rl_input_path)

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

    # Step 3: Prepare Hugging Face export package
    exp_dir = prepare_hf_export(public_dir=pub_path)
    report["export_directory"] = str(exp_dir)

    # Step 4: Optional Hugging Face upload
    if upload_hf:
        LOGGER.info("Uploading dataset to Hugging Face Hub")
        hf_status = upload_to_hf(export_dir=exp_dir)
        report["hf_upload"] = hf_status

    # Step 5: Optional Google Drive sync
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
    parser.add_argument("--cache-path", type=str, default=None, help="Path to Stockfish Parquet cache file.")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive sync directory.")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish executable.")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth.")
    parser.add_argument("--workers", type=int, default=4, help="Stockfish worker count.")
    parser.add_argument("--rl-target-count", type=int, default=20000, help="Target number of RL prompts.")
    parser.add_argument("--upload-hf", action="store_true", help="Upload public files to Hugging Face Hub.")
    parser.add_argument("--sync-drive", action="store_true", help="Synchronize files to Google Drive.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    report = run_pipeline(
        sft_input_path=args.sft_input,
        rl_input_path=args.rl_input,
        output_dir=args.output_dir,
        public_dir=args.public_dir,
        cache_path=args.cache_path,
        drive_dir=args.drive_dir,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        num_workers=args.workers,
        rl_target_count=args.rl_target_count,
        upload_hf=args.upload_hf,
        sync_drive=args.sync_drive,
    )
    print("\n=== Data Pipeline Execution Report ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
