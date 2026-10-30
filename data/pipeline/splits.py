"""Game-level dataset splitting module.

This module splits chess dataset records by game.
It guarantees that positions from one game never appear in multiple splits.
It guarantees that no position FEN appears in both train and evaluation splits.
It produces train, validation, and test pool partitions.
It exports standard datasets and public redistributable subsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_VAL_RATIO = 0.02
DEFAULT_TEST_RATIO = 0.08
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_PUBLIC_DIR = Path("data/processed/public")


def hash_game_id(game_id: str, seed: int = 42) -> float:
    """Compute a deterministic hash value in range zero to one for a game identifier.

    The function uses SHA-256 for stability across platforms and runs.
    """
    key = f"{game_id}_{seed}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # Use first 8 hex characters (32 bits)
    val = int(digest[:8], 16)
    return val / 0xFFFFFFFF


def split_by_game(
    records: list[dict[str, Any]],
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split dataset records into train, val, and test pool splits.

    This function groups records by game identifier.
    It assigns each game to a single split deterministically.
    It resolves position collisions so no FEN appears in multiple splits.
    If a game shares a position with train, the function assigns the game to train.
    """
    if not records:
        return {"train": [], "val": [], "test_pool": []}

    # Group records and unique FENs by game_id
    games_records: dict[str, list[dict[str, Any]]] = {}
    game_fens: dict[str, set[str]] = {}

    for i, rec in enumerate(records):
        gid = str(rec.get("game_id") or f"unassigned_game_{i}")
        if gid not in games_records:
            games_records[gid] = []
            game_fens[gid] = set()
        games_records[gid].append(rec)
        fen = rec.get("fen")
        if fen:
            game_fens[gid].add(fen)

    # Initial deterministic assignment by game_id hash
    game_splits: dict[str, str] = {}
    for gid in sorted(games_records.keys()):
        h = hash_game_id(gid, seed=seed)
        if h < val_ratio:
            game_splits[gid] = "val"
        elif h < (val_ratio + test_ratio):
            game_splits[gid] = "test_pool"
        else:
            game_splits[gid] = "train"

    # Resolve cross-split FEN collisions while keeping val and test_pool ratios intact:
    # 1. Games stay in their hash-assigned split (no game is ever split across partitions).
    # 2. If a specific position FEN appears in both train and val/test_pool (e.g., 1. e4 e5),
    #    drop that common-opening record from val/test_pool so evaluation only tests unseen positions!
    # 3. If a specific position FEN appears in both val and test_pool, drop that record from test_pool.
    train_fens: set[str] = set()
    for gid, split in game_splits.items():
        if split == "train":
            train_fens.update(game_fens[gid])

    val_fens: set[str] = set()
    for gid, split in game_splits.items():
        if split == "val":
            val_fens.update(game_fens[gid] - train_fens)

    # Build final split lists
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test_pool": [],
    }
    for gid, split_name in game_splits.items():
        for rec in games_records[gid]:
            fen = rec.get("fen")
            if split_name == "val" and fen in train_fens:
                continue
            if split_name == "test_pool" and (fen in train_fens or fen in val_fens):
                continue
            splits[split_name].append(rec)

    # Verify invariants
    train_fens_final = {r["fen"] for r in splits["train"] if "fen" in r}
    val_fens_final = {r["fen"] for r in splits["val"] if "fen" in r}
    test_fens_final = {r["fen"] for r in splits["test_pool"] if "fen" in r}

    overlap_train_test = train_fens_final & test_fens_final
    overlap_train_val = train_fens_final & val_fens_final
    overlap_val_test = val_fens_final & test_fens_final

    if overlap_train_test or overlap_train_val or overlap_val_test:
        raise RuntimeError(
            f"FEN overlap detected between splits! "
            f"train/test: {len(overlap_train_test)}, "
            f"train/val: {len(overlap_train_val)}, "
            f"val/test: {len(overlap_val_test)}"
        )

    return splits


def _write_records(
    records: list[dict[str, Any]],
    jsonl_path: Path,
    parquet_path: Path,
) -> None:
    """Write records to JSONL and Parquet files.

    Create parent directories automatically.
    Write JSONL line by line with UTF-8 encoding.
    Write Parquet table using pandas.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(parquet_path, index=False)
    else:
        empty_df = pd.DataFrame()
        empty_df.to_parquet(parquet_path, index=False)


def save_splits(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    public_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Save split records to disk in JSONL and Parquet formats.

    Write full splits to the output directory.
    Write redistributable records to the public directory.
    Return a summary dictionary with counts.
    """
    out_path = Path(output_dir)
    pub_path = Path(public_dir) if public_dir else out_path / "public"

    stats: dict[str, Any] = {
        "total_records": 0,
        "splits": {},
        "public_splits": {},
    }

    for split_name in ("train", "val", "test_pool"):
        recs = splits.get(split_name, [])
        stats["total_records"] += len(recs)
        stats["splits"][split_name] = len(recs)

        jsonl_file = out_path / f"sft_{split_name}.jsonl"
        parquet_file = out_path / f"sft_{split_name}.parquet"
        _write_records(recs, jsonl_file, parquet_file)

        # Public subset: only records where redistributable is True
        public_recs = [r for r in recs if r.get("redistributable") is True]
        stats["public_splits"][split_name] = len(public_recs)

        pub_jsonl = pub_path / f"sft_{split_name}.jsonl"
        pub_parquet = pub_path / f"sft_{split_name}.parquet"
        _write_records(public_recs, pub_jsonl, pub_parquet)

    LOGGER.info(
        "Saved splits: train=%d, val=%d, test_pool=%d (public: train=%d, val=%d, test_pool=%d)",
        stats["splits"]["train"],
        stats["splits"]["val"],
        stats["splits"]["test_pool"],
        stats["public_splits"]["train"],
        stats["public_splits"]["val"],
        stats["public_splits"]["test_pool"],
    )
    return stats


def main() -> None:
    """Command line entry point for splitting datasets."""
    parser = argparse.ArgumentParser(description="Split chess dataset records by game.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file of processed records.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Output directory for full splits.")
    parser.add_argument("--public-dir", type=str, default=str(DEFAULT_PUBLIC_DIR), help="Output directory for public splits.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation set split ratio.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test pool split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic hashing.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_file = Path(args.input)
    if not input_file.exists():
        LOGGER.error("Input file not found: %s", input_file)
        sys.exit(1)

    records: list[dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    LOGGER.info("Loaded %d records from %s", len(records), input_file)
    splits = split_by_game(
        records,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_splits(splits, output_dir=args.output_dir, public_dir=args.public_dir)


if __name__ == "__main__":
    main()
