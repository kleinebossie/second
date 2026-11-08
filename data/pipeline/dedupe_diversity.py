"""Diversity deduplication and human data rescue module for Stage T2.5.

This module re-filters human commentary from SFT v1 using strict Stage T1 verification.
It combines verified human data with verified teacher synthesis data.
It enforces n-gram and prefix diversity so no opening phrase dominates the dataset.
It saves the unified SFT v2 training dataset.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess

from rewards.checks import check_coach_voice
from rewards.scorer import _tokenize_words, compute_reward

LOGGER = logging.getLogger(__name__)

DEFAULT_SFT_TRAIN_PATH = Path("data/processed/sft_train.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/sft_v2_train.jsonl")
DEFAULT_MAX_PREFIX_PCT = 0.015  # No 5-word prefix exceeds 1.5%
DEFAULT_JACCARD_THRESHOLD = 0.80


def extract_prefix(text: str, word_count: int = 5) -> str:
    """Extract normalized opening word prefix from an explanation text.

    Tokenize text into lowercase alphanumeric words.
    Return joined prefix string of length word_count.
    """
    words = _tokenize_words(text)
    if not words:
        return ""
    return " ".join(words[:word_count])


def compute_ngrams(words: list[str], n: int = 3) -> set[tuple[str, ...]]:
    """Compute n-gram tuples from a list of words.

    Return set of n-gram tuples.
    """
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set_a: set[Any], set_b: set[Any]) -> float:
    """Compute Jaccard similarity between two sets.

    Return intersection divided by union.
    If both sets are empty, return 0.0.
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def rescue_human_data(
    sft_train_path: Path | str = DEFAULT_SFT_TRAIN_PATH,
    min_words: int = 30,
    min_key_point: float = 0.50,
) -> list[dict[str, Any]]:
    """Filter human commentary records with strict Stage T1 verifier.

    Check explanation word count against min_words.
    Verify coach voice with zero first-person pronouns.
    Verify all hard rule checks pass under Stage T1 reward v2.
    Verify key point score meets min_key_point.
    Return list of verified gold human records.
    """
    in_file = Path(sft_train_path)
    if not in_file.is_file() and (PROJECT_ROOT / in_file).is_file():
        in_file = PROJECT_ROOT / in_file

    rescued: list[dict[str, Any]] = []
    if not in_file.is_file():
        LOGGER.warning("Human SFT file not found: %s", in_file)
        return rescued

    total_scanned = 0
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            total_scanned += 1
            rec = json.loads(line_str)
            comment = rec.get("comment", "")
            words = _tokenize_words(comment)

            if len(words) < min_words:
                continue

            voice_res = check_coach_voice(comment)
            if not voice_res.passed:
                continue

            reward_res = compute_reward(
                text=comment,
                fen=rec.get("fen", chess.STARTING_FEN),
                level=rec.get("level", "club"),
                detail=rec.get("detail", "short"),
                verdict=rec.get("verdict"),
                move_uci=rec.get("move_uci"),
                facts=rec.get("facts", []),
                prompt_text=rec.get("user_message", ""),
                reward_version="v2",
            )

            if not reward_res.passed_all_hard:
                continue

            if reward_res.soft_scores.get("key_point", 0.0) < min_key_point:
                continue

            updated_rec = dict(rec)
            updated_rec["source_origin"] = "human_gold_rescued"
            updated_rec["teacher_reward"] = reward_res.reward
            if "thinking_mode" not in updated_rec:
                updated_rec["thinking_mode"] = "off"
            if "assistant_response" not in updated_rec:
                updated_rec["assistant_response"] = f"<think>\n\n</think>\n\n{comment}"
            rescued.append(updated_rec)

    LOGGER.info(
        "Rescued %d gold human comments out of %d scanned (%.1f%%)",
        len(rescued),
        total_scanned,
        (len(rescued) / total_scanned * 100) if total_scanned else 0.0,
    )
    return rescued


def prune_by_diversity(
    records: list[dict[str, Any]],
    max_prefix_pct: float = DEFAULT_MAX_PREFIX_PCT,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prune repetitive explanations using opening prefixes and n-gram similarity.

    Limit identical 5-word opening prefixes to max_prefix_pct of the dataset.
    Prune near-duplicate explanations with 3-gram Jaccard similarity above threshold.
    Use an inverted index over 3-grams for fast sub-linear deduplication.
    Return tuple of (pruned_records, pruning_metrics).
    """
    total_in = len(records)
    if total_in == 0:
        return [], {"total_in": 0, "kept": 0}

    max_prefix_count = max(1, int(total_in * max_prefix_pct))

    kept_records: list[dict[str, Any]] = []
    prefix_counts: Counter[str] = Counter()
    pruned_prefix_count = 0
    pruned_jaccard_count = 0

    kept_ngrams: list[set[tuple[str, ...]]] = []
    ngram_index: dict[tuple[str, ...], list[int]] = defaultdict(list)

    for item in records:
        text = item.get("comment") or item.get("explanation") or ""
        words = _tokenize_words(text)
        prefix = " ".join(words[:5]) if len(words) >= 5 else " ".join(words)

        # 1. Check opening prefix cap
        if prefix and prefix_counts[prefix] >= max_prefix_count:
            pruned_prefix_count += 1
            continue

        # 2. Check near-duplicate Jaccard similarity via inverted index
        item_ngrams = compute_ngrams(words, n=3)
        candidate_indices: set[int] = set()
        for ng in item_ngrams:
            if ng in ngram_index:
                candidate_indices.update(ngram_index[ng])

        is_duplicate = False
        for c_idx in candidate_indices:
            sim = jaccard_similarity(item_ngrams, kept_ngrams[c_idx])
            if sim >= jaccard_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            pruned_jaccard_count += 1
            continue

        prefix_counts[prefix] += 1
        new_idx = len(kept_records)
        kept_ngrams.append(item_ngrams)
        for ng in item_ngrams:
            ngram_index[ng].append(new_idx)
        kept_records.append(item)

    metrics = {
        "total_in": total_in,
        "kept_records": len(kept_records),
        "pruned_by_prefix_cap": pruned_prefix_count,
        "pruned_by_jaccard_similarity": pruned_jaccard_count,
        "max_prefix_allowed": max_prefix_count,
        "unique_prefixes": len(prefix_counts),
    }

    LOGGER.info("Diversity pruning metrics: %s", json.dumps(metrics, indent=2))
    return kept_records, metrics


def build_sft_v2_dataset(
    teacher_records: list[dict[str, Any]],
    human_records: list[dict[str, Any]] | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    max_prefix_pct: float = DEFAULT_MAX_PREFIX_PCT,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> list[dict[str, Any]]:
    """Combine teacher records with rescued human comments and apply diversity pruning.

    Merge teacher records and human records.
    Run diversity pruning.
    Save dataset to output JSONL.
    Return list of final SFT v2 training records.
    """
    human = human_records or []
    combined = list(teacher_records) + list(human)

    pruned, metrics = prune_by_diversity(
        records=combined,
        max_prefix_pct=max_prefix_pct,
        jaccard_threshold=jaccard_threshold,
    )

    out_file = Path(output_path)
    if not out_file.is_absolute() and not out_file.parent.exists():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in pruned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOGGER.info("Saved SFT v2 dataset with %d records to %s", len(pruned), out_file)
    return pruned


def main() -> None:
    """CLI entry point for diversity deduplication and human data rescue."""
    parser = argparse.ArgumentParser(
        description="Rescue gold human comments and apply diversity deduplication for SFT v2."
    )
    parser.add_argument(
        "--teacher-data",
        type=str,
        default="data/processed/teacher_verified_non_thinking.jsonl",
        help="Path to verified teacher completions JSONL",
    )
    parser.add_argument(
        "--human-data",
        type=str,
        default=str(DEFAULT_SFT_TRAIN_PATH),
        help="Path to SFT v1 training data JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for sft_v2_train.jsonl",
    )
    parser.add_argument(
        "--max-prefix-pct",
        type=float,
        default=DEFAULT_MAX_PREFIX_PCT,
        help="Maximum dataset share allowed per opening 5-word prefix",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    teacher_path = Path(args.teacher_data)
    if not teacher_path.is_file() and (PROJECT_ROOT / teacher_path).is_file():
        teacher_path = PROJECT_ROOT / teacher_path

    teacher_records: list[dict[str, Any]] = []
    if teacher_path.is_file():
        with open(teacher_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    teacher_records.append(json.loads(line))

    rescued_human = rescue_human_data(args.human_data)
    final_records = build_sft_v2_dataset(
        teacher_records=teacher_records,
        human_records=rescued_human,
        output_path=args.output,
        max_prefix_pct=args.max_prefix_pct,
    )
    print(f"SFT v2 build complete. Produced {len(final_records)} training records.")


if __name__ == "__main__":
    main()
