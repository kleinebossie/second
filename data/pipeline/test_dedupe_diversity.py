"""Unit tests for data/pipeline/dedupe_diversity.py.

All test functions follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from data.pipeline.dedupe_diversity import (
    build_sft_v2_dataset,
    compute_ngrams,
    extract_prefix,
    jaccard_similarity,
    prune_by_diversity,
    rescue_human_data,
)


def test_extract_prefix() -> None:
    """Verify prefix extraction returns first n lowercase words."""
    text = "Playing e4 is a strong move that controls the center."
    prefix = extract_prefix(text, word_count=5)
    assert prefix == "playing e4 is a strong"


def test_compute_ngrams_and_jaccard() -> None:
    """Verify n-gram generation and Jaccard similarity between phrases."""
    words_a = ["playing", "e4", "controls", "the", "center"]
    words_b = ["playing", "e4", "controls", "the", "board"]
    ngrams_a = compute_ngrams(words_a, n=3)
    ngrams_b = compute_ngrams(words_b, n=3)

    assert len(ngrams_a) == 3
    assert len(ngrams_b) == 3

    sim = jaccard_similarity(ngrams_a, ngrams_b)
    # ngrams_a: ('playing', 'e4', 'controls'), ('e4', 'controls', 'the'), ('controls', 'the', 'center')
    # ngrams_b: ('playing', 'e4', 'controls'), ('e4', 'controls', 'the'), ('controls', 'the', 'board')
    # intersection: 2, union: 4 -> 0.5
    assert sim == 0.5


def test_prune_by_diversity_prefix_cap() -> None:
    """Verify diversity pruning caps duplicate opening prefixes."""
    repetitive_items = [
        {"comment": f"Playing e4 is a strong move in variation {i}."}
        for i in range(10)
    ]
    diverse_items = [
        {"comment": f"In this position Black develops the bishop to square {i}."}
        for i in range(10)
    ]

    records = repetitive_items + diverse_items
    pruned, metrics = prune_by_diversity(
        records,
        max_prefix_pct=0.20,  # Max 4 items per prefix
        jaccard_threshold=0.95,
    )

    # 4 repetitive items kept, 6 repetitive items pruned; 4 diverse kept, 6 diverse pruned
    assert len(pruned) == 8
    assert metrics["pruned_by_prefix_cap"] == 12


def test_rescue_human_data(tmp_path: Path) -> None:
    """Verify human data rescue filters comments on length, voice, and rules."""
    sft_test_file = tmp_path / "sft_sample.jsonl"

    good_comment = (
        "White develops the knight to f3 to attack the black pawn on e5 and control d4. "
        "Material remains equal in this quiet opening position. "
        "Black should respond with Nc6 to protect the pawn safely."
    )
    first_person_comment = (
        "I played Nf3 because I wanted to attack e5. My plan worked well and my opponent struggled."
    )
    short_comment = "Nf3 attacks e5."

    entries = [
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "verdict": "Good",
            "move_uci": "g1f3",
            "facts": ["After Nf3, Black's best reply is Nc6.", "Material: equal.", "Phase: opening."],
            "user_message": "task: explain_move\nMove played: Nf3.",
            "comment": good_comment,
        },
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "verdict": "Good",
            "move_uci": "g1f3",
            "facts": ["After Nf3, Black's best reply is Nc6."],
            "user_message": "task: explain_move\nMove played: Nf3.",
            "comment": first_person_comment,
        },
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "verdict": "Good",
            "move_uci": "g1f3",
            "facts": ["After Nf3, Black's best reply is Nc6."],
            "user_message": "task: explain_move\nMove played: Nf3.",
            "comment": short_comment,
        },
    ]

    with open(sft_test_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    rescued = rescue_human_data(sft_test_file, min_words=20, min_key_point=0.5)
    assert len(rescued) == 1
    assert rescued[0]["source_origin"] == "human_gold_rescued"
    assert rescued[0]["comment"] == good_comment
    assert rescued[0]["thinking_mode"] == "off"
    assert rescued[0]["assistant_response"].startswith("<think>\n\n</think>")
