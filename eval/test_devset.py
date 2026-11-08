"""Tests for the Phase 6.5 stratified development set.

This module validates eval/devset-1.0.jsonl and eval/build_devset.py.
It checks total count, stratification mix, level balance, and concept tier tags.
It verifies zero contamination against testset-1.0.jsonl and all training splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.pipeline.rl_prompts import load_excluded_keys
from data.pipeline.splits import normalize_fen
from eval.build_devset import (
    DEV_LONG_BAD,
    DEV_LONG_GOOD,
    DEV_LONG_POS,
    DEV_SHORT_BAD,
    DEV_SHORT_GOOD,
    DEV_SHORT_POS,
    verify_devset_zero_contamination,
)
from eval.build_testset import BAD_VERDICTS, GOOD_VERDICTS
from eval.concept_tags import CONCEPT_TIERS

DEVSET_PATH = Path("eval/devset-1.0.jsonl")
TESTSET_PATH = Path("eval/testset-1.0.jsonl")
SPLIT_DIR = Path("data/processed")


@pytest.fixture(scope="module")
def devset_records() -> list[dict[str, Any]]:
    """Load records from the dev set file."""
    assert DEVSET_PATH.exists(), f"Dev set file not found at {DEVSET_PATH}"
    records: list[dict[str, Any]] = []
    with open(DEVSET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def test_devset_count(devset_records: list[dict[str, Any]]) -> None:
    """Verify that devset contains exactly 150 items."""
    assert len(devset_records) == 150


def test_devset_stratification(devset_records: list[dict[str, Any]]) -> None:
    """Verify exact counts across task, detail, and verdict buckets."""
    short_bad = [
        r for r in devset_records
        if r["task"] == "explain_move" and r["detail"] == "short" and r["verdict"] in BAD_VERDICTS
    ]
    short_good = [
        r for r in devset_records
        if r["task"] == "explain_move" and r["detail"] == "short" and r["verdict"] in GOOD_VERDICTS
    ]
    short_pos = [
        r for r in devset_records
        if r["task"] == "explain_position" and r["detail"] == "short" and r["verdict"] is None
    ]
    long_items = [r for r in devset_records if r["detail"] == "long"]

    assert len(short_bad) == DEV_SHORT_BAD       # 60
    assert len(short_good) == DEV_SHORT_GOOD     # 45
    assert len(short_pos) == DEV_SHORT_POS       # 22
    assert len(long_items) == 23

    long_bad = [r for r in long_items if r["task"] == "explain_move" and r["verdict"] in BAD_VERDICTS]
    long_good = [r for r in long_items if r["task"] == "explain_move" and r["verdict"] in GOOD_VERDICTS]
    long_pos = [r for r in long_items if r["task"] == "explain_position"]

    assert len(long_bad) == DEV_LONG_BAD         # 11
    assert len(long_good) == DEV_LONG_GOOD       # 8
    assert len(long_pos) == DEV_LONG_POS         # 4


def test_devset_levels_balance(devset_records: list[dict[str, Any]]) -> None:
    """Verify that dev set contains exactly 75 beginner and 75 club items."""
    beg_count = sum(1 for r in devset_records if r["level"] == "beginner")
    club_count = sum(1 for r in devset_records if r["level"] == "club")

    assert beg_count == 75
    assert club_count == 75


def test_devset_concept_tiers(devset_records: list[dict[str, Any]]) -> None:
    """Verify that every dev set item has a valid concept_tier."""
    tier_counts: dict[str, int] = {}
    for r in devset_records:
        assert "concept_tier" in r
        tier = r["concept_tier"]
        assert tier in CONCEPT_TIERS
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Verify multiple tiers are represented
    assert len(tier_counts) >= 3


def test_devset_zero_contamination(devset_records: list[dict[str, Any]]) -> None:
    """Verify zero overlap with testset and all training splits."""
    verify_devset_zero_contamination(devset_records, testset_path=TESTSET_PATH, split_dir=SPLIT_DIR)
