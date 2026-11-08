"""Unit tests for calibration module and agreement metrics.

This module tests Cohen's Kappa, Weighted Cohen's Kappa, and calibration set loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.calibration import (
    DEFAULT_CALIBRATION_PATH,
    compute_cohens_kappa,
    compute_weighted_cohens_kappa,
)


def test_cohens_kappa_perfect_agreement() -> None:
    """Verify kappa is 1.0 when two raters agree completely."""
    r1 = ["A", "B", "A", "tie", "B"]
    r2 = ["A", "B", "A", "tie", "B"]
    kappa = compute_cohens_kappa(r1, r2)
    assert kappa == 1.0


def test_cohens_kappa_disagreement() -> None:
    """Verify kappa is low or negative when raters disagree."""
    r1 = ["A"] * 20 + ["B"] * 20
    r2 = ["B"] * 20 + ["A"] * 20
    kappa = compute_cohens_kappa(r1, r2)
    assert kappa < 0.0


def test_weighted_cohens_kappa_perfect() -> None:
    """Verify weighted kappa is 1.0 on identical ordinal scores."""
    r1 = [1, 2, 3, 4, 5, 2, 3, 4]
    r2 = [1, 2, 3, 4, 5, 2, 3, 4]
    kw = compute_weighted_cohens_kappa(r1, r2, min_rating=1, max_rating=5)
    assert kw == 1.0


def test_weighted_cohens_kappa_distance_penalty() -> None:
    """Verify that off-by-one errors have higher kappa than off-by-four errors."""
    target = [1, 2, 3, 4, 5] * 10
    off_by_one = [2, 3, 4, 5, 4] * 10
    off_by_four = [5, 5, 1, 1, 1] * 10

    kw_near = compute_weighted_cohens_kappa(target, off_by_one)
    kw_far = compute_weighted_cohens_kappa(target, off_by_four)
    assert kw_near > kw_far


def test_calibration_set_structure() -> None:
    """Verify that calibration_set.jsonl contains 60 valid items."""
    assert DEFAULT_CALIBRATION_PATH.exists()
    items = []
    with open(DEFAULT_CALIBRATION_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    assert len(items) == 60
    for it in items:
        assert "id" in it
        assert "source_model" in it
        assert "prompt_text" in it
        assert "completion" in it
        assert "reference_rubric" in it
        assert "reference_pairwise" in it
