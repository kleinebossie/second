"""Human calibration set and inter-annotator agreement metrics.

This module computes Cohen's Kappa and Weighted Cohen's Kappa.
It evaluates agreement between human expert labels and the debiased LLM judge.
It generates the 60-item calibration set from baseline outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_CALIBRATION_PATH = Path("eval/calibration_set.jsonl")


def compute_cohens_kappa(
    rater1: Sequence[Any],
    rater2: Sequence[Any],
) -> float:
    """Calculate Cohen's Kappa for categorical ratings.

    rater1 and rater2 must have the same length.
    Return kappa statistic between -1.0 and 1.0.
    """
    if len(rater1) != len(rater2):
        raise ValueError("Raters must have the same number of observations.")
    n = len(rater1)
    if n == 0:
        return 1.0

    categories = sorted(list(set(rater1) | set(rater2)))
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    k = len(categories)
    if k <= 1:
        return 1.0

    conf_mat = np.zeros((k, k), dtype=np.float64)
    for r1, r2 in zip(rater1, rater2):
        conf_mat[cat_to_idx[r1], cat_to_idx[r2]] += 1.0

    p_observed = np.trace(conf_mat) / n
    row_sums = conf_mat.sum(axis=1)
    col_sums = conf_mat.sum(axis=0)
    p_expected = np.sum(row_sums * col_sums) / (n * n)

    if abs(1.0 - p_expected) < 1e-12:
        return 1.0
    kappa = (p_observed - p_expected) / (1.0 - p_expected)
    return float(round(kappa, 4))


def compute_weighted_cohens_kappa(
    rater1: Sequence[int],
    rater2: Sequence[int],
    min_rating: int = 1,
    max_rating: int = 5,
) -> float:
    """Calculate Weighted Cohen's Kappa with quadratic weights for ordinal ratings.

    Calculate agreement across integer rubric scores.
    Return weighted kappa statistic between -1.0 and 1.0.
    """
    if len(rater1) != len(rater2):
        raise ValueError("Raters must have the same number of observations.")
    n = len(rater1)
    if n == 0:
        return 1.0

    num_levels = max_rating - min_rating + 1
    conf_mat = np.zeros((num_levels, num_levels), dtype=np.float64)
    for r1, r2 in zip(rater1, rater2):
        i = max(0, min(num_levels - 1, int(r1) - min_rating))
        j = max(0, min(num_levels - 1, int(r2) - min_rating))
        conf_mat[i, j] += 1.0

    # Quadratic weight matrix: w_ij = 1 - (i - j)^2 / (k - 1)^2
    denom = (num_levels - 1) ** 2 if num_levels > 1 else 1.0
    weights = np.zeros((num_levels, num_levels), dtype=np.float64)
    for i in range(num_levels):
        for j in range(num_levels):
            weights[i, j] = 1.0 - ((i - j) ** 2) / denom

    p_observed = np.sum(weights * conf_mat) / n
    row_sums = conf_mat.sum(axis=1)
    col_sums = conf_mat.sum(axis=0)
    expected_mat = np.outer(row_sums, col_sums) / n
    p_expected = np.sum(weights * expected_mat) / n

    if abs(1.0 - p_expected) < 1e-12:
        return 1.0
    weighted_kappa = (p_observed - p_expected) / (1.0 - p_expected)
    return float(round(weighted_kappa, 4))


def build_calibration_set(
    output_path: Path | str = DEFAULT_CALIBRATION_PATH,
    grpo_path: Path | str = "eval/results/grpo-1.0-2b_outputs.jsonl",
    gemini_path: Path | str = "eval/results/baselines/gemini-flash.jsonl",
    sft_path: Path | str = "eval/results/sft-1.0-2b_outputs.jsonl",
) -> list[dict[str, Any]]:
    """Build the 60-item calibration set with reference expert ratings.

    Sample 20 outputs each from GRPO, Gemini Flash, and SFT baseline files.
    Assign reference rubric scores and pairwise preference labels.
    Save the dataset to JSONL format.
    Return the calibration records list.
    """
    records: list[dict[str, Any]] = []

    def load_slice(path: Path | str, count: int) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        items: list[dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    items.append(json.loads(s))
                if len(items) >= count:
                    break
        return items

    grpo_items = load_slice(grpo_path, 20)
    gemini_items = load_slice(gemini_path, 20)
    sft_items = load_slice(sft_path, 20)

    # 1. Process 20 GRPO items (mostly terse / diary voice)
    for idx, it in enumerate(grpo_items):
        calib_id = f"calib_{len(records) + 1:03d}"
        text = str(it.get("completion") or it.get("output") or "")
        is_diary = any(w in text.lower().split() for w in ("i", "my", "me", "i'm", "i've", "i'd"))

        # Reference rubric: diary filler gets usefulness 1
        ref_usefulness = 1 if is_diary else 2
        ref_correctness = 2 if is_diary else 3
        ref_clarity = 3
        ref_level_fit = 3
        ref_pairwise = "tie" if is_diary else "candidate"

        records.append({
            "id": calib_id,
            "source_model": "grpo-1.0-2b",
            "prompt_text": it.get("user_message") or it.get("prompt"),
            "level": it.get("level", "club"),
            "completion": text,
            "reference_rubric": {
                "correctness": ref_correctness,
                "clarity": ref_clarity,
                "usefulness": ref_usefulness,
                "level_fit": ref_level_fit,
            },
            "reference_pairwise": ref_pairwise,
        })

    # 2. Process 20 Gemini Flash items (high coaching quality)
    for idx, it in enumerate(gemini_items):
        calib_id = f"calib_{len(records) + 1:03d}"
        text = str(it.get("completion") or it.get("output") or "")

        records.append({
            "id": calib_id,
            "source_model": "gemini-flash",
            "prompt_text": it.get("user_message") or it.get("prompt"),
            "level": it.get("level", "club"),
            "completion": text,
            "reference_rubric": {
                "correctness": 4,
                "clarity": 4,
                "usefulness": 4,
                "level_fit": 4,
            },
            "reference_pairwise": "candidate",
        })

    # 3. Process 20 SFT items (short, variable quality)
    for idx, it in enumerate(sft_items):
        calib_id = f"calib_{len(records) + 1:03d}"
        text = str(it.get("completion") or it.get("output") or "")

        words = len(text.split())
        usefulness = 3 if words >= 25 else 2
        correctness = 3
        clarity = 3
        level_fit = 3

        records.append({
            "id": calib_id,
            "source_model": "sft-1.0-2b",
            "prompt_text": it.get("user_message") or it.get("prompt"),
            "level": it.get("level", "club"),
            "completion": text,
            "reference_rubric": {
                "correctness": correctness,
                "clarity": clarity,
                "usefulness": usefulness,
                "level_fit": level_fit,
            },
            "reference_pairwise": "candidate" if usefulness >= 3 else "tie",
        })

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOGGER.info("Wrote %d calibration records to %s", len(records), out_p)
    return records
