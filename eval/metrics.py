"""Automatic evaluation metrics for chess explanations.

This module calculates rule check metrics and soft scores for model completions.
It aggregates hard failure rates, mean rewards, verdict accuracy, and length stats.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

from rewards.api import (
    THINK_BLOCK_RE,
    RewardResult,
    parse_prompt,
    score_completion,
    verdict_accuracy,
)
from rewards.scorer import (
    V2_WEIGHT_BETTER_MOVE,
    V2_WEIGHT_KEY_POINT,
    V2_WEIGHT_LEVEL_FIT,
    V2_WEIGHT_NO_REPEATS,
    V2_WEIGHT_NOT_COPIED,
    WEIGHT_BETTER_MOVE,
    WEIGHT_KEY_POINT,
    WEIGHT_LENGTH,
    WEIGHT_LEVEL_FIT,
    WEIGHT_NO_REPEATS,
    WEIGHT_NOT_COPIED,
)
from eval.concept_tags import CONCEPT_TIERS, classify_concept_tier

LOGGER = logging.getLogger(__name__)

HARD_CHECK_NAMES: tuple[str, ...] = (
    "legal_moves",
    "grounded_moves",
    "piece_on_square",
    "verdict_tone",
    "format",
    "beginner_jargon",
    "coach_voice",
    "opening_claims",
    "relational_claims",
)

HARD_CHECK_NAMES_V1: tuple[str, ...] = HARD_CHECK_NAMES[:6]

HARD_CHECK_NAMES_V3: tuple[str, ...] = HARD_CHECK_NAMES + (
    "vague_boilerplate",
    "positional_grounding",
)


def get_hard_check_names(reward_version: str = "v1") -> tuple[str, ...]:
    """Return ordered hard check names matching the specified reward version."""
    if reward_version == "v1":
        return HARD_CHECK_NAMES_V1
    if reward_version == "v3":
        return HARD_CHECK_NAMES_V3
    return HARD_CHECK_NAMES

SOFT_SCORE_NAMES: tuple[str, ...] = (
    "key_point",
    "better_move",
    "length",
    "level_fit",
    "not_copied",
    "no_repeats",
)

SOFT_SCORE_WEIGHTS: dict[str, float] = {
    "key_point": WEIGHT_KEY_POINT,
    "better_move": WEIGHT_BETTER_MOVE,
    "length": WEIGHT_LENGTH,
    "level_fit": WEIGHT_LEVEL_FIT,
    "not_copied": WEIGHT_NOT_COPIED,
    "no_repeats": WEIGHT_NO_REPEATS,
}

SOFT_SCORE_WEIGHTS_V2: dict[str, float] = {
    "key_point": V2_WEIGHT_KEY_POINT,
    "better_move": V2_WEIGHT_BETTER_MOVE,
    "level_fit": V2_WEIGHT_LEVEL_FIT,
    "not_copied": V2_WEIGHT_NOT_COPIED,
    "no_repeats": V2_WEIGHT_NO_REPEATS,
}


def get_soft_score_weights(reward_version: str = "v1") -> dict[str, float]:
    """Return canonical soft score weights for the requested reward version."""
    if reward_version in ("v2", "v3"):
        return dict(SOFT_SCORE_WEIGHTS_V2)
    return dict(SOFT_SCORE_WEIGHTS)


@dataclass(frozen=True)
class ItemMetric:
    """Evaluation result for one test set completion.

    The dataclass stores identifiers, task settings, check outcomes, and text length.
    """

    id: str
    task: str
    level: str
    detail: str
    word_count: int
    in_length_band: bool
    passed_all_hard: bool
    failed_checks: list[str]
    hard_checks: dict[str, bool]
    soft_scores: dict[str, float]
    reward: float
    verdict: str | None
    verdict_accurate: bool | None
    verdict_covered: bool
    verdict_tone_passed: bool
    completion: str
    concept_tier: str | None = None

    def composite_soft_score(self, reward_version: str = "v1") -> float:
        """Compute weighted sum of soft scores using canonical reward weights."""
        weights = get_soft_score_weights(reward_version)
        total = sum(
            weight * self.soft_scores.get(name, 0.0)
            for name, weight in weights.items()
        )
        if reward_version in ("v2", "v3"):
            total *= self.soft_scores.get("length_multiplier", 1.0)
        return total

    def to_dict(self) -> dict[str, Any]:
        """Convert item metric to a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class AggregateMetrics:
    """Summary metrics across a full evaluation set.

    The dataclass stores rates as percentages from 0.0 to 100.0.
    It stores soft scores and rewards in their native ranges.
    """

    total_items: int
    hard_fail_rate: float
    hard_check_fail_rates: dict[str, float]
    mean_reward: float
    mean_soft_score_all: float
    mean_soft_score_passing: float
    soft_score_breakdown: dict[str, float]
    avg_word_count_overall: float
    avg_word_count_short: float
    avg_word_count_long: float
    pct_in_band_overall: float
    pct_in_band_short: float
    pct_in_band_long: float
    verdict_items_count: int
    verdict_coverage_rate: float
    verdict_accuracy_all: float
    verdict_accuracy_covered: float
    verdict_tone_pass_rate: float
    breakdowns: dict[str, dict[str, dict[str, float]]]

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate metrics to a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class EvalMetrics:
    """Container holding aggregate metrics and per-item results."""

    aggregate: AggregateMetrics
    items: list[ItemMetric]

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation metrics container to a plain dictionary."""
        return {
            "aggregate": self.aggregate.to_dict(),
            "items": [item.to_dict() for item in self.items],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize evaluation metrics to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def clean_completion_text(completion: str) -> str:
    """Strip thinking blocks and outer whitespace from completion text.

    If completion contains a think block, remove the think block.
    Return the trimmed completion string.
    """
    if not completion:
        return ""
    return THINK_BLOCK_RE.sub("", completion).strip()


def count_words(text: str) -> int:
    """Count whitespace-separated words in text.

    If text is empty, return zero.
    Return the total number of words.
    """
    if not text:
        return 0
    return len(text.split())


def is_in_length_band(detail: str, word_count: int) -> bool:
    """Check if word count is inside the target range for detail mode.

    When detail is short, the band is 25 to 80 words.
    When detail is long, the band is 80 to 200 words.
    """
    norm_detail = detail.strip().lower()
    if norm_detail == "long":
        return 80 <= word_count <= 200
    return 25 <= word_count <= 80


def ensure_prompt_text(item: dict[str, Any]) -> str:
    """Extract or construct prompt text from an item record.

    If user_message or prompt exists, return that text.
    Otherwise build a minimal valid prompt string from item fields.
    """
    user_msg = item.get("user_message")
    if isinstance(user_msg, str) and user_msg.strip():
        return user_msg.strip()

    prompt_val = item.get("prompt")
    if isinstance(prompt_val, str) and prompt_val.strip():
        return prompt_val.strip()

    task = str(item.get("task", "explain_move"))
    level = str(item.get("level", "club"))
    detail = str(item.get("detail", "short"))
    fen = str(item.get("fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    move_played = item.get("move_played") or item.get("move_uci") or ""
    verdict = item.get("verdict", "")

    lines: list[str] = [
        f"task: {task}",
        f"level: {level}",
        f"detail: {detail}",
        "",
        f"FEN: {fen}",
    ]
    if task == "explain_move" and move_played:
        lines.append(f"Move played: {move_played}.")
    if verdict:
        lines.append(f"Verdict: {verdict}.")
    lines.append("Facts:")

    facts = item.get("facts")
    if isinstance(facts, list) and facts:
        for f in facts:
            lines.append(f"- {f}")
    else:
        note = item.get("note")
        if note:
            lines.append(f"- {note}")
        else:
            lines.append("- Material: equal.")

    return "\n".join(lines)


def evaluate_item(
    item: dict[str, Any],
    completion: str | None = None,
    reward_version: str = "v1",
) -> ItemMetric:
    """Evaluate one model completion against item prompt and rules.

    If completion argument is None, read completion or output from item dict.
    Calculate hard checks, soft scores, word counts, and verdict agreement.
    Return an ItemMetric instance.
    """
    item_id = str(item.get("id", "unknown"))
    raw_completion = completion
    if raw_completion is None:
        raw_completion = str(item.get("completion") or item.get("output") or item.get("text") or "")

    cleaned = clean_completion_text(raw_completion)
    prompt_str = ensure_prompt_text(item)

    # Parse metadata from prompt or item
    try:
        info = parse_prompt(prompt_str)
        task = info.task
        level = info.level
        detail = info.detail
        verdict = info.verdict
    except Exception as exc:
        LOGGER.warning("Failed to parse prompt for item %s: %s", item_id, exc)
        task = str(item.get("task", "explain_move"))
        level = str(item.get("level", "club"))
        detail = str(item.get("detail", "short"))
        verdict = str(item.get("verdict")) if item.get("verdict") else None

    check_names = get_hard_check_names(reward_version)

    # Score completion with rewards system
    try:
        reward_result: RewardResult = score_completion(
            prompt_str,
            raw_completion,
            reward_version=reward_version,
        )
        passed_all_hard = reward_result.passed_all_hard
        hard_checks = {k: v.passed for k, v in reward_result.hard_checks.items()}
        failed_checks = [k for k, v in reward_result.hard_checks.items() if not v.passed]
        soft_scores = dict(reward_result.soft_scores)
        reward = reward_result.reward
    except Exception as exc:
        LOGGER.warning("Reward scoring failed for item %s: %s", item_id, exc)
        passed_all_hard = False
        hard_checks = {name: False for name in check_names}
        failed_checks = list(check_names)
        soft_scores = {name: 0.0 for name in SOFT_SCORE_NAMES}
        reward = -1.0

    words = count_words(cleaned)
    in_band = is_in_length_band(detail, words)

    # Calculate verdict metrics
    verdict_accurate: bool | None = None
    verdict_covered = False
    verdict_tone_passed = hard_checks.get("verdict_tone", True)

    if task == "explain_move" and verdict:
        acc_result = verdict_accuracy(cleaned, verdict)
        verdict_accurate = acc_result
        verdict_covered = acc_result is not None

    concept_tier = item.get("concept_tier")
    if not concept_tier:
        try:
            concept_tier = classify_concept_tier(item)
        except Exception:
            concept_tier = "tier_4_positional"

    return ItemMetric(
        id=item_id,
        task=task,
        level=level,
        detail=detail,
        word_count=words,
        in_length_band=in_band,
        passed_all_hard=passed_all_hard,
        failed_checks=failed_checks,
        hard_checks=hard_checks,
        soft_scores=soft_scores,
        reward=reward,
        verdict=verdict,
        verdict_accurate=verdict_accurate,
        verdict_covered=verdict_covered,
        verdict_tone_passed=verdict_tone_passed,
        completion=raw_completion,
        concept_tier=concept_tier,
    )


def _compute_slice_stats(items: list[ItemMetric]) -> dict[str, float]:
    """Calculate summary statistics for a slice of items.

    Return count, hard fail rate, mean reward, and average word count.
    """
    total = len(items)
    if total == 0:
        return {
            "count": 0.0,
            "hard_fail_rate": 0.0,
            "mean_reward": 0.0,
            "avg_word_count": 0.0,
            "pct_in_band": 0.0,
        }

    fails = sum(1 for it in items if not it.passed_all_hard)
    mean_rew = sum(it.reward for it in items) / total
    avg_words = sum(it.word_count for it in items) / total
    in_band_count = sum(1 for it in items if it.in_length_band)

    return {
        "count": float(total),
        "hard_fail_rate": round(100.0 * fails / total, 2),
        "mean_reward": round(mean_rew, 4),
        "avg_word_count": round(avg_words, 2),
        "pct_in_band": round(100.0 * in_band_count / total, 2),
    }


def aggregate_metrics(
    item_metrics: list[ItemMetric],
    reward_version: str = "v1",
) -> AggregateMetrics:
    """Aggregate individual item metrics into set-level metrics.

    Calculate hard fail rates, soft score averages, and group breakdowns.
    Return an AggregateMetrics instance.
    """
    check_names = get_hard_check_names(reward_version)
    total = len(item_metrics)
    if total == 0:
        empty_check_rates = {k: 0.0 for k in check_names}
        empty_soft = {k: 0.0 for k in SOFT_SCORE_NAMES}
        return AggregateMetrics(
            total_items=0,
            hard_fail_rate=0.0,
            hard_check_fail_rates=empty_check_rates,
            mean_reward=0.0,
            mean_soft_score_all=0.0,
            mean_soft_score_passing=0.0,
            soft_score_breakdown=empty_soft,
            avg_word_count_overall=0.0,
            avg_word_count_short=0.0,
            avg_word_count_long=0.0,
            pct_in_band_overall=0.0,
            pct_in_band_short=0.0,
            pct_in_band_long=0.0,
            verdict_items_count=0,
            verdict_coverage_rate=0.0,
            verdict_accuracy_all=0.0,
            verdict_accuracy_covered=0.0,
            verdict_tone_pass_rate=0.0,
            breakdowns={},
        )

    # 1. Hard check failure rates
    hard_fail_count = sum(1 for it in item_metrics if not it.passed_all_hard)
    hard_fail_rate = round(100.0 * hard_fail_count / total, 2)

    hard_check_fail_rates: dict[str, float] = {}
    for check_name in check_names:
        fails = sum(1 for it in item_metrics if not it.hard_checks.get(check_name, True))
        hard_check_fail_rates[check_name] = round(100.0 * fails / total, 2)

    # 2. Reward and soft scores
    mean_reward = round(sum(it.reward for it in item_metrics) / total, 4)

    passing_items = [it for it in item_metrics if it.passed_all_hard]

    mean_soft_all = round(
        sum(it.composite_soft_score(reward_version) for it in item_metrics) / total,
        4,
    )
    if passing_items:
        mean_soft_passing = round(
            sum(it.composite_soft_score(reward_version) for it in passing_items) / len(passing_items),
            4,
        )
    else:
        mean_soft_passing = 0.0

    soft_breakdown: dict[str, float] = {}
    for soft_name in SOFT_SCORE_NAMES:
        avg_score = sum(it.soft_scores.get(soft_name, 0.0) for it in item_metrics) / total
        soft_breakdown[soft_name] = round(avg_score, 4)

    # 3. Word counts and bands
    short_items = [it for it in item_metrics if it.detail == "short"]
    long_items = [it for it in item_metrics if it.detail == "long"]

    avg_words_overall = round(sum(it.word_count for it in item_metrics) / total, 2)
    avg_words_short = (
        round(sum(it.word_count for it in short_items) / len(short_items), 2)
        if short_items
        else 0.0
    )
    avg_words_long = (
        round(sum(it.word_count for it in long_items) / len(long_items), 2)
        if long_items
        else 0.0
    )

    in_band_all = sum(1 for it in item_metrics if it.in_length_band)
    in_band_short = sum(1 for it in short_items if it.in_length_band)
    in_band_long = sum(1 for it in long_items if it.in_length_band)

    pct_band_overall = round(100.0 * in_band_all / total, 2)
    pct_band_short = round(100.0 * in_band_short / len(short_items), 2) if short_items else 0.0
    pct_band_long = round(100.0 * in_band_long / len(long_items), 2) if long_items else 0.0

    # 4. Verdict accuracy metrics
    verdict_items = [
        it for it in item_metrics
        if it.task == "explain_move" and it.verdict
    ]
    verdict_count = len(verdict_items)

    if verdict_count > 0:
        covered_count = sum(1 for it in verdict_items if it.verdict_covered)
        accurate_count = sum(1 for it in verdict_items if it.verdict_accurate is True)
        tone_passed_count = sum(1 for it in verdict_items if it.verdict_tone_passed)

        verdict_coverage_rate = round(100.0 * covered_count / verdict_count, 2)
        verdict_accuracy_all = round(100.0 * accurate_count / verdict_count, 2)
        verdict_accuracy_covered = (
            round(100.0 * accurate_count / covered_count, 2)
            if covered_count > 0
            else 0.0
        )
        verdict_tone_pass_rate = round(100.0 * tone_passed_count / verdict_count, 2)
    else:
        verdict_coverage_rate = 0.0
        verdict_accuracy_all = 0.0
        verdict_accuracy_covered = 0.0
        verdict_tone_pass_rate = 0.0

    # 5. Group breakdowns
    breakdowns: dict[str, dict[str, dict[str, float]]] = {
        "level": {
            "beginner": _compute_slice_stats([it for it in item_metrics if it.level == "beginner"]),
            "club": _compute_slice_stats([it for it in item_metrics if it.level == "club"]),
        },
        "task": {
            "explain_move": _compute_slice_stats([it for it in item_metrics if it.task == "explain_move"]),
            "explain_position": _compute_slice_stats([it for it in item_metrics if it.task == "explain_position"]),
        },
        "detail": {
            "short": _compute_slice_stats(short_items),
            "long": _compute_slice_stats(long_items),
        },
        "concept_tier": {
            tier: _compute_slice_stats([it for it in item_metrics if it.concept_tier == tier])
            for tier in CONCEPT_TIERS
            if any(it.concept_tier == tier for it in item_metrics)
        },
    }

    return AggregateMetrics(
        total_items=total,
        hard_fail_rate=hard_fail_rate,
        hard_check_fail_rates=hard_check_fail_rates,
        mean_reward=mean_reward,
        mean_soft_score_all=mean_soft_all,
        mean_soft_score_passing=mean_soft_passing,
        soft_score_breakdown=soft_breakdown,
        avg_word_count_overall=avg_words_overall,
        avg_word_count_short=avg_words_short,
        avg_word_count_long=avg_words_long,
        pct_in_band_overall=pct_band_overall,
        pct_in_band_short=pct_band_short,
        pct_in_band_long=pct_band_long,
        verdict_items_count=verdict_count,
        verdict_coverage_rate=verdict_coverage_rate,
        verdict_accuracy_all=verdict_accuracy_all,
        verdict_accuracy_covered=verdict_accuracy_covered,
        verdict_tone_pass_rate=verdict_tone_pass_rate,
        breakdowns=breakdowns,
    )


def evaluate_completions(
    records: list[dict[str, Any]],
    reward_version: str = "v1",
) -> EvalMetrics:
    """Evaluate a collection of prompt completion records.

    Each record must have prompt or user_message, and completion or output.
    Calculate per-item metrics and aggregate summary statistics.
    Return an EvalMetrics container.
    """
    items = [evaluate_item(rec, reward_version=reward_version) for rec in records]
    agg = aggregate_metrics(items, reward_version=reward_version)
    return EvalMetrics(aggregate=agg, items=items)


def evaluate_file(
    path: str | Path,
    reward_version: str = "v1",
) -> EvalMetrics:
    """Load completion records from a file and calculate metrics.

    Support both newline-delimited JSONL and standard JSON list formats.
    Return an EvalMetrics container.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Evaluation file not found: {p}")

    records: list[dict[str, Any]] = []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return evaluate_completions([], reward_version=reward_version)

    if text.startswith("["):
        loaded = json.loads(text)
        if isinstance(loaded, list):
            records = loaded
    else:
        for line in text.splitlines():
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    return evaluate_completions(records, reward_version=reward_version)
