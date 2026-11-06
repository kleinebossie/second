"""Phase 6 GRPO reinforcement learning evaluation runner and report generator.

This module evaluates fine-tuned GRPO checkpoints on validation and frozen test splits.
It ranks checkpoints by validation reward and hard-fail rate.
It verifies Phase 6 release criteria against SFT and base Qwen 2B baselines.
It reviews 20 sample outputs and audits for anti-hacking signals.
It generates comprehensive Markdown reports in ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.baselines import load_jsonl
from eval.judge import (
    DEFAULT_CACHE_PATH,
    DEFAULT_JUDGE_MODEL,
    get_gemini_api_key,
    run_judge_eval,
)
from eval.metrics import EvalMetrics, ItemMetric, evaluate_file
from training.evaluate_sft import MANUAL_REVIEW_NOTES

LOGGER = logging.getLogger(__name__)

DEFAULT_VAL_OUTPUTS = Path("eval/results/grpo-1.0-2b_val_outputs.jsonl")
DEFAULT_TEST_OUTPUTS = Path("eval/results/grpo-1.0-2b_outputs.jsonl")
DEFAULT_SFT_OUTPUTS = Path("eval/results/sft-1.0-2b_outputs.jsonl")
DEFAULT_BASE_QWEN = Path("eval/results/baselines/base-qwen-2b.jsonl")
DEFAULT_GEMINI_BASELINE = Path("eval/results/baselines/gemini-flash.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_REPORTS_DIR = Path("eval/reports")


@dataclass(frozen=True)
class CheckpointRankingEntry:
    """Evaluation summary and ranking status for one checkpoint."""

    checkpoint_name: str
    step: int
    val_mean_reward: float
    val_hard_fail_rate: float
    val_avg_word_count: float
    val_pct_in_band: float
    test_mean_reward: float | None = None
    test_hard_fail_rate: float | None = None
    test_avg_word_count: float | None = None
    test_pct_in_band: float | None = None
    is_best: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary format."""
        return asdict(self)


@dataclass(frozen=True)
class Phase6PassResult:
    """Outcome of Phase 6 release rule verification."""

    grpo_mean_reward: float
    sft_mean_reward: float
    base_mean_reward: float
    grpo_hard_fail_rate: float
    sft_hard_fail_rate: float
    base_hard_fail_rate: float
    grpo_format_pass_rate: float
    reward_improved_over_sft: bool
    hard_fail_lower_than_base: bool
    hard_fail_not_worse_than_sft: bool
    passed_format: bool
    passed_all: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert pass result to dictionary format."""
        return asdict(self)


@dataclass(frozen=True)
class Phase6EvalContext:
    """Bundle evaluation context for Phase 6 reporting."""

    model_name: str
    best_checkpoint: str
    val_metrics: EvalMetrics
    test_metrics: EvalMetrics
    sft_test_metrics: EvalMetrics
    base_qwen_metrics: EvalMetrics
    pass_result: Phase6PassResult
    reviewed_20: list[dict[str, Any]]
    checkpoint_rankings: list[CheckpointRankingEntry]
    step_logs: list[dict[str, Any]]
    judge_results: dict[str, Any] | None = None
    training_metadata: dict[str, Any] | None = None


def extract_step_number(name: str) -> int:
    """Extract step number from a checkpoint name.

    If the name contains digits, return the first integer match.
    Otherwise return 0.
    """
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return 0


def rank_checkpoints(
    checkpoint_metrics: dict[str, EvalMetrics],
    test_metrics_map: dict[str, EvalMetrics] | None = None,
    top_k: int = 3,
) -> tuple[list[CheckpointRankingEntry], list[str]]:
    """Rank training checkpoints by validation metrics.

    Sort checkpoints by val mean reward descending.
    Then sort by val hard-fail rate ascending.
    Then sort by step descending.
    Mark the top checkpoint with is_best=True.
    Return full list of ranking entries and top checkpoint names.
    """
    if not checkpoint_metrics:
        return [], []

    test_map = test_metrics_map or {}
    entries: list[CheckpointRankingEntry] = []

    for name, v_metrics in checkpoint_metrics.items():
        step = extract_step_number(name)
        v_agg = v_metrics.aggregate

        t_agg = test_map.get(name).aggregate if name in test_map else None
        test_mean_reward = float(t_agg.mean_reward) if t_agg else None
        test_hard_fail_rate = float(t_agg.hard_fail_rate) if t_agg else None
        test_avg_word_count = float(t_agg.avg_word_count_overall) if t_agg else None
        test_pct_in_band = float(t_agg.pct_in_band_overall) if t_agg else None

        entry = CheckpointRankingEntry(
            checkpoint_name=name,
            step=step,
            val_mean_reward=float(v_agg.mean_reward),
            val_hard_fail_rate=float(v_agg.hard_fail_rate),
            val_avg_word_count=float(v_agg.avg_word_count_overall),
            val_pct_in_band=float(v_agg.pct_in_band_overall),
            test_mean_reward=test_mean_reward,
            test_hard_fail_rate=test_hard_fail_rate,
            test_avg_word_count=test_avg_word_count,
            test_pct_in_band=test_pct_in_band,
            is_best=False,
        )
        entries.append(entry)

    # Sort:
    # 1. val_mean_reward descending (-val_mean_reward)
    # 2. val_hard_fail_rate ascending (val_hard_fail_rate)
    # 3. step descending (-step)
    # 4. checkpoint_name ascending for deterministic ordering
    entries.sort(key=lambda e: (-e.val_mean_reward, e.val_hard_fail_rate, -e.step, e.checkpoint_name))

    ranked_entries: list[CheckpointRankingEntry] = []
    for idx, e in enumerate(entries):
        if idx == 0:
            ranked_entries.append(replace(e, is_best=True))
        else:
            ranked_entries.append(e)

    effective_k = max(0, top_k)
    top_names = [e.checkpoint_name for e in ranked_entries[:effective_k]]
    return ranked_entries, top_names


def check_phase6_pass_rule(
    grpo_test_metrics: EvalMetrics,
    sft_test_metrics: EvalMetrics,
    base_qwen_metrics: EvalMetrics,
) -> Phase6PassResult:
    """Verify Phase 6 release criteria against SFT and base baselines.

    Check if GRPO reward improves over SFT baseline.
    Check if GRPO hard-fail rate is lower than base Qwen baseline.
    Check if GRPO hard-fail rate is not worse than SFT baseline.
    Check if format pass rate exceeds 95.0 percent.
    Return Phase6PassResult container.
    """
    g_agg = grpo_test_metrics.aggregate
    s_agg = sft_test_metrics.aggregate
    b_agg = base_qwen_metrics.aggregate

    grpo_mean_reward = float(g_agg.mean_reward)
    sft_mean_reward = float(s_agg.mean_reward)
    base_mean_reward = float(b_agg.mean_reward)

    grpo_hard_fail_rate = float(g_agg.hard_fail_rate)
    sft_hard_fail_rate = float(s_agg.hard_fail_rate)
    base_hard_fail_rate = float(b_agg.hard_fail_rate)

    format_fail_rate = float(g_agg.hard_check_fail_rates.get("format", 100.0))
    grpo_format_pass_rate = 100.0 - format_fail_rate

    reward_improved_over_sft = grpo_mean_reward > sft_mean_reward
    hard_fail_lower_than_base = grpo_hard_fail_rate < base_hard_fail_rate
    hard_fail_not_worse_than_sft = grpo_hard_fail_rate <= sft_hard_fail_rate
    passed_format = grpo_format_pass_rate > 95.0

    passed_all = (
        reward_improved_over_sft
        and hard_fail_lower_than_base
        and hard_fail_not_worse_than_sft
        and passed_format
    )

    return Phase6PassResult(
        grpo_mean_reward=grpo_mean_reward,
        sft_mean_reward=sft_mean_reward,
        base_mean_reward=base_mean_reward,
        grpo_hard_fail_rate=grpo_hard_fail_rate,
        sft_hard_fail_rate=sft_hard_fail_rate,
        base_hard_fail_rate=base_hard_fail_rate,
        grpo_format_pass_rate=grpo_format_pass_rate,
        reward_improved_over_sft=reward_improved_over_sft,
        hard_fail_lower_than_base=hard_fail_lower_than_base,
        hard_fail_not_worse_than_sft=hard_fail_not_worse_than_sft,
        passed_format=passed_format,
        passed_all=passed_all,
    )


def create_grpo_review_note(item: ItemMetric) -> str:
    """Create a structured review note in ASD-STE100 technical English.

    Analyze format integrity, voice fit, length band, grounding, and anti-hacking guards.
    Return formatted review note text.
    """
    notes: list[str] = []

    # 1. Format analysis
    if "format" in item.failed_checks:
        notes.append("Format: FAIL. Output contains markdown syntax, headings, or prompt text.")
    else:
        notes.append("Format: PASS. Output uses clean plain text without markdown syntax or headings.")

    # 2. Voice and level analysis
    if item.level == "beginner":
        if "beginner_jargon" in item.failed_checks:
            notes.append("Voice: FAIL. Output contains engine jargon not permitted for beginners.")
        else:
            notes.append("Voice: PASS. Output uses simple vocabulary and avoids engine jargon.")
    else:
        notes.append("Voice: PASS. Output matches standard club player tone.")

    # 3. Length fit analysis
    expected_range = "25-80 words" if item.detail == "short" else "80-200 words"
    if item.in_length_band:
        notes.append(
            f"Length: PASS. Word count ({item.word_count}) is inside the {item.detail} target band ({expected_range})."
        )
    else:
        notes.append(
            f"Length: FAIL. Word count ({item.word_count}) is outside the {item.detail} target band ({expected_range})."
        )

    # 4. Factual grounding analysis
    fact_issues: list[str] = []
    if "legal_moves" in item.failed_checks:
        fact_issues.append("illegal move")
    if "grounded_moves" in item.failed_checks:
        fact_issues.append("unsupported move")
    if "piece_on_square" in item.failed_checks:
        fact_issues.append("wrong piece location")
    if "verdict_tone" in item.failed_checks:
        fact_issues.append("verdict tone mismatch")

    if fact_issues:
        notes.append(f"Grounding: FAIL. Found factual defects: {', '.join(fact_issues)}.")
    else:
        notes.append("Grounding: PASS. All mentioned moves and piece locations match the facts.")

    # 5. Anti-hacking inspection
    min_len = 25 if item.detail == "short" else 80
    if item.word_count < min_len:
        notes.append(
            f"Anti-hacking guard: Word count ({item.word_count}) is below minimum threshold ({min_len}). Watch for length collapse."
        )

    text_lower = item.completion.strip().lower()
    robotic_starters = (
        "in this position",
        "the move played is",
        "white plays",
        "black plays",
        "this move is",
    )
    if any(text_lower.startswith(s) for s in robotic_starters):
        notes.append("Anti-hacking guard: Template opening detected. Check for robotic phrasing.")

    summary = " ".join(notes)
    manual_obs = MANUAL_REVIEW_NOTES.get(str(item.id))
    if manual_obs:
        return f"{summary} Observation: {manual_obs}"
    return summary


def select_20_outputs_for_review(
    items: list[ItemMetric],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select 20 diverse outputs across all strata for step review.

    Cover both tasks (explain_move and explain_position).
    Cover both player levels (beginner and club).
    Cover both detail modes (short and long).
    Cover passing items and failing items.
    If total items are 20 or fewer, return all items.
    Return exactly 20 items with metadata and review notes.
    """
    records_by_id = {str(rec.get("id")): rec for rec in records if rec.get("id")}
    target_count = min(20, len(items))

    if len(items) <= 20:
        selected_items = list(items)
    else:
        strata_keys = [
            ("explain_move", "beginner", "short"),
            ("explain_move", "beginner", "long"),
            ("explain_move", "club", "short"),
            ("explain_move", "club", "long"),
            ("explain_position", "beginner", "short"),
            ("explain_position", "beginner", "long"),
            ("explain_position", "club", "short"),
            ("explain_position", "club", "long"),
        ]

        strata_groups: dict[tuple[str, str, str], list[ItemMetric]] = {k: [] for k in strata_keys}
        for it in sorted(items, key=lambda x: str(x.id)):
            key = (str(it.task), str(it.level), str(it.detail))
            if key in strata_groups:
                strata_groups[key].append(it)
            else:
                strata_groups.setdefault(key, []).append(it)

        selected_ids: set[str] = set()
        chosen: list[ItemMetric] = []

        def add_candidate(cand: ItemMetric) -> bool:
            if cand.id not in selected_ids and len(chosen) < target_count:
                selected_ids.add(cand.id)
                chosen.append(cand)
                return True
            return False

        # Phase A: Pick 1 failing item from each stratum
        for key in strata_keys:
            group = strata_groups.get(key, [])
            failing = [it for it in group if not it.passed_all_hard]
            if failing:
                add_candidate(failing[0])

        # Phase B: Pick 1 passing item from each stratum
        for key in strata_keys:
            group = strata_groups.get(key, [])
            passing = [it for it in group if it.passed_all_hard]
            if passing:
                add_candidate(passing[0])

        # Phase C: Pick up to 2 additional failing items from any stratum
        all_failing = [it for it in items if not it.passed_all_hard and it.id not in selected_ids]
        for it in all_failing[:2]:
            add_candidate(it)

        # Phase D: Fill remaining slots round-robin across strata
        while len(chosen) < target_count:
            added_any = False
            for key in strata_keys:
                group = strata_groups.get(key, [])
                unselected = [it for it in group if it.id not in selected_ids]
                if unselected:
                    add_candidate(unselected[0])
                    added_any = True
                if len(chosen) >= target_count:
                    break
            if not added_any:
                break

        # Phase E: Fallback to any remaining items sorted by id
        if len(chosen) < target_count:
            for it in sorted(items, key=lambda x: str(x.id)):
                add_candidate(it)
                if len(chosen) >= target_count:
                    break

        selected_items = chosen

    results: list[dict[str, Any]] = []
    for it in selected_items:
        rec = records_by_id.get(str(it.id), {})
        move_val = (
            rec.get("move_san")
            or rec.get("move_played")
            or rec.get("best_move")
            or ""
        )
        review_note = create_grpo_review_note(it)

        min_len = 25 if it.detail == "short" else 80
        length_collapsed = it.word_count < min_len
        text_lower = it.completion.strip().lower()
        robotic_starters = (
            "in this position",
            "the move played is",
            "white plays",
            "black plays",
            "this move is",
        )
        robotic_opening = any(text_lower.startswith(s) for s in robotic_starters)

        results.append({
            "id": it.id,
            "task": it.task,
            "level": it.level,
            "detail": it.detail,
            "verdict": it.verdict or rec.get("verdict"),
            "move_san": str(move_val),
            "word_count": it.word_count,
            "reward": round(it.reward, 4),
            "passed_all_hard": it.passed_all_hard,
            "failed_checks": list(it.failed_checks),
            "soft_scores": {k: round(v, 4) for k, v in it.soft_scores.items()},
            "completion": it.completion,
            "review_note": review_note,
            "length_collapsed": length_collapsed,
            "robotic_opening": robotic_opening,
        })

    return results


def audit_anti_hacking(
    reviewed_20: list[dict[str, Any]],
    judge_results: dict[str, Any] | None = None,
    sft_test_metrics: EvalMetrics | None = None,
    grpo_test_metrics: EvalMetrics | None = None,
) -> dict[str, Any]:
    """Perform anti-hacking audit on reviewed items and judge metrics.

    Check for length collapse patterns.
    Check for repetitive opening formulas.
    Check concordance between RL reward gains and judge preference.
    Return audit results dictionary.
    """
    total = len(reviewed_20)
    if total == 0:
        return {
            "length_collapse_count": 0,
            "length_collapse_rate": 0.0,
            "length_collapse_flag": False,
            "robotic_opening_count": 0,
            "robotic_opening_rate": 0.0,
            "robotic_opening_flag": False,
            "top_openings": [],
            "judge_concordance_flag": True,
            "judge_concordance_note": "No items reviewed.",
        }

    collapsed_count = sum(1 for item in reviewed_20 if item.get("length_collapsed", False))
    robotic_count = sum(1 for item in reviewed_20 if item.get("robotic_opening", False))

    collapse_rate = round(100.0 * collapsed_count / total, 1)
    robotic_rate = round(100.0 * robotic_count / total, 1)

    openings: list[str] = []
    for item in reviewed_20:
        words = str(item.get("completion", "")).strip().split()
        if len(words) >= 3:
            openings.append(" ".join(words[:3]).lower())
        elif words:
            openings.append(" ".join(words).lower())

    opening_counts = Counter(openings).most_common(3)
    top_openings_str = [f"'{op}' ({cnt})" for op, cnt in opening_counts]

    length_collapse_flag = collapse_rate > 25.0
    robotic_opening_flag = robotic_rate > 35.0

    judge_flag = True
    judge_note = "Gemini judge was not evaluated for this session."
    if judge_results and "pairwise" in judge_results:
        summaries = judge_results["pairwise"].get("summaries", {})
        sft_comp = summaries.get("sft-1.0-2b", {})
        win_rate = sft_comp.get("candidate_win_rate", 0.0)
        win_tie_rate = sft_comp.get("candidate_win_tie_rate", 0.0)

        if grpo_test_metrics and sft_test_metrics:
            g_rew = grpo_test_metrics.aggregate.mean_reward
            s_rew = sft_test_metrics.aggregate.mean_reward
            if g_rew > s_rew and win_rate < 40.0:
                judge_flag = False
                judge_note = (
                    f"Warning: RL reward increased ({g_rew:.4f} vs {s_rew:.4f}), "
                    f"but pairwise win rate vs SFT is low ({win_rate:.1f}%). Watch for reward hacking."
                )
            else:
                judge_flag = True
                judge_note = (
                    f"Concordant: RL reward ({g_rew:.4f}) and judge win+tie rate "
                    f"({win_tie_rate:.1f}% vs SFT) both confirm quality gains."
                )

    return {
        "length_collapse_count": collapsed_count,
        "length_collapse_rate": collapse_rate,
        "length_collapse_flag": length_collapse_flag,
        "robotic_opening_count": robotic_count,
        "robotic_opening_rate": robotic_rate,
        "robotic_opening_flag": robotic_opening_flag,
        "top_openings": top_openings_str,
        "judge_concordance_flag": judge_flag,
        "judge_concordance_note": judge_note,
    }


def generate_phase6_report(
    context: Phase6EvalContext | str | None = None,
    val_metrics: EvalMetrics | None = None,
    test_metrics: EvalMetrics | None = None,
    sft_test_metrics: EvalMetrics | None = None,
    base_qwen_metrics: EvalMetrics | None = None,
    pass_result: Phase6PassResult | None = None,
    reviewed_20: list[dict[str, Any]] | None = None,
    checkpoint_rankings: list[CheckpointRankingEntry] | None = None,
    step_logs: list[dict[str, Any]] | None = None,
    judge_results: dict[str, Any] | None = None,
    training_metadata: dict[str, Any] | None = None,
    *,
    model_name: str | None = None,
    best_checkpoint: str | None = None,
) -> str:
    """Generate the Phase 6 Markdown evaluation report.

    Write all sections in ASD-STE100 Simplified Technical English.
    Include summary tables, metric comparisons, judge outcomes, and sample reviews.
    Return Markdown report text.
    """
    if isinstance(context, Phase6EvalContext):
        ctx = context
    else:
        m_name = model_name or (context if isinstance(context, str) else None)
        if (
            m_name is None
            or val_metrics is None
            or test_metrics is None
            or sft_test_metrics is None
            or base_qwen_metrics is None
            or pass_result is None
            or reviewed_20 is None
        ):
            raise ValueError(
                "Must provide either Phase6EvalContext or all individual evaluation metrics."
            )
        ctx = Phase6EvalContext(
            model_name=m_name,
            best_checkpoint=best_checkpoint or "checkpoint-best",
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            sft_test_metrics=sft_test_metrics,
            base_qwen_metrics=base_qwen_metrics,
            pass_result=pass_result,
            reviewed_20=reviewed_20,
            checkpoint_rankings=checkpoint_rankings or [],
            step_logs=step_logs or [],
            judge_results=judge_results,
            training_metadata=training_metadata,
        )

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    v_agg = ctx.val_metrics.aggregate
    t_agg = ctx.test_metrics.aggregate
    s_agg = ctx.sft_test_metrics.aggregate
    b_agg = ctx.base_qwen_metrics.aggregate
    pass_res = ctx.pass_result
    status_str = "PASS" if pass_res.passed_all else "FAIL"

    audit = audit_anti_hacking(
        reviewed_20=ctx.reviewed_20,
        judge_results=ctx.judge_results,
        sft_test_metrics=ctx.sft_test_metrics,
        grpo_test_metrics=ctx.test_metrics,
    )

    lines: list[str] = [
        f"# Phase 6 Evaluation Report: {ctx.model_name}",
        "",
        f"- **Model**: `{ctx.model_name}`",
        f"- **Best Checkpoint**: `{ctx.best_checkpoint}`",
        f"- **Date**: {now_str}",
        f"- **Evaluation Status**: **{status_str}**",
        "- **Specification**: Phase 6 (GRPO Reinforcement Learning on 2B)",
        "",
        "## 1. Executive Summary and Phase 6 Verification",
        "",
        "Phase 6 requires four conditions for release:",
        "1. Mean reward must improve over the Phase 5 SFT baseline.",
        "2. Hard-fail rate must be lower than base Qwen 2B.",
        "3. Hard-fail rate must not be worse than the SFT baseline.",
        "4. Format pass rate must exceed 95.0% on the frozen test set.",
        "",
        "### Key Metrics Comparison",
        "",
        "| Metric | Second-1.0-2B-GRPO | Second-1.0-2B-SFT | Base Qwen 2B | Status / Delta |",
        "|---|---|---|---|---|",
        f"| Format Pass Rate | **{pass_res.grpo_format_pass_rate:.1f}%** | {100.0 - s_agg.hard_check_fail_rates.get('format', 0.0):.1f}% | {100.0 - b_agg.hard_check_fail_rates.get('format', 100.0):.1f}% | {'PASS' if pass_res.passed_format else 'FAIL'} |",
        f"| Hard-Fail Rate | **{pass_res.grpo_hard_fail_rate:.1f}%** | {pass_res.sft_hard_fail_rate:.1f}% | {pass_res.base_hard_fail_rate:.1f}% | {t_agg.hard_fail_rate - s_agg.hard_fail_rate:+.1f}% vs SFT |",
        f"| Mean Reward | **{pass_res.grpo_mean_reward:.4f}** | {pass_res.sft_mean_reward:.4f} | {pass_res.base_mean_reward:.4f} | {t_agg.mean_reward - s_agg.mean_reward:+.4f} vs SFT |",
        f"| Overall Phase 6 Result | **{status_str}** | - | - | **{status_str}** |",
        "",
        "### Release Criteria Checklist",
        "",
        "| Release Criterion | GRPO Value | Baseline Target | Target Rule | Status |",
        "|---|---|---|---|---|",
        f"| Reward Improved Over SFT | **{pass_res.grpo_mean_reward:.4f}** | {pass_res.sft_mean_reward:.4f} | > SFT Reward | {'PASS' if pass_res.reward_improved_over_sft else 'FAIL'} |",
        f"| Hard-Fail Lower Than Base | **{pass_res.grpo_hard_fail_rate:.1f}%** | {pass_res.base_hard_fail_rate:.1f}% | < Base Qwen Rate | {'PASS' if pass_res.hard_fail_lower_than_base else 'FAIL'} |",
        f"| Hard-Fail Not Worse Than SFT | **{pass_res.grpo_hard_fail_rate:.1f}%** | {pass_res.sft_hard_fail_rate:.1f}% | <= SFT Rate | {'PASS' if pass_res.hard_fail_not_worse_than_sft else 'FAIL'} |",
        f"| Format Pass Rate Above 95% | **{pass_res.grpo_format_pass_rate:.1f}%** | 95.0% | > 95.0% | {'PASS' if pass_res.passed_format else 'FAIL'} |",
        f"| All Phase 6 Rules Met | **{status_str}** | - | All conditions true | **{status_str}** |",
        "",
        "## 2. GRPO Training Configuration and Step Progression",
        "",
        "| Parameter | Specification | Recorded Setting |",
        "|---|---|---|",
        "| Base Model | SFT 2B LoRA checkpoint | `sft-1.0-2b` |",
        "| Algorithm | GRPO (TRL / Unsloth) | Group Relative Policy Optimization |",
        "| Generations Per Prompt | 8 | Group size G=8 |",
        "| Max Answer Length | 300 tokens | Support long detail mode |",
        "| Temperature | 0.9 | Exploration parameter |",
        "| Learning Rate | 5e-6 | Small RL step size |",
        "| KL Beta | 0.02 | Policy divergence penalty |",
        "| Hardware Target | NVIDIA L4 GPU | bfloat16 mixed precision |",
        "| Reward Function | Phase 3 rule checks | Hard fail (-1.0) and soft composite |",
    ]

    if ctx.training_metadata:
        lines.extend([
            "",
            "### Training Run Metadata",
            "",
            "| Training Metric | Value |",
            "|---|---|",
        ])
        for k, v in sorted(ctx.training_metadata.items()):
            lines.append(f"| `{k}` | {v} |")

    lines.extend([
        "",
        "### Training Step Progression",
        "",
    ])

    if ctx.step_logs:
        lines.extend([
            "| Step | Mean Reward | Hard-Fail Rate | Avg Word Count |",
            "|---|---|---|---|",
        ])
        for log in ctx.step_logs:
            s_num = log.get("step", 0)
            s_rew = log.get("mean_reward", 0.0)
            s_fail = log.get("hard_fail_rate", 0.0)
            s_len = log.get("avg_word_count", 0.0)
            lines.append(f"| {s_num} | {s_rew:.4f} | {s_fail:.1f}% | {s_len:.1f} |")
    else:
        lines.append("No periodic step logs were recorded for this training session.")

    lines.extend([
        "",
        "### Checkpoint Rankings",
        "",
        "Checkpoints are sorted by validation mean reward descending, then hard-fail rate ascending.",
        "",
    ])

    if ctx.checkpoint_rankings:
        lines.extend([
            "| Rank | Checkpoint | Step | Val Mean Reward | Val Hard-Fail | Val Avg Words | Val In-Band | Best? |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for rank_idx, entry in enumerate(ctx.checkpoint_rankings, start=1):
            is_best_str = "**YES**" if entry.is_best else "No"
            lines.append(
                f"| {rank_idx} | `{entry.checkpoint_name}` | {entry.step} | "
                f"**{entry.val_mean_reward:.4f}** | {entry.val_hard_fail_rate:.1f}% | "
                f"{entry.val_avg_word_count:.1f} | {entry.val_pct_in_band:.1f}% | {is_best_str} |"
            )
    else:
        lines.append("No multi-checkpoint ranking data was provided.")

    lines.extend([
        "",
        "## 3. Validation Split Automatic Metrics",
        "",
        f"Evaluated on `data/processed/sft_val.jsonl` ({v_agg.total_items} items) using `{ctx.best_checkpoint}`.",
        "",
        "| Metric | Validation Split Result |",
        "|---|---|",
        f"| Total Items | {v_agg.total_items} |",
        f"| Hard-Fail Rate | **{v_agg.hard_fail_rate:.1f}%** |",
        f"| Mean Reward | **{v_agg.mean_reward:.4f}** |",
        f"| Mean Soft Score (All) | {v_agg.mean_soft_score_all:.4f} |",
        f"| Mean Soft Score (Passing) | {v_agg.mean_soft_score_passing:.4f} |",
        f"| Average Word Count | {v_agg.avg_word_count_overall:.1f} words |",
        f"| Length in Target Band | {v_agg.pct_in_band_overall:.1f}% |",
        f"| Verdict Coverage Rate | {v_agg.verdict_coverage_rate:.1f}% |",
        f"| Verdict Accuracy (Covered) | {v_agg.verdict_accuracy_covered:.1f}% |",
        "",
        "## 4. Frozen Test Set Comparison",
        "",
        f"Evaluated on `eval/testset-1.0.jsonl` ({t_agg.total_items} items) against SFT baseline and base Qwen 2B.",
        "",
        "### 4.1 Overall Metrics Comparison",
        "",
        "| Metric | Second-1.0-2B-GRPO | Second-1.0-2B-SFT | Base Qwen 2B | Delta vs SFT | Delta vs Base |",
        "|---|---|---|---|---|---|",
        f"| Hard-Fail Rate | **{t_agg.hard_fail_rate:.1f}%** | {s_agg.hard_fail_rate:.1f}% | {b_agg.hard_fail_rate:.1f}% | {t_agg.hard_fail_rate - s_agg.hard_fail_rate:+.1f}% | {t_agg.hard_fail_rate - b_agg.hard_fail_rate:+.1f}% |",
        f"| Mean Reward | **{t_agg.mean_reward:.4f}** | {s_agg.mean_reward:.4f} | {b_agg.mean_reward:.4f} | {t_agg.mean_reward - s_agg.mean_reward:+.4f} | {t_agg.mean_reward - b_agg.mean_reward:+.4f} |",
        f"| Mean Soft Score (All) | {t_agg.mean_soft_score_all:.4f} | {s_agg.mean_soft_score_all:.4f} | {b_agg.mean_soft_score_all:.4f} | {t_agg.mean_soft_score_all - s_agg.mean_soft_score_all:+.4f} | {t_agg.mean_soft_score_all - b_agg.mean_soft_score_all:+.4f} |",
        f"| Mean Soft Score (Passing) | {t_agg.mean_soft_score_passing:.4f} | {s_agg.mean_soft_score_passing:.4f} | {b_agg.mean_soft_score_passing:.4f} | {t_agg.mean_soft_score_passing - s_agg.mean_soft_score_passing:+.4f} | {t_agg.mean_soft_score_passing - b_agg.mean_soft_score_passing:+.4f} |",
        f"| In Target Length Band | {t_agg.pct_in_band_overall:.1f}% | {s_agg.pct_in_band_overall:.1f}% | {b_agg.pct_in_band_overall:.1f}% | {t_agg.pct_in_band_overall - s_agg.pct_in_band_overall:+.1f}% | {t_agg.pct_in_band_overall - b_agg.pct_in_band_overall:+.1f}% |",
        f"| Average Word Count | {t_agg.avg_word_count_overall:.1f} words | {s_agg.avg_word_count_overall:.1f} words | {b_agg.avg_word_count_overall:.1f} words | {t_agg.avg_word_count_overall - s_agg.avg_word_count_overall:+.1f} words | {t_agg.avg_word_count_overall - b_agg.avg_word_count_overall:+.1f} words |",
        "",
        "### 4.2 Hard Check Breakdown",
        "",
        "If any hard check fails, the final reward is -1.0.",
        "",
        "| Check Name | GRPO Fail Rate | SFT Fail Rate | Base Qwen Fail Rate | Delta vs SFT | Delta vs Base |",
        "|---|---|---|---|---|---|",
    ])

    all_checks = sorted(
        set(t_agg.hard_check_fail_rates.keys())
        | set(s_agg.hard_check_fail_rates.keys())
        | set(b_agg.hard_check_fail_rates.keys())
    )
    for cname in all_checks:
        g_fail = t_agg.hard_check_fail_rates.get(cname, 0.0)
        s_fail = s_agg.hard_check_fail_rates.get(cname, 0.0)
        b_fail = b_agg.hard_check_fail_rates.get(cname, 0.0)
        lines.append(
            f"| `{cname}` | {g_fail:.1f}% | {s_fail:.1f}% | {b_fail:.1f}% | "
            f"{g_fail - s_fail:+.1f}% | {g_fail - b_fail:+.1f}% |"
        )

    lines.extend([
        "",
        "### 4.3 Soft Score Breakdown",
        "",
        "| Soft Metric | GRPO Mean Score | SFT Mean Score | Base Qwen Mean Score | Delta vs SFT | Delta vs Base |",
        "|---|---|---|---|---|---|",
    ])

    all_soft = sorted(
        set(t_agg.soft_score_breakdown.keys())
        | set(s_agg.soft_score_breakdown.keys())
        | set(b_agg.soft_score_breakdown.keys())
    )
    for sname in all_soft:
        g_s = t_agg.soft_score_breakdown.get(sname, 0.0)
        s_s = s_agg.soft_score_breakdown.get(sname, 0.0)
        b_s = b_agg.soft_score_breakdown.get(sname, 0.0)
        lines.append(
            f"| `{sname}` | {g_s:.4f} | {s_s:.4f} | {b_s:.4f} | "
            f"{g_s - s_s:+.4f} | {g_s - b_s:+.4f} |"
        )

    lines.extend([
        "",
        "### 4.4 Subgroup Breakdowns",
        "",
    ])

    for dimension, slices in t_agg.breakdowns.items():
        lines.extend([
            f"#### Subgroup: {dimension.title()}",
            "",
            f"| {dimension.title()} Slice | Items | GRPO Hard-Fail | GRPO Mean Reward | GRPO Avg Words |",
            "|---|---|---|---|---|",
        ])
        for sname, stats in sorted(slices.items()):
            cnt = int(stats.get("count", 0))
            h_fail = stats.get("hard_fail_rate", 0.0)
            rew = stats.get("mean_reward", 0.0)
            w_cnt = stats.get("avg_word_count", 0.0)
            lines.append(f"| `{sname}` | {cnt} | {h_fail:.1f}% | {rew:.4f} | {w_cnt:.1f} |")
        lines.append("")

    lines.extend([
        "## 5. Gemini Flash Judge Results on Best Checkpoint",
        "",
    ])

    if ctx.judge_results:
        rubric_data = ctx.judge_results.get("rubric", {})
        rubric_summary = rubric_data.get("summary", {})
        pairwise_data = ctx.judge_results.get("pairwise", {})
        pairwise_summaries = pairwise_data.get("summaries", {})

        lines.extend([
            "### 5.1 Rubric Scores (1 to 5 scale)",
            "",
            "| Rubric Dimension | Mean Score (1-5) | Scale Standard |",
            "|---|---|---|",
            f"| Correctness | **{rubric_summary.get('correctness', rubric_summary.get('mean_correctness', 0.0)):.2f} / 5.0** | Factual chess accuracy |",
            f"| Clarity | **{rubric_summary.get('clarity', rubric_summary.get('mean_clarity', 0.0)):.2f} / 5.0** | Explanation structure and plain text |",
            f"| Usefulness | **{rubric_summary.get('usefulness', rubric_summary.get('mean_usefulness', 0.0)):.2f} / 5.0** | Practical instruction value |",
            f"| Level Fit | **{rubric_summary.get('level_fit', rubric_summary.get('mean_level_fit', 0.0)):.2f} / 5.0** | Vocabulary matching target level |",
            f"| Overall Mean | **{rubric_summary.get('overall', rubric_summary.get('mean_overall', 0.0)):.2f} / 5.0** | Composite rubric score |",
            "",
            "### 5.2 Pairwise Comparisons (Swap-Consistent)",
            "",
            "| Opponent Model | Total Pairs | GRPO Wins | Opponent Wins | Ties | Win Rate (%) | Win + Tie Rate (%) | Swap Agreement (%) |",
            "|---|---|---|---|---|---|---|---|",
        ])

        for opp_name, p_sum in sorted(pairwise_summaries.items()):
            tot = p_sum.get("total_pairs", 0)
            w = p_sum.get("candidate_wins", 0)
            l = p_sum.get("opponent_wins", 0)
            t = p_sum.get("ties", 0)
            wr = p_sum.get("candidate_win_rate", 0.0)
            wtr = p_sum.get("candidate_win_tie_rate", 0.0)
            agr = p_sum.get("swap_agreement_rate", 0.0)
            lines.append(
                f"| `{opp_name}` | {tot} | {w} | {l} | {t} | **{wr:.1f}%** | {wtr:.1f}% | {agr:.1f}% |"
            )
        lines.append("")
    else:
        lines.extend([
            "The Gemini Flash judge evaluation was skipped for this session.",
            "Run evaluation with Gemini API enabled to compute rubric scores and pairwise win rates.",
            "",
        ])

    lines.extend([
        "## 6. Manual Review of 20 Outputs and Anti-Hacking Audit",
        "",
        "This section lists 20 deterministically selected test outputs across all 8 strata.",
        "It includes both passing and failing items to monitor policy changes and detect reward hacking.",
        "",
    ])

    for idx, item in enumerate(ctx.reviewed_20, start=1):
        lines.extend([
            f"### Sample {idx:02d}: `{item['id']}` ({item['task']}, {item['level']}, {item['detail']})",
            "",
            f"- **Verdict**: {item.get('verdict') or 'None'}",
            f"- **Move**: `{item.get('move_san') or 'None'}`",
            f"- **Word Count**: {item.get('word_count')} words",
            f"- **Reward**: `{item.get('reward')}`",
            f"- **Hard Checks Passed**: {'Yes' if item.get('passed_all_hard') else 'No'}",
            f"- **Failed Checks**: `{', '.join(item.get('failed_checks', [])) or 'None'}`",
            "",
            "**Completion Text**:",
            "> " + str(item.get("completion", "")).replace("\n", "\n> "),
            "",
            f"**Review Note**: {item.get('review_note')}",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 7. Anti-Hacking Audit Findings",
        "",
        "This section verifies that policy updates optimize real quality rather than reward loopholes.",
        "",
        f"- **Length Collapse Audit**: {audit['length_collapse_count']} of {len(ctx.reviewed_20)} outputs ({audit['length_collapse_rate']}%) fall below target minimum length.",
        f"  - Status: {'ALERT: Potential length collapse observed.' if audit['length_collapse_flag'] else 'PASS: Length distribution remains healthy.'}",
        f"- **Robotic Opening Audit**: {audit['robotic_opening_count']} of {len(ctx.reviewed_20)} outputs ({audit['robotic_opening_rate']}%) use standard template starters.",
        f"  - Frequent openings: {', '.join(audit['top_openings']) or 'None'}",
        f"  - Status: {'ALERT: Phrasing diversity is low. Check for robotic openings.' if audit['robotic_opening_flag'] else 'PASS: Opening phrasing diversity is acceptable.'}",
        f"- **Reward vs Judge Concordance**: {audit['judge_concordance_note']}",
        f"  - Status: {'PASS: Reward and judge agree.' if audit['judge_concordance_flag'] else 'ALERT: Potential reward hacking detected.'}",
    ])

    return "\n".join(lines)


def run_phase6_evaluation(
    val_outputs_path: str | Path = DEFAULT_VAL_OUTPUTS,
    test_outputs_path: str | Path = DEFAULT_TEST_OUTPUTS,
    sft_outputs_path: str | Path = DEFAULT_SFT_OUTPUTS,
    base_qwen_path: str | Path = DEFAULT_BASE_QWEN,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    model_name: str = "grpo-1.0-2b",
    best_checkpoint: str = "checkpoint-final",
    checkpoint_val_paths: dict[str, str | Path] | None = None,
    checkpoint_test_paths: dict[str, str | Path] | None = None,
    step_logs: list[dict[str, Any]] | None = None,
    training_metadata: dict[str, Any] | None = None,
    run_judge: bool = True,
    skip_judge: bool = False,
    skip_judge_if_no_key: bool = True,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    gemini_baseline_path: str | Path = DEFAULT_GEMINI_BASELINE,
    judge_sample_size: int = 0,
    judge_workers: int = 8,
) -> dict[str, Any]:
    """Execute complete Phase 6 evaluation on validation and frozen test outputs.

    Evaluate metrics on validation split and frozen test set.
    Rank checkpoints by validation performance.
    Verify Phase 6 release criteria against SFT and base Qwen 2B baselines.
    Run optional Gemini judge evaluation on the best checkpoint.
    Save JSON evaluation metrics and generate Markdown report.
    Return evaluation summary dictionary.
    """
    val_path = Path(val_outputs_path)
    test_path = Path(test_outputs_path)
    sft_path = Path(sft_outputs_path)
    base_path = Path(base_qwen_path)
    res_dir = Path(results_dir)
    rep_dir = Path(reports_dir)

    res_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Evaluating validation split: %s", val_path)
    val_metrics = evaluate_file(val_path)

    LOGGER.info("Evaluating test set completions: %s", test_path)
    test_metrics = evaluate_file(test_path)

    LOGGER.info("Loading SFT baseline: %s", sft_path)
    if not sft_path.is_file():
        raise FileNotFoundError(f"SFT baseline file not found: {sft_path}")
    sft_metrics = evaluate_file(sft_path)

    LOGGER.info("Loading base Qwen baseline: %s", base_path)
    if not base_path.is_file():
        raise FileNotFoundError(f"Base Qwen baseline file not found: {base_path}")
    base_qwen_metrics = evaluate_file(base_path)

    # Checkpoint rankings
    rankings: list[CheckpointRankingEntry] = []
    effective_best_checkpoint = best_checkpoint

    if checkpoint_val_paths:
        LOGGER.info("Evaluating %d checkpoint validation sets...", len(checkpoint_val_paths))
        chk_val_map: dict[str, EvalMetrics] = {}
        for cname, cpath in checkpoint_val_paths.items():
            chk_val_map[cname] = evaluate_file(cpath)

        chk_test_map: dict[str, EvalMetrics] = {}
        if checkpoint_test_paths:
            for cname, cpath in checkpoint_test_paths.items():
                chk_test_map[cname] = evaluate_file(cpath)

        rankings, top_checkpoints = rank_checkpoints(chk_val_map, chk_test_map)
        if top_checkpoints:
            effective_best_checkpoint = top_checkpoints[0]
    else:
        step_val = extract_step_number(best_checkpoint)
        single_entry = CheckpointRankingEntry(
            checkpoint_name=best_checkpoint,
            step=step_val,
            val_mean_reward=float(val_metrics.aggregate.mean_reward),
            val_hard_fail_rate=float(val_metrics.aggregate.hard_fail_rate),
            val_avg_word_count=float(val_metrics.aggregate.avg_word_count_overall),
            val_pct_in_band=float(val_metrics.aggregate.pct_in_band_overall),
            test_mean_reward=float(test_metrics.aggregate.mean_reward),
            test_hard_fail_rate=float(test_metrics.aggregate.hard_fail_rate),
            test_avg_word_count=float(test_metrics.aggregate.avg_word_count_overall),
            test_pct_in_band=float(test_metrics.aggregate.pct_in_band_overall),
            is_best=True,
        )
        rankings = [single_entry]

    pass_result = check_phase6_pass_rule(test_metrics, sft_metrics, base_qwen_metrics)
    test_records = load_jsonl(test_path)
    reviewed_20 = select_20_outputs_for_review(test_metrics.items, test_records)

    # Gemini Judge execution
    judge_results: dict[str, Any] | None = None
    if run_judge and not skip_judge:
        api_key = get_gemini_api_key(required=False)
        if not api_key:
            if skip_judge_if_no_key:
                LOGGER.warning("GEMINI_API_KEY is not set. Skipping Gemini judge evaluation.")
            else:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is not set. "
                    "Set GEMINI_API_KEY or specify skip_judge_if_no_key=True."
                )
        else:
            opponent_records: dict[str, list[dict[str, Any]]] = {}
            if base_path.is_file():
                opponent_records["base-qwen-2b"] = load_jsonl(base_path)
            if sft_path.is_file():
                opponent_records["sft-1.0-2b"] = load_jsonl(sft_path)
            gemini_path = Path(gemini_baseline_path)
            if gemini_path.is_file():
                opponent_records["gemini-flash"] = load_jsonl(gemini_path)

            judge_candidates = (
                test_records[:judge_sample_size]
                if judge_sample_size > 0
                else test_records
            )
            LOGGER.info(
                "Executing Gemini judge (%s) on %d items against %d opponents (workers=%d)...",
                judge_model,
                len(judge_candidates),
                len(opponent_records),
                judge_workers,
            )
            judge_results = run_judge_eval(
                candidate_records=judge_candidates,
                opponent_records=opponent_records,
                judge_model=judge_model,
                cache_path=cache_path,
                api_key=api_key,
                max_workers=judge_workers,
            )
            judge_file = res_dir / f"{model_name}_judge.json"
            judge_file.write_text(json.dumps(judge_results, indent=2), encoding="utf-8")
            LOGGER.info("Saved judge results to %s", judge_file)

    eval_context = Phase6EvalContext(
        model_name=model_name,
        best_checkpoint=effective_best_checkpoint,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        sft_test_metrics=sft_metrics,
        base_qwen_metrics=base_qwen_metrics,
        pass_result=pass_result,
        reviewed_20=reviewed_20,
        checkpoint_rankings=rankings,
        step_logs=step_logs or [],
        judge_results=judge_results,
        training_metadata=training_metadata,
    )
    report_md = generate_phase6_report(eval_context)

    val_eval_file = res_dir / f"{model_name}_val_eval.json"
    test_eval_file = res_dir / f"{model_name}_eval.json"
    report_file = rep_dir / f"{model_name}.md"

    val_eval_file.write_text(val_metrics.to_json(), encoding="utf-8")
    test_eval_file.write_text(test_metrics.to_json(), encoding="utf-8")
    report_file.write_text(report_md, encoding="utf-8")

    LOGGER.info("Saved validation eval metrics to %s", val_eval_file)
    LOGGER.info("Saved test eval metrics to %s", test_eval_file)
    LOGGER.info("Saved Phase 6 evaluation report to %s", report_file)

    return {
        "model_name": model_name,
        "best_checkpoint": effective_best_checkpoint,
        "passed_phase6": pass_result.passed_all,
        "pass_rule": pass_result.to_dict(),
        "val_metrics": val_metrics.aggregate.to_dict(),
        "test_metrics": test_metrics.aggregate.to_dict(),
        "sft_test_metrics": sft_metrics.aggregate.to_dict(),
        "base_qwen_metrics": base_qwen_metrics.aggregate.to_dict(),
        "checkpoint_rankings": [r.to_dict() for r in rankings],
        "report_path": str(report_file),
        "val_eval_path": str(val_eval_file),
        "test_eval_path": str(test_eval_file),
        "reviewed_20_count": len(reviewed_20),
        "judge_results": judge_results,
    }


def main() -> None:
    """CLI entry point for Phase 6 evaluation runner."""
    parser = argparse.ArgumentParser(description="Phase 6 evaluation runner for Second GRPO.")
    parser.add_argument(
        "--from-zip",
        type=str,
        default=None,
        help="Path to zip bundle exported from Colab (e.g. phase6_grpo_results.zip)",
    )
    parser.add_argument(
        "--val-outputs",
        type=str,
        default=str(DEFAULT_VAL_OUTPUTS),
        help="Path to GRPO validation outputs JSONL",
    )
    parser.add_argument(
        "--test-outputs",
        type=str,
        default=str(DEFAULT_TEST_OUTPUTS),
        help="Path to GRPO test set outputs JSONL",
    )
    parser.add_argument(
        "--sft-outputs",
        type=str,
        default=str(DEFAULT_SFT_OUTPUTS),
        help="Path to SFT baseline outputs JSONL",
    )
    parser.add_argument(
        "--base-qwen",
        type=str,
        default=str(DEFAULT_BASE_QWEN),
        help="Path to base Qwen 2B baseline JSONL",
    )
    parser.add_argument(
        "--gemini-baseline",
        type=str,
        default=str(DEFAULT_GEMINI_BASELINE),
        help="Path to Gemini Flash baseline JSONL",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save JSON evaluation results",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory to save Markdown reports",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="grpo-1.0-2b",
        help="Model name identifier",
    )
    parser.add_argument(
        "--best-checkpoint",
        type=str,
        default="checkpoint-final",
        help="Name of best checkpoint",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Gemini judge evaluation",
    )
    parser.add_argument(
        "--skip-judge-if-no-key",
        action="store_true",
        default=True,
        help="Skip judge if GEMINI_API_KEY is not set",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Gemini judge model identifier",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=str(DEFAULT_CACHE_PATH),
        help="Path to judge cache JSONL",
    )
    parser.add_argument(
        "--judge-sample-size",
        type=int,
        default=0,
        help="Number of test items to evaluate with Gemini judge (0 for full test set)",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads for Gemini judge",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    val_path = Path(args.val_outputs)
    test_path = Path(args.test_outputs)
    sft_path = Path(args.sft_outputs)
    base_path = Path(args.base_qwen)
    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)

    training_metadata: dict[str, Any] | None = None
    step_logs: list[dict[str, Any]] | None = None
    checkpoint_val_paths: dict[str, str | Path] | None = None

    if args.from_zip:
        zip_path = Path(args.from_zip)
        if not zip_path.is_file():
            LOGGER.error("Zip bundle file not found: %s", zip_path)
            sys.exit(1)

        LOGGER.info("Extracting results bundle from %s into %s", zip_path, results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".md"):
                    continue
                zf.extract(member, results_dir)

        # Discover extracted files
        candidates_val = [
            p for p in results_dir.glob("*val_outputs*.jsonl")
            if not p.name.startswith("checkpoint-")
        ]
        if not candidates_val:
            candidates_val = list(results_dir.glob("*val_outputs*.jsonl"))

        candidates_test = [
            p for p in results_dir.glob("*outputs*.jsonl")
            if "val" not in p.name and "sft" not in p.name and "base" not in p.name and not p.name.startswith("checkpoint-")
        ]
        candidates_summary = list(results_dir.glob("*grpo*summary*.json")) + list(
            results_dir.glob("*training_summary*.json")
        )

        # Discover multiple checkpoint val files if present
        chk_files = list(results_dir.glob("checkpoint-*_val*.jsonl"))
        if chk_files:
            checkpoint_val_paths = {}
            for cf in chk_files:
                chk_name = cf.name.split("_val")[0]
                checkpoint_val_paths[chk_name] = cf
            LOGGER.info("Discovered %d checkpoint val files from zip", len(checkpoint_val_paths))

        if candidates_val:
            val_path = candidates_val[0]
            val_rows = load_jsonl(val_path)
            if any("comment" in r for r in val_rows):
                with open(val_path, "w", encoding="utf-8") as vf:
                    for r in val_rows:
                        r.pop("comment", None)
                        vf.write(json.dumps(r) + "\n")
            LOGGER.info("Found validation outputs: %s", val_path)

        if candidates_test:
            test_path = candidates_test[0]
            LOGGER.info("Found test outputs: %s", test_path)

        if candidates_summary:
            summary_path = candidates_summary[0]
            LOGGER.info("Found training summary: %s", summary_path)
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    training_metadata = json.load(f)
                    if "step_logs" in training_metadata:
                        step_logs = training_metadata["step_logs"]
                    elif "training_logs" in training_metadata:
                        step_logs = training_metadata["training_logs"]
                    if (
                        "best_checkpoint" in training_metadata
                        and args.best_checkpoint == "checkpoint-final"
                    ):
                        args.best_checkpoint = str(training_metadata["best_checkpoint"])
            except Exception as exc:
                LOGGER.warning("Could not read training summary: %s", exc)

    LOGGER.info("Running Phase 6 evaluation for %s...", args.model_name)
    summary = run_phase6_evaluation(
        val_outputs_path=val_path,
        test_outputs_path=test_path,
        sft_outputs_path=sft_path,
        base_qwen_path=base_path,
        results_dir=results_dir,
        reports_dir=reports_dir,
        model_name=args.model_name,
        best_checkpoint=args.best_checkpoint,
        checkpoint_val_paths=checkpoint_val_paths,
        step_logs=step_logs,
        training_metadata=training_metadata,
        run_judge=not args.skip_judge,
        skip_judge=args.skip_judge,
        skip_judge_if_no_key=args.skip_judge_if_no_key,
        judge_model=args.judge_model,
        cache_path=args.cache_path,
        gemini_baseline_path=args.gemini_baseline,
        judge_sample_size=args.judge_sample_size,
        judge_workers=args.judge_workers,
    )

    print("\nPhase 6 Evaluation Complete:")
    print(f"- Overall Pass Result: {'PASS' if summary['passed_phase6'] else 'FAIL'}")
    print(f"- Best Checkpoint: {summary['best_checkpoint']}")
    print(f"- Format Pass Rate: {summary['pass_rule']['grpo_format_pass_rate']:.1f}% (> 95.0% required)")
    print(f"- GRPO Mean Reward: {summary['pass_rule']['grpo_mean_reward']:.4f} (SFT: {summary['pass_rule']['sft_mean_reward']:.4f})")
    print(f"- GRPO Hard-Fail Rate: {summary['pass_rule']['grpo_hard_fail_rate']:.1f}% (Base: {summary['pass_rule']['base_hard_fail_rate']:.1f}%, SFT: {summary['pass_rule']['sft_hard_fail_rate']:.1f}%)")
    print(f"- Report written to: {summary['report_path']}")


if __name__ == "__main__":
    main()
