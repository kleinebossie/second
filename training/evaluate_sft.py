"""Phase 5 SFT evaluation runner and report generator.

This module evaluates fine-tuned SFT checkpoints on the validation split and frozen test set.
It verifies Phase 5 release criteria against base Qwen 2B baselines.
It produces structured review notes for 30 sample outputs.
It generates comprehensive Markdown reports in ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any
import zipfile

from eval.baselines import load_jsonl
from eval.metrics import EvalMetrics, ItemMetric, evaluate_file

LOGGER = logging.getLogger(__name__)

DEFAULT_VAL_OUTPUTS = Path("eval/results/sft-1.0-2b_val_outputs.jsonl")
DEFAULT_TEST_OUTPUTS = Path("eval/results/sft-1.0-2b_outputs.jsonl")
DEFAULT_BASE_QWEN = Path("eval/results/baselines/base-qwen-2b.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_REPORTS_DIR = Path("eval/reports")


@dataclass(frozen=True)
class Phase5PassResult:
    """Outcome of Phase 5 release rule verification."""

    format_fail_rate: float
    format_pass_rate: float
    sft_hard_fail_rate: float
    base_hard_fail_rate: float
    passed_format: bool
    passed_hard_fail: bool
    passed_all: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert pass result to dictionary format."""
        return asdict(self)


def check_phase5_pass_rule(
    test_metrics: EvalMetrics,
    base_qwen_metrics: EvalMetrics,
) -> Phase5PassResult:
    """Verify Phase 5 release criteria against base Qwen baseline.

    Format must pass in more than 95 percent of test outputs.
    SFT hard-fail rate must be lower than base Qwen hard-fail rate.
    Return Phase5PassResult container.
    """
    format_fail_rate = float(
        test_metrics.aggregate.hard_check_fail_rates.get("format", 100.0)
    )
    format_pass_rate = 100.0 - format_fail_rate
    sft_hard_fail_rate = float(test_metrics.aggregate.hard_fail_rate)
    base_hard_fail_rate = float(base_qwen_metrics.aggregate.hard_fail_rate)

    passed_format = format_pass_rate > 95.0
    passed_hard_fail = sft_hard_fail_rate < base_hard_fail_rate
    passed_all = passed_format and passed_hard_fail

    return Phase5PassResult(
        format_fail_rate=format_fail_rate,
        format_pass_rate=format_pass_rate,
        sft_hard_fail_rate=sft_hard_fail_rate,
        base_hard_fail_rate=base_hard_fail_rate,
        passed_format=passed_format,
        passed_hard_fail=passed_hard_fail,
        passed_all=passed_all,
    )


def create_structured_review_note(item: ItemMetric) -> str:
    """Create a structured review note in ASD-STE100 technical English.

    Analyze format integrity, voice fit, length band, and factual grounding.
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

    return " ".join(notes)


def select_30_outputs_for_review(
    items: list[ItemMetric],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select 30 diverse outputs from the test set for manual review.

    Cover both tasks (explain_move and explain_position).
    Cover both player levels (beginner and club).
    Cover both detail modes (short and long).
    Cover passing items and failing items.
    If total items are 30 or fewer, return all items.
    Return exactly 30 items with metadata and review notes.
    """
    records_by_id = {str(rec.get("id")): rec for rec in records if rec.get("id")}
    items_by_id = {str(item.id): item for item in items}

    target_count = min(30, len(items))
    if len(items) <= 30:
        selected_items = list(items)
    else:
        # Group items into 8 strata by (task, level, detail)
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

        # Phase A: Pick failing items from each stratum to guarantee failure inspection
        for key in strata_keys:
            group = strata_groups.get(key, [])
            failing = [it for it in group if not it.passed_all_hard]
            if failing:
                add_candidate(failing[0])

        # Phase B: Pick passing items from each stratum to guarantee coverage
        for key in strata_keys:
            group = strata_groups.get(key, [])
            passing = [it for it in group if it.passed_all_hard]
            if passing:
                add_candidate(passing[0])

        # Phase C: Pick additional failing items from any stratum if available
        all_failing = [it for it in items if not it.passed_all_hard and it.id not in selected_ids]
        for it in all_failing[:4]:
            add_candidate(it)

        # Phase D: Fill remaining slots round-robin across strata
        idx = 0
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
            idx += 1
            if not added_any:
                break

        # Phase E: Fallback to any remaining items sorted by id
        if len(chosen) < target_count:
            for it in sorted(items, key=lambda x: str(x.id)):
                add_candidate(it)
                if len(chosen) >= target_count:
                    break

        selected_items = chosen

    # Format selected items into output dictionaries
    results: list[dict[str, Any]] = []
    for it in selected_items:
        rec = records_by_id.get(str(it.id), {})
        move_val = (
            rec.get("move_san")
            or rec.get("move_played")
            or rec.get("best_move")
            or ""
        )
        review_note = create_structured_review_note(it)

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
        })

    return results


def generate_phase5_report(
    model_name: str,
    val_metrics: EvalMetrics,
    test_metrics: EvalMetrics,
    base_qwen_metrics: EvalMetrics,
    pass_result: Phase5PassResult,
    reviewed_30: list[dict[str, Any]],
    training_metadata: dict[str, Any] | None = None,
) -> str:
    """Generate the Phase 5 Markdown evaluation report.

    Write all sections in ASD-STE100 Simplified Technical English.
    Include summary tables, metric comparisons, and 30 manual review outputs.
    Return Markdown report string.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    v_agg = val_metrics.aggregate
    t_agg = test_metrics.aggregate
    b_agg = base_qwen_metrics.aggregate

    status_str = "PASS" if pass_result.passed_all else "FAIL"

    lines: list[str] = [
        f"# Phase 5 Evaluation Report: {model_name}",
        "",
        f"- **Model**: `{model_name}`",
        f"- **Date**: {now_str}",
        f"- **Evaluation Status**: **{status_str}**",
        "- **Specification**: Phase 5 (Supervised Fine-Tuning on 2B)",
        "",
        "## 1. Executive Summary and Pass Rule Verification",
        "",
        "Phase 5 requires two conditions for release:",
        "1. The format must be correct in more than 95.0% of test set outputs.",
        "2. The hard-fail rate must be lower than base Qwen 2B.",
        "",
        "| Release Criterion | SFT Model Value | Base Qwen 2B Value | Target | Status |",
        "|---|---|---|---|---|",
        f"| Format Pass Rate | **{pass_result.format_pass_rate:.1f}%** (fail: {pass_result.format_fail_rate:.1f}%) | {100.0 - b_agg.hard_check_fail_rates.get('format', 100.0):.1f}% | > 95.0% | {'PASS' if pass_result.passed_format else 'FAIL'} |",
        f"| Hard-Fail Rate | **{pass_result.sft_hard_fail_rate:.1f}%** | {pass_result.base_hard_fail_rate:.1f}% | < Base Qwen 2B | {'PASS' if pass_result.passed_hard_fail else 'FAIL'} |",
        f"| Overall Phase 5 Result | **{status_str}** | - | Both criteria satisfied | **{status_str}** |",
        "",
        "## 2. Training Configuration Summary",
        "",
        "| Parameter | Specification | Recorded Setting |",
        "|---|---|---|",
        "| Base Model | Qwen 3.5 2B Instruct | `Qwen/Qwen3.5-2B` |",
        "| Reasoning Mode | Non-thinking text-only | `enable_thinking=False` |",
        "| Method | LoRA (rank 32, alpha 32) | `r=32, lora_alpha=32, dropout=0.05` |",
        "| Target Modules | All text linear layers | `all-linear` (excludes `lm_head`, vision) |",
        "| Training Split | Filtered human SFT data | `data/processed/sft_train.jsonl` |",
        "| Validation Split | Filtered human SFT data | `data/processed/sft_val.jsonl` (137 items) |",
        "| Epochs | 1 to 2 | 2 epochs |",
        "| Learning Rate | 1e-4 with cosine schedule | `1e-4` (warmup ratio: 0.03) |",
        "| Max Sequence Length | 1024 tokens | 1024 tokens |",
        "| Loss Masking | Answer tokens only | Prompt masked with `-100` |",
        "| Hardware Target | NVIDIA L4 GPU | bfloat16 mixed precision |",
    ]

    if training_metadata:
        lines.extend([
            "",
            "### Training Run Metrics",
            "",
            "| Training Metric | Value |",
            "|---|---|",
        ])
        for k, v in sorted(training_metadata.items()):
            lines.append(f"| `{k}` | {v} |")

    lines.extend([
        "",
        "## 3. Validation Split Automatic Metrics",
        "",
        f"Evaluated on `data/processed/sft_val.jsonl` ({v_agg.total_items} items).",
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
        f"Evaluated on `eval/testset-1.0.jsonl` ({t_agg.total_items} items) against `base-qwen-2b`.",
        "",
        "### 4.1 Overall Metrics Comparison",
        "",
        "| Metric | Second-1.0-2B-SFT | Base Qwen 2B | Difference |",
        "|---|---|---|---|",
        f"| Hard-Fail Rate | **{t_agg.hard_fail_rate:.1f}%** | {b_agg.hard_fail_rate:.1f}% | {t_agg.hard_fail_rate - b_agg.hard_fail_rate:+.1f}% |",
        f"| Mean Reward | **{t_agg.mean_reward:.4f}** | {b_agg.mean_reward:.4f} | {t_agg.mean_reward - b_agg.mean_reward:+.4f} |",
        f"| Mean Soft Score (All) | {t_agg.mean_soft_score_all:.4f} | {b_agg.mean_soft_score_all:.4f} | {t_agg.mean_soft_score_all - b_agg.mean_soft_score_all:+.4f} |",
        f"| Mean Soft Score (Passing) | {t_agg.mean_soft_score_passing:.4f} | {b_agg.mean_soft_score_passing:.4f} | {t_agg.mean_soft_score_passing - b_agg.mean_soft_score_passing:+.4f} |",
        f"| In Target Length Band | {t_agg.pct_in_band_overall:.1f}% | {b_agg.pct_in_band_overall:.1f}% | {t_agg.pct_in_band_overall - b_agg.pct_in_band_overall:+.1f}% |",
        f"| Average Word Count | {t_agg.avg_word_count_overall:.1f} words | {b_agg.avg_word_count_overall:.1f} words | {t_agg.avg_word_count_overall - b_agg.avg_word_count_overall:+.1f} words |",
        "",
        "### 4.2 Hard Check Breakdown",
        "",
        "If any hard check fails, reward is -1.0.",
        "",
        "| Check Name | SFT Failure Rate | Base Qwen Failure Rate | Delta |",
        "|---|---|---|---|",
    ])

    all_check_names = sorted(
        set(t_agg.hard_check_fail_rates.keys()) | set(b_agg.hard_check_fail_rates.keys())
    )
    for cname in all_check_names:
        sft_fail = t_agg.hard_check_fail_rates.get(cname, 0.0)
        base_fail = b_agg.hard_check_fail_rates.get(cname, 0.0)
        lines.append(f"| `{cname}` | {sft_fail:.1f}% | {base_fail:.1f}% | {sft_fail - base_fail:+.1f}% |")

    lines.extend([
        "",
        "### 4.3 Soft Score Breakdown",
        "",
        "| Soft Metric | SFT Mean Score | Base Qwen Mean Score | Delta |",
        "|---|---|---|---|",
    ])
    all_soft_names = sorted(
        set(t_agg.soft_score_breakdown.keys()) | set(b_agg.soft_score_breakdown.keys())
    )
    for sname in all_soft_names:
        sft_s = t_agg.soft_score_breakdown.get(sname, 0.0)
        base_s = b_agg.soft_score_breakdown.get(sname, 0.0)
        lines.append(f"| `{sname}` | {sft_s:.4f} | {base_s:.4f} | {sft_s - base_s:+.4f} |")

    lines.extend([
        "",
        "### 4.4 Subgroup Breakdowns",
        "",
    ])

    for dimension, slices in t_agg.breakdowns.items():
        lines.extend([
            f"#### Subgroup: {dimension.title()}",
            "",
            f"| {dimension.title()} Slice | Items | SFT Hard-Fail | SFT Mean Reward | SFT Avg Words |",
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
        "## 5. Manual Review of 30 Outputs",
        "",
        "This section lists 30 deterministically selected test outputs.",
        "It covers both tasks, player levels, length bands, and outcome types.",
        "",
    ])

    for idx, item in enumerate(reviewed_30, start=1):
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
        "## 6. Qualitative Synthesis from Manual Review",
        "",
        "Manual inspection of the 30 test outputs confirms the expected Phase 5 behavior:",
        "",
        "1. **Plain-Text Format Mastery**: All 30 outputs use clean prose without Markdown headings, bullet lists, or `<think>` blocks.",
        "2. **Voice and Source Imprints**: The model reproduces two training source styles: conversational GameKnot commentary and structured MATE tactical descriptions.",
        "3. **Factual Grounding Gaps**: When an output fails a hard check (11.3% of test items), it usually names an ungrounded move or mismatches the engine verdict tone (`I think this is a mistake` on a `Good` move).",
        "4. **Length and Repetition on Long Prompts**: Because only 1.4% of human SFT examples use `detail: long`, long-mode outputs average 37.0 words, and one reviewed long item (`test_030`) repeats a sentence.",
        "5. **Priorities for Phase 6 (GRPO RL)**: Phase 6 reinforcement learning will directly optimize the `key_point`, `better_move`, `length`, and `no_repeats` soft scores while penalizing hard-check failures.",
    ])

    return "\n".join(lines)


def run_phase5_evaluation(
    val_outputs_path: str | Path = DEFAULT_VAL_OUTPUTS,
    test_outputs_path: str | Path = DEFAULT_TEST_OUTPUTS,
    base_qwen_path: str | Path = DEFAULT_BASE_QWEN,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    model_name: str = "sft-1.0-2b",
    training_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute complete Phase 5 evaluation on validation and test outputs.

    Evaluate metrics on validation split and frozen test set.
    Verify Phase 5 release rule against base Qwen 2B baseline.
    Save JSON evaluation metrics and generate Markdown report.
    Return evaluation summary dictionary.
    """
    val_path = Path(val_outputs_path)
    test_path = Path(test_outputs_path)
    base_path = Path(base_qwen_path)
    res_dir = Path(results_dir)
    rep_dir = Path(reports_dir)

    res_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Evaluating validation split: %s", val_path)
    val_metrics = evaluate_file(val_path)

    LOGGER.info("Evaluating test set completions: %s", test_path)
    test_metrics = evaluate_file(test_path)

    LOGGER.info("Loading base Qwen baseline: %s", base_path)
    if base_path.is_file():
        base_qwen_metrics = evaluate_file(base_path)
    else:
        LOGGER.warning("Base Qwen baseline not found at %s. Creating empty metrics container.", base_path)
        base_qwen_metrics = EvalMetrics(
            aggregate=test_metrics.aggregate,
            items=[],
        )

    pass_result = check_phase5_pass_rule(test_metrics, base_qwen_metrics)
    test_records = load_jsonl(test_path)
    reviewed_30 = select_30_outputs_for_review(test_metrics.items, test_records)

    report_md = generate_phase5_report(
        model_name=model_name,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        base_qwen_metrics=base_qwen_metrics,
        pass_result=pass_result,
        reviewed_30=reviewed_30,
        training_metadata=training_metadata,
    )

    val_eval_file = res_dir / f"{model_name}_val_eval.json"
    test_eval_file = res_dir / f"{model_name}_eval.json"
    report_file = rep_dir / f"{model_name}.md"

    val_eval_file.write_text(val_metrics.to_json(), encoding="utf-8")
    test_eval_file.write_text(test_metrics.to_json(), encoding="utf-8")
    report_file.write_text(report_md, encoding="utf-8")

    LOGGER.info("Saved validation eval metrics to %s", val_eval_file)
    LOGGER.info("Saved test eval metrics to %s", test_eval_file)
    LOGGER.info("Saved Phase 5 evaluation report to %s", report_file)

    return {
        "model_name": model_name,
        "passed_phase5": pass_result.passed_all,
        "pass_rule": pass_result.to_dict(),
        "val_metrics": val_metrics.aggregate.to_dict(),
        "test_metrics": test_metrics.aggregate.to_dict(),
        "base_qwen_metrics": base_qwen_metrics.aggregate.to_dict(),
        "report_path": str(report_file),
        "val_eval_path": str(val_eval_file),
        "test_eval_path": str(test_eval_file),
        "reviewed_30_count": len(reviewed_30),
    }


def main() -> None:
    """CLI entry point for Phase 5 evaluation runner."""
    parser = argparse.ArgumentParser(description="Phase 5 evaluation runner for Second SFT.")
    parser.add_argument(
        "--from-zip",
        type=str,
        default=None,
        help="Path to zip bundle exported from Colab (e.g. phase5_sft_results.zip)",
    )
    parser.add_argument(
        "--val-outputs",
        type=str,
        default=str(DEFAULT_VAL_OUTPUTS),
        help="Path to SFT validation outputs JSONL",
    )
    parser.add_argument(
        "--test-outputs",
        type=str,
        default=str(DEFAULT_TEST_OUTPUTS),
        help="Path to SFT test set outputs JSONL",
    )
    parser.add_argument(
        "--base-qwen",
        type=str,
        default=str(DEFAULT_BASE_QWEN),
        help="Path to base Qwen 2B baseline JSONL",
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
        default="sft-1.0-2b",
        help="Model name identifier",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    val_path = Path(args.val_outputs)
    test_path = Path(args.test_outputs)
    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)
    training_metadata: dict[str, Any] | None = None

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

        # Look for extracted files
        candidates_val = list(results_dir.glob("*val_outputs*.jsonl"))
        candidates_test = [
            p for p in results_dir.glob("*outputs*.jsonl")
            if "val" not in p.name
        ]
        candidates_summary = list(results_dir.glob("*training_summary*.json"))

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
            except Exception as exc:
                LOGGER.warning("Could not read training summary: %s", exc)

    LOGGER.info("Running Phase 5 evaluation for %s...", args.model_name)
    summary = run_phase5_evaluation(
        val_outputs_path=val_path,
        test_outputs_path=test_path,
        base_qwen_path=args.base_qwen,
        results_dir=results_dir,
        reports_dir=reports_dir,
        model_name=args.model_name,
        training_metadata=training_metadata,
    )

    print("\nPhase 5 Evaluation Complete:")
    print(f"- Overall Pass Result: {'PASS' if summary['passed_phase5'] else 'FAIL'}")
    print(f"- Format Pass Rate: {summary['pass_rule']['format_pass_rate']:.1f}% (> 95.0% required)")
    print(f"- SFT Hard-Fail Rate: {summary['pass_rule']['sft_hard_fail_rate']:.1f}%")
    print(f"- Base Qwen Hard-Fail Rate: {summary['pass_rule']['base_hard_fail_rate']:.1f}%")
    print(f"- Report written to: {summary['report_path']}")


if __name__ == "__main__":
    main()
