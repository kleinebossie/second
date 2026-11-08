"""Single-command evaluation harness for the Second chess assistant.

This module evaluates a model checkpoint or output file on the frozen test set.
It calculates automatic metrics and runs the Gemini Flash judge.
It saves JSON evaluation results and generates a Markdown report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

from eval.baselines import (
    DEFAULT_BASELINES_DIR,
    DEFAULT_TESTSET_PATH,
    generate_baseline,
    load_baseline,
    load_jsonl,
)
from eval.judge import (
    DEFAULT_CACHE_PATH,
    DEFAULT_JUDGE_MODEL,
    GeminiClient,
    GeminiJudge,
    JudgeCache,
    get_gemini_api_key,
    run_judge_eval,
)
from eval.metrics import (
    SOFT_SCORE_WEIGHTS,
    AggregateMetrics,
    EvalMetrics,
    ItemMetric,
    evaluate_completions,
    evaluate_file,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_REPORTS_DIR = Path("eval/reports")


def generate_markdown_report(
    model_name: str,
    metrics: EvalMetrics,
    judge_results: dict[str, Any] | None = None,
    testset_path: str = "eval/testset-1.0.jsonl",
) -> str:
    """Format evaluation outcomes into a structured Markdown report.

    Write sections using ASD-STE100 Simplified Technical English.
    Include summary tables, metric breakdowns, judge outcomes, and sample outputs.
    Return Markdown report text.
    """
    agg = metrics.aggregate
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        f"# Evaluation Report: {model_name}",
        "",
        f"- **Model**: {model_name}",
        f"- **Date**: {now_str}",
        f"- **Test Set**: `{testset_path}`",
        f"- **Total Items**: {agg.total_items}",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Result | Target / Standard |",
        "|---|---|---|",
        f"| Total Test Items | {agg.total_items} | Fixed test set |",
        f"| Hard Check Failure Rate | **{agg.hard_fail_rate:.1f}%** | Lower is better (0% ideal) |",
        f"| Mean Reward | **{agg.mean_reward:.4f}** | Range -1.0 to 1.0 |",
        f"| Mean Soft Score (All) | {agg.mean_soft_score_all:.4f} | Range 0.0 to 1.0 |",
        f"| Mean Soft Score (Passing) | {agg.mean_soft_score_passing:.4f} | Range 0.0 to 1.0 |",
        f"| Verdict Accuracy (Covered) | {agg.verdict_accuracy_covered:.1f}% | Concordance with engine |",
        f"| Length in Target Band | {agg.pct_in_band_overall:.1f}% | Correct detail word count |",
    ]

    if judge_results and "rubric" in judge_results:
        rub_summary = judge_results["rubric"].get("summary", {})
        mean_ov = rub_summary.get("mean_overall", 0.0)
        lines.append(f"| Judge Rubric Overall | **{mean_ov:.2f} / 5.0** | Gemini Flash evaluation |")

    lines.extend([
        "",
        "## 2. Hard Check Hallucination Metrics",
        "",
        "If any hard check fails, the final reward is -1.0.",
        "",
        "| Check Name | Failure Rate (%) | Outcome Description |",
        "|---|---|---|",
    ])

    for check_name, fail_rate in agg.hard_check_fail_rates.items():
        status = "Clean" if fail_rate == 0.0 else f"{fail_rate:.1f}% failed"
        lines.append(f"| `{check_name}` | {fail_rate:.1f}% | {status} |")

    lines.extend([
        "",
        "## 3. Soft Score Breakdown",
        "",
        "Soft scores reward key points, clear suggestions, style fit, and phrasing.",
        "",
        "| Soft Metric | Weight | Average Score (0.0 - 1.0) |",
        "|---|---|---|",
        f"| `key_point` | {SOFT_SCORE_WEIGHTS['key_point']:.2f} | {agg.soft_score_breakdown.get('key_point', 0.0):.4f} |",
        f"| `better_move` | {SOFT_SCORE_WEIGHTS['better_move']:.2f} | {agg.soft_score_breakdown.get('better_move', 0.0):.4f} |",
        f"| `length` | {SOFT_SCORE_WEIGHTS['length']:.2f} | {agg.soft_score_breakdown.get('length', 0.0):.4f} |",
        f"| `level_fit` | {SOFT_SCORE_WEIGHTS['level_fit']:.2f} | {agg.soft_score_breakdown.get('level_fit', 0.0):.4f} |",
        f"| `not_copied` | {SOFT_SCORE_WEIGHTS['not_copied']:.2f} | {agg.soft_score_breakdown.get('not_copied', 0.0):.4f} |",
        f"| `no_repeats` | {SOFT_SCORE_WEIGHTS['no_repeats']:.2f} | {agg.soft_score_breakdown.get('no_repeats', 0.0):.4f} |",
        "",
        "## 4. Subgroup Breakdowns",
        "",
        "### 4.1 Task Slices",
        "",
        "| Task | Count | Hard Fail (%) | Mean Reward | Avg Words | In Band (%) |",
        "|---|---|---|---|---|---|",
    ])

    for task_name in ("explain_move", "explain_position"):
        t_data = agg.breakdowns.get("task", {}).get(task_name, {})
        c = int(t_data.get("count", 0))
        hf = t_data.get("hard_fail_rate", 0.0)
        mr = t_data.get("mean_reward", 0.0)
        aw = t_data.get("avg_word_count", 0.0)
        pb = t_data.get("pct_in_band", 0.0)
        lines.append(f"| `{task_name}` | {c} | {hf:.1f}% | {mr:.4f} | {aw:.1f} | {pb:.1f}% |")

    lines.extend([
        "",
        "### 4.2 Level Slices",
        "",
        "| Level | Count | Hard Fail (%) | Mean Reward | Avg Words | In Band (%) |",
        "|---|---|---|---|---|---|",
    ])

    for lvl_name in ("beginner", "club"):
        l_data = agg.breakdowns.get("level", {}).get(lvl_name, {})
        c = int(l_data.get("count", 0))
        hf = l_data.get("hard_fail_rate", 0.0)
        mr = l_data.get("mean_reward", 0.0)
        aw = l_data.get("avg_word_count", 0.0)
        pb = l_data.get("pct_in_band", 0.0)
        lines.append(f"| `{lvl_name}` | {c} | {hf:.1f}% | {mr:.4f} | {aw:.1f} | {pb:.1f}% |")

    lines.extend([
        "",
        "### 4.3 Detail Length Slices",
        "",
        "| Detail Mode | Target Band | Count | Avg Words | In Target Band (%) |",
        "|---|---|---|---|---|",
        f"| `short` | 25–80 words | {int(agg.breakdowns.get('detail', {}).get('short', {}).get('count', 0))} | {agg.avg_word_count_short:.1f} | {agg.pct_in_band_short:.1f}% |",
        f"| `long` | 80–200 words | {int(agg.breakdowns.get('detail', {}).get('long', {}).get('count', 0))} | {agg.avg_word_count_long:.1f} | {agg.pct_in_band_long:.1f}% |",
        "",
        "### 4.4 Move Verdict Metrics",
        "",
        f"- Items with verdict: {agg.verdict_items_count}",
        f"- Explicit verdict mention rate: {agg.verdict_coverage_rate:.1f}%",
        f"- Overall verdict accuracy: {agg.verdict_accuracy_all:.1f}%",
        f"- Accuracy among mentions: {agg.verdict_accuracy_covered:.1f}%",
        f"- Verdict tone check pass rate: {agg.verdict_tone_pass_rate:.1f}%",
        "",
        "## 5. LLM Judge Results",
        "",
    ])

    if judge_results and "rubric" in judge_results:
        rub = judge_results["rubric"].get("summary", {})
        lines.extend([
            "### 5.1 Rubric Ratings (1 to 5 scale)",
            "",
            "| Dimension | Average Score (1 to 5) | Description |",
            "|---|---|---|",
            f"| `correctness` | {rub.get('mean_correctness', 0.0):.2f} | Move and fact accuracy |",
            f"| `clarity` | {rub.get('mean_clarity', 0.0):.2f} | Simple and readable text |",
            f"| `usefulness` | {rub.get('mean_usefulness', 0.0):.2f} | Chess advice quality |",
            f"| `level_fit` | {rub.get('mean_level_fit', 0.0):.2f} | Appropriate depth and vocabulary |",
            f"| **Overall Average** | **{rub.get('mean_overall', 0.0):.2f}** | Arithmetic mean of dimensions |",
            "",
            "### 5.2 Pairwise Matches (Swap-Consistent)",
            "",
            "The judge evaluates each pair twice with swapped order.",
            "If picks disagree across orders, the judge records a tie.",
            "",
            "| Opponent Baseline | Total Pairs | Wins | Losses | Ties | Win Rate (%) | LC-WR (%) | Win+Tie (%) | Swap Agreement (%) |",
            "|---|---|---|---|---|---|---|---|---|",
        ])

        pairwise_sums = judge_results.get("pairwise", {}).get("summaries", {})
        if pairwise_sums:
            for opp_name, s in pairwise_sums.items():
                tp = s.get("total_pairs", 0)
                w = s.get("candidate_wins", 0)
                l = s.get("opponent_wins", 0)
                t = s.get("ties", 0)
                wr = s.get("candidate_win_rate", 0.0)
                lc_wr = s.get("lc_win_rate", 0.0)
                wtr = s.get("candidate_win_tie_rate", 0.0)
                agr = s.get("swap_agreement_rate", 0.0)
                lines.append(
                    f"| `{opp_name}` | {tp} | {w} | {l} | {t} | {wr:.1f}% | {lc_wr:.1f}% | {wtr:.1f}% | {agr:.1f}% |"
                )
        else:
            lines.append("| *None* | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0.0% |")
            lines.append("")
            lines.append("*Note: No opponent baseline files were available for pairwise comparisons.*")
    else:
        lines.extend([
            "*The runner skipped LLM judge evaluation for this run.*",
            "To enable judge evaluation, provide GEMINI_API_KEY in environment or .env file.",
        ])

    lines.extend([
        "",
        "## 6. Sample Outputs",
        "",
    ])

    sorted_items = sorted(metrics.items, key=lambda x: x.reward, reverse=True)
    top_items = sorted_items[:3]
    bottom_items = sorted_items[-3:] if len(sorted_items) > 3 else []

    lines.append("### 6.1 Highest Scoring Examples")
    lines.append("")
    for it in top_items:
        lines.append(f"#### Item `{it.id}` (Reward: {it.reward:.4f})")
        lines.append(f"- **Task**: `{it.task}`, **Level**: `{it.level}`, **Detail**: `{it.detail}`")
        lines.append(f"- **Passed All Hard**: `{it.passed_all_hard}`")
        lines.append(f"- **Word Count**: {it.word_count}")
        lines.append("```text")
        lines.append(it.completion.strip())
        lines.append("```")
        lines.append("")

    if bottom_items:
        lines.append("### 6.2 Lowest Scoring Examples")
        lines.append("")
        for it in bottom_items:
            lines.append(f"#### Item `{it.id}` (Reward: {it.reward:.4f})")
            lines.append(f"- **Task**: `{it.task}`, **Level**: `{it.level}`, **Detail**: `{it.detail}`")
            lines.append(f"- **Passed All Hard**: `{it.passed_all_hard}`")
            if it.failed_checks:
                lines.append(f"- **Failed Checks**: `{', '.join(it.failed_checks)}`")
            lines.append(f"- **Word Count**: {it.word_count}")
            lines.append("```text")
            lines.append(it.completion.strip())
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def run_eval(
    model_name: str | None = None,
    outputs_path: str | Path | None = None,
    model_path: str | None = None,
    backend: str = "transformers",
    testset_path: str | Path = DEFAULT_TESTSET_PATH,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    report_dir: str | Path = DEFAULT_REPORTS_DIR,
    baselines_dir: str | Path = DEFAULT_BASELINES_DIR,
    opponents: list[str] | None = None,
    skip_judge: bool = False,
    skip_judge_if_no_key: bool = False,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    limit: int | None = None,
) -> dict[str, Any]:
    """Execute evaluation workflow and create evaluation report.

    Load or generate model completions for the test set.
    Compute automatic rule check metrics and soft scores.
    Run Gemini Flash judge if API key is present and not skipped.
    Save evaluation JSON to results directory and report to reports directory.
    Return full evaluation dictionary.
    """
    res_dir = Path(output_dir)
    rep_dir = Path(report_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine model name and completion records
    if outputs_path is not None:
        p_outputs = Path(outputs_path)
        if not p_outputs.is_file():
            raise FileNotFoundError(f"Completions file not found: {p_outputs}")
        resolved_name = model_name or p_outputs.stem.removesuffix("_outputs").removesuffix("_eval")
        records = load_jsonl(p_outputs)
        if not records:
            raise ValueError(f"Completions file is empty: {p_outputs}")
    elif model_path is not None:
        resolved_name = model_name or Path(model_path).stem
        target_outputs_file = res_dir / f"{resolved_name}_outputs.jsonl"
        LOGGER.info("Generating completions for %s using backend %s...", resolved_name, backend)
        records = generate_baseline(
            baseline_name=f"{resolved_name}_outputs",
            backend=backend,
            testset_path=testset_path,
            output_dir=res_dir,
            model_id=model_path,
            limit=limit,
        )
    else:
        raise ValueError("Must provide either --outputs or --model-path.")

    if limit is not None and limit > 0:
        records = records[:limit]

    # 2. Run Automatic Metrics (Phase 4.2)
    LOGGER.info("Computing automatic metrics on %d completions...", len(records))
    eval_metrics = evaluate_completions(records)

    # 3. Handle Opponent Baselines
    opponent_records: dict[str, list[dict[str, Any]]] = {}
    p_baselines = Path(baselines_dir)
    target_opponents: list[str] = []

    if opponents:
        target_opponents = opponents
    elif p_baselines.is_dir():
        # Discover available baselines
        for p in p_baselines.glob("*.jsonl"):
            if p.stem != resolved_name and p.stem != f"{resolved_name}_outputs":
                target_opponents.append(p.stem)

    for opp_name in target_opponents:
        opp_list = load_baseline(opp_name, baselines_dir=p_baselines)
        if opp_list:
            opponent_records[opp_name] = opp_list

    # 4. Run Gemini Judge (Phase 4.3)
    judge_results: dict[str, Any] | None = None
    should_run_judge = not skip_judge

    if should_run_judge:
        api_key = get_gemini_api_key(required=False)
        if not api_key:
            if skip_judge_if_no_key:
                LOGGER.info("GEMINI_API_KEY not found. Skipping judge evaluation.")
                should_run_judge = False
            else:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is not set. "
                    "Set GEMINI_API_KEY in the environment or in a .env file, "
                    "or pass --skip-judge-if-no-key to skip judge evaluation."
                )

    if should_run_judge:
        LOGGER.info("Running Gemini Flash judge (%s)...", judge_model)
        judge_results = run_judge_eval(
            candidate_records=records,
            opponent_records=opponent_records if opponent_records else None,
            judge_model=judge_model,
            cache_path=cache_path,
        )

    # 5. Format and Save Outputs
    eval_output = {
        "model_name": resolved_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "testset": str(testset_path),
        "total_items": eval_metrics.aggregate.total_items,
        "automatic_metrics": eval_metrics.to_dict(),
        "judge_results": judge_results,
    }

    eval_json_path = res_dir / f"{resolved_name}_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)
    LOGGER.info("Saved evaluation JSON to %s", eval_json_path)

    report_md = generate_markdown_report(
        model_name=resolved_name,
        metrics=eval_metrics,
        judge_results=judge_results,
        testset_path=str(testset_path),
    )
    report_path = rep_dir / f"{resolved_name}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    LOGGER.info("Saved Markdown report to %s", report_path)

    return eval_output


def main() -> None:
    """CLI entry point for the single-command evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Run complete evaluation on testset (metrics + judge)."
    )
    parser.add_argument("--outputs", type=str, default=None, help="Path to pre-generated JSONL completions")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model weights or checkpoint")
    parser.add_argument("--model-name", type=str, default=None, help="Evaluation identifier name")
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "openai", "gemini", "rule-baseline"],
        help="Inference backend for model-path generation",
    )
    parser.add_argument("--testset", type=str, default=str(DEFAULT_TESTSET_PATH), help="Test set JSONL path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Results output directory")
    parser.add_argument("--report-dir", type=str, default=str(DEFAULT_REPORTS_DIR), help="Reports output directory")
    parser.add_argument(
        "--baselines-dir",
        type=str,
        default=str(DEFAULT_BASELINES_DIR),
        help="Baselines JSONL directory",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default=None,
        help="Comma-separated opponent baseline names (e.g. base-qwen-2b,gemini-flash)",
    )
    parser.add_argument("--skip-judge", action="store_true", help="Skip Gemini judge evaluation")
    parser.add_argument(
        "--skip-judge-if-no-key",
        action="store_true",
        help="Skip judge automatically if GEMINI_API_KEY is not set",
    )
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model name")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH), help="Judge cache file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items evaluated")

    args = parser.parse_args()

    opponents_list = [o.strip() for o in args.opponents.split(",")] if args.opponents else None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        results = run_eval(
            model_name=args.model_name,
            outputs_path=args.outputs,
            model_path=args.model_path,
            backend=args.backend,
            testset_path=args.testset,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            baselines_dir=args.baselines_dir,
            opponents=opponents_list,
            skip_judge=args.skip_judge,
            skip_judge_if_no_key=args.skip_judge_if_no_key,
            judge_model=args.judge_model,
            cache_path=args.cache_path,
            limit=args.limit,
        )
        print("\n" + "=" * 50)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 50)
        agg = results["automatic_metrics"]["aggregate"]
        print(f"Model: {results['model_name']}")
        print(f"Total items: {agg['total_items']}")
        print(f"Hard check fail rate: {agg['hard_fail_rate']}%")
        print(f"Mean reward: {agg['mean_reward']}")
        print(f"Verdict accuracy (covered): {agg['verdict_accuracy_covered']}%")
        print(f"Outputs in length band: {agg['pct_in_band_overall']}%")
        if results.get("judge_results"):
            rub = results["judge_results"]["rubric"]["summary"]
            print(f"Judge rubric overall: {rub['mean_overall']} / 5.0")
        print("=" * 50)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
