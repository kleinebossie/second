"""Stage T3 SFT v2 evaluation runner and report generator.

This module evaluates fine-tuned SFT v2 checkpoints on the development set.
It verifies Stage T3 pass gates for hard-fail rate, length fit, and key points.
It produces evaluation summaries in ASD-STE100 Simplified Technical English.
All docstrings follow ASD-STE100 rules.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess

from rewards.scorer import compute_reward

LOGGER = logging.getLogger(__name__)

DEFAULT_DEVSET_PATH: Path = Path("eval/devset-1.0.jsonl")
DEFAULT_OUTPUTS_PATH: Path = Path("eval/results/sft_v2_dev_outputs.jsonl")
DEFAULT_REPORTS_DIR: Path = Path("eval/reports")

# Stage T3 Pass Gates from Section 7.2 of TRAINING_PLAN.md
GATE_MAX_HARD_FAIL_RATE: float = 6.0  # <= 6.0%
GATE_MIN_IN_BAND_LENGTH: float = 75.0  # >= 75.0%
GATE_MIN_SLICE_LENGTH: float = 70.0  # both short and long >= 70.0%
GATE_MIN_KEY_POINT: float = 0.65  # >= 0.65


@dataclass(frozen=True)
class StageT3PassResult:
    """Outcome of Stage T3 SFT v2 gate verification."""

    hard_fail_rate: float
    passed_hard_fail: bool
    overall_length_in_band_rate: float
    short_length_in_band_rate: float
    long_length_in_band_rate: float
    passed_length: bool
    mean_key_point: float
    passed_key_point: bool
    mean_reward: float
    passed_all_gates: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert pass result to dictionary format."""
        return asdict(self)


def evaluate_sft_v2_item(
    dev_item: dict[str, Any],
    completion: str,
) -> dict[str, Any]:
    """Evaluate one model completion on a development set item.

    Compute Stage T1 hard rule checks and scalar reward.
    Determine whether explanation length is inside designated target band.
    Return evaluation dictionary.
    """
    fen = dev_item.get("fen", chess.STARTING_FEN)
    level = dev_item.get("level", "club")
    detail = dev_item.get("detail", "short")
    verdict = dev_item.get("verdict")
    move_uci = dev_item.get("move_uci")
    facts = dev_item.get("facts", [])
    user_msg = dev_item.get("user_message", "")

    reward_res = compute_reward(
        text=completion,
        fen=fen,
        level=level,
        detail=detail,
        verdict=verdict,
        move_uci=move_uci,
        facts=facts,
        prompt_text=user_msg,
        reward_version="v2",
    )

    words = completion.split()
    word_count = len(words)
    if detail == "long":
        in_band = 80 <= word_count <= 200
    else:
        in_band = 25 <= word_count <= 80

    return {
        "id": dev_item.get("id"),
        "task": dev_item.get("task", "explain_move"),
        "level": level,
        "detail": detail,
        "verdict": verdict,
        "completion": completion,
        "word_count": word_count,
        "in_length_band": in_band,
        "reward": reward_res.reward,
        "passed_all_hard": reward_res.passed_all_hard,
        "hard_checks": {k: v.passed for k, v in reward_res.hard_checks.items()},
        "failed_checks": [k for k, v in reward_res.hard_checks.items() if not v.passed],
        "soft_scores": reward_res.soft_scores,
    }


def compute_sft_v2_summary(
    results: list[dict[str, Any]],
) -> tuple[dict[str, Any], StageT3PassResult]:
    """Compute aggregate evaluation metrics across dev set results.

    Compute hard-fail rate and slice-level length rates.
    Compute mean reward and key point score.
    Verify Stage T3 pass gates.
    Return tuple of (summary_dict, pass_result).
    """
    total = len(results)
    if total == 0:
        empty_res = StageT3PassResult(
            hard_fail_rate=100.0,
            passed_hard_fail=False,
            overall_length_in_band_rate=0.0,
            short_length_in_band_rate=0.0,
            long_length_in_band_rate=0.0,
            passed_length=False,
            mean_key_point=0.0,
            passed_key_point=False,
            mean_reward=0.0,
            passed_all_gates=False,
        )
        return {"total": 0}, empty_res

    hard_fails = sum(1 for r in results if not r["passed_all_hard"])
    hard_fail_rate = (hard_fails / total) * 100.0

    in_band_total = sum(1 for r in results if r["in_length_band"])
    overall_len_rate = (in_band_total / total) * 100.0

    short_items = [r for r in results if r["detail"] == "short"]
    long_items = [r for r in results if r["detail"] == "long"]

    short_in_band = sum(1 for r in short_items if r["in_length_band"])
    short_rate = (short_in_band / len(short_items) * 100.0) if short_items else 100.0

    long_in_band = sum(1 for r in long_items if r["in_length_band"])
    long_rate = (long_in_band / len(long_items) * 100.0) if long_items else 100.0

    mean_key_point = sum(r["soft_scores"].get("key_point", 0.0) for r in results) / total
    mean_reward = sum(r["reward"] for r in results) / total

    # Check gates
    pass_hard = hard_fail_rate <= GATE_MAX_HARD_FAIL_RATE
    pass_len = (
        overall_len_rate >= GATE_MIN_IN_BAND_LENGTH
        and short_rate >= GATE_MIN_SLICE_LENGTH
        and long_rate >= GATE_MIN_SLICE_LENGTH
    )
    pass_kp = mean_key_point >= GATE_MIN_KEY_POINT
    pass_all = pass_hard and pass_len and pass_kp

    pass_result = StageT3PassResult(
        hard_fail_rate=round(hard_fail_rate, 2),
        passed_hard_fail=pass_hard,
        overall_length_in_band_rate=round(overall_len_rate, 2),
        short_length_in_band_rate=round(short_rate, 2),
        long_length_in_band_rate=round(long_rate, 2),
        passed_length=pass_len,
        mean_key_point=round(mean_key_point, 4),
        passed_key_point=pass_kp,
        mean_reward=round(mean_reward, 4),
        passed_all_gates=pass_all,
    )

    summary = {
        "total_evaluated": total,
        "hard_fail_count": hard_fails,
        "hard_fail_rate": pass_result.hard_fail_rate,
        "overall_in_band_length_rate": pass_result.overall_length_in_band_rate,
        "short_in_band_length_rate": pass_result.short_length_in_band_rate,
        "long_in_band_length_rate": pass_result.long_length_in_band_rate,
        "mean_key_point": pass_result.mean_key_point,
        "mean_reward": pass_result.mean_reward,
        "passed_all_gates": pass_result.passed_all_gates,
    }

    return summary, pass_result


def generate_sft_v2_report(
    summary: dict[str, Any],
    pass_result: StageT3PassResult,
    experiment_id: str = "SFT-A1",
    model_name: str = "Second-1.0-2B-SFT-v2",
) -> str:
    """Generate Markdown report for Stage T3 SFT ablation.

    Format metrics in clean tables.
    Use ASD-STE100 technical English.
    Return Markdown report text.
    """
    status_str = "PASSED" if pass_result.passed_all_gates else "FAILED"

    report_lines = [
        f"# Stage T3 SFT v2 Evaluation Report: {experiment_id}",
        "",
        f"**Model**: `{model_name}`  ",
        f"**Evaluation Set**: `eval/devset-1.0.jsonl` ({summary.get('total_evaluated', 0)} items)  ",
        f"**Gate Status**: **{status_str}**",
        "",
        "## Stage T3 Pass Gates",
        "",
        "| Gate Dimension | Metric Target | Actual Result | Gate Outcome |",
        "|---|---|---|---|",
        f"| **Hard Fail Rate** | &le; 6.0% | {pass_result.hard_fail_rate}% | {'PASS' if pass_result.passed_hard_fail else 'FAIL'} |",
        f"| **Length In-Band (Overall)** | &ge; 75.0% | {pass_result.overall_length_in_band_rate}% | {'PASS' if pass_result.overall_length_in_band_rate >= GATE_MIN_IN_BAND_LENGTH else 'FAIL'} |",
        f"| **Length In-Band (Short)** | &ge; 70.0% | {pass_result.short_length_in_band_rate}% | {'PASS' if pass_result.short_length_in_band_rate >= GATE_MIN_SLICE_LENGTH else 'FAIL'} |",
        f"| **Length In-Band (Long)** | &ge; 70.0% | {pass_result.long_length_in_band_rate}% | {'PASS' if pass_result.long_length_in_band_rate >= GATE_MIN_SLICE_LENGTH else 'FAIL'} |",
        f"| **Key Point Score** | &ge; 0.65 | {pass_result.mean_key_point:.4f} | {'PASS' if pass_result.passed_key_point else 'FAIL'} |",
        f"| **Mean Scalar Reward** | Informational | {pass_result.mean_reward:.4f} | TRACKED |",
        "",
        "## Gate Decision",
        "",
    ]

    if pass_result.passed_all_gates:
        report_lines.append(
            "The model satisfies all Stage T3 pass gates. "
            "It is approved to proceed to Stage T4 GRPO optimization."
        )
    else:
        report_lines.append(
            "The model did not satisfy all Stage T3 pass gates. "
            "Review gate failure details before proceeding to Stage T4."
        )

    report_lines.append("")
    return "\n".join(report_lines)


def main() -> None:
    """CLI entry point for Stage T3 SFT v2 evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate SFT v2 completions on the development set."
    )
    parser.add_argument(
        "--devset",
        type=str,
        default=str(DEFAULT_DEVSET_PATH),
        help="Path to development set JSONL",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        required=True,
        help="Path to generated outputs JSONL",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="SFT-A1",
        help="Experiment identifier (e.g. SFT-A1, SFT-A4, SFT-A5)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory to save Markdown report",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    devset_path = Path(args.devset)
    outputs_path = Path(args.outputs)

    if not devset_path.is_file() and (PROJECT_ROOT / devset_path).is_file():
        devset_path = PROJECT_ROOT / devset_path

    if not outputs_path.is_file() and (PROJECT_ROOT / outputs_path).is_file():
        outputs_path = PROJECT_ROOT / outputs_path

    dev_items: list[dict[str, Any]] = []
    with open(devset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_items.append(json.loads(line))

    completions_by_id: dict[str, str] = {}
    with open(outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cid = str(item.get("id"))
            text = item.get("completion") or item.get("comment") or ""
            completions_by_id[cid] = text

    results: list[dict[str, Any]] = []
    for d in dev_items:
        cid = str(d.get("id"))
        comp = completions_by_id.get(cid, "")
        evaluated = evaluate_sft_v2_item(d, comp)
        results.append(evaluated)

    summary, pass_res = compute_sft_v2_summary(results)
    report_md = generate_sft_v2_report(
        summary=summary,
        pass_result=pass_res,
        experiment_id=args.experiment_id,
    )

    rep_dir = Path(args.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    report_file = rep_dir / f"stage_t3_{args.experiment_id.lower()}_report.md"
    report_file.write_text(report_md, encoding="utf-8")

    print(f"Stage T3 Evaluation: {args.experiment_id}")
    print(f"Hard Fail Rate: {pass_res.hard_fail_rate}% (target <= 6.0%)")
    print(f"Overall In-Band Length: {pass_res.overall_length_in_band_rate}% (target >= 75.0%)")
    print(f"Key Point Score: {pass_res.mean_key_point} (target >= 0.65)")
    print(f"Mean Reward: {pass_res.mean_reward}")
    print(f"All Gates Passed: {pass_res.passed_all_gates}")
    print(f"Report written to: {report_file}")


if __name__ == "__main__":
    main()
