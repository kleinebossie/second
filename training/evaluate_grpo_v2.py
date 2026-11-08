"""Stage T4 GRPO v2 evaluation runner and report generator.

This module evaluates fine-tuned GRPO v2 checkpoints on the development set.
It verifies Stage T4 pass gates for hard-fail rate, length fit, and key points.
It produces evaluation summaries in ASD-STE100 Simplified Technical English.
All docstrings follow ASD-STE100 rules.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess

from rewards.scorer import compute_reward

LOGGER = logging.getLogger(__name__)

DEFAULT_DEVSET_PATH: Path = Path("eval/devset-1.0.jsonl")
DEFAULT_OUTPUTS_PATH: Path = Path("eval/results/grpo_v2_dev_outputs.jsonl")
DEFAULT_REPORTS_DIR: Path = Path("eval/reports")
DEFAULT_LEDGER_PATH: Path = Path("eval/results/experiment_ledger.jsonl")

# Stage T4 Pass Gates from Section 3 & Section 8 of TRAINING_PLAN.md
GATE_MAX_HARD_FAIL_RATE: float = 5.0  # <= 5.0%
GATE_MIN_IN_BAND_LENGTH: float = 80.0  # >= 80.0%
GATE_MAX_FIRST_PERSON_RATE: float = 2.0  # <= 2.0%
GATE_MIN_KEY_POINT: float = 0.70  # >= 0.70
GATE_MIN_BETTER_MOVE: float = 0.75  # >= 0.75


@dataclass(frozen=True)
class StageT4PassResult:
    """Outcome of Stage T4 GRPO v2 gate verification."""

    hard_fail_rate: float
    passed_hard_fail: bool
    overall_length_in_band_rate: float
    short_length_in_band_rate: float
    long_length_in_band_rate: float
    passed_length: bool
    first_person_rate: float
    passed_first_person: bool
    mean_key_point: float
    passed_key_point: bool
    mean_better_move: float
    passed_better_move: bool
    mean_reward: float
    avg_word_count: float
    passed_all_gates: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert pass result to dictionary format."""
        return asdict(self)


def evaluate_grpo_v2_item(
    dev_item: dict[str, Any],
    completion: str,
) -> dict[str, Any]:
    """Evaluate one model completion on a development set item with Reward v2.

    Compute Stage T4 Reward v2 (graded penalties + multiplicative length).
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

    coach_voice_passed = reward_res.hard_checks.get("coach_voice")
    has_first_person = not (coach_voice_passed.passed if coach_voice_passed else True)

    return {
        "id": dev_item.get("id"),
        "task": dev_item.get("task", "explain_move"),
        "level": level,
        "detail": detail,
        "verdict": verdict,
        "completion": completion,
        "word_count": word_count,
        "in_length_band": in_band,
        "has_first_person": has_first_person,
        "reward": reward_res.reward,
        "passed_all_hard": reward_res.passed_all_hard,
        "hard_checks": {k: v.passed for k, v in reward_res.hard_checks.items()},
        "failed_checks": [k for k, v in reward_res.hard_checks.items() if not v.passed],
        "soft_scores": reward_res.soft_scores,
    }


def compute_grpo_v2_summary(
    results: list[dict[str, Any]],
) -> tuple[dict[str, Any], StageT4PassResult]:
    """Compute aggregate evaluation metrics across dev set results.

    Compute hard-fail rate, check-level breakdown, and length rates.
    Compute mean Reward v2, key point score, and better move score.
    Verify Stage T4 pass gates.
    Return tuple of (summary_dict, pass_result).
    """
    total = len(results)
    if total == 0:
        empty_res = StageT4PassResult(
            hard_fail_rate=100.0,
            passed_hard_fail=False,
            overall_length_in_band_rate=0.0,
            short_length_in_band_rate=0.0,
            long_length_in_band_rate=0.0,
            passed_length=False,
            first_person_rate=100.0,
            passed_first_person=False,
            mean_key_point=0.0,
            passed_key_point=False,
            mean_better_move=0.0,
            passed_better_move=False,
            mean_reward=0.0,
            avg_word_count=0.0,
            passed_all_gates=False,
        )
        return {"total": 0}, empty_res

    hard_fails = sum(1 for r in results if not r["passed_all_hard"])
    hard_fail_rate = (hard_fails / total) * 100.0

    first_person_count = sum(1 for r in results if r.get("has_first_person", False))
    first_person_rate = (first_person_count / total) * 100.0

    in_band_total = sum(1 for r in results if r["in_length_band"])
    overall_len_rate = (in_band_total / total) * 100.0

    short_items = [r for r in results if r["detail"] == "short"]
    long_items = [r for r in results if r["detail"] == "long"]

    short_in_band = sum(1 for r in short_items if r["in_length_band"])
    short_rate = (short_in_band / len(short_items) * 100.0) if short_items else 100.0

    long_in_band = sum(1 for r in long_items if r["in_length_band"])
    long_rate = (long_in_band / len(long_items) * 100.0) if long_items else 100.0

    mean_key_point = sum(r["soft_scores"].get("key_point", 0.0) for r in results) / total
    mean_better_move = sum(r["soft_scores"].get("better_move", 0.0) for r in results) / total
    mean_reward = sum(r["reward"] for r in results) / total
    avg_words = sum(r["word_count"] for r in results) / total

    # Check failure breakdown
    check_fail_counts: dict[str, int] = {}
    for r in results:
        for fc in r.get("failed_checks", []):
            check_fail_counts[fc] = check_fail_counts.get(fc, 0) + 1

    pass_hard = hard_fail_rate <= GATE_MAX_HARD_FAIL_RATE
    pass_len = overall_len_rate >= GATE_MIN_IN_BAND_LENGTH
    pass_fp = first_person_rate <= GATE_MAX_FIRST_PERSON_RATE
    pass_kp = mean_key_point >= GATE_MIN_KEY_POINT
    pass_bm = mean_better_move >= GATE_MIN_BETTER_MOVE
    pass_all = pass_hard and pass_len and pass_fp and pass_kp and pass_bm

    pass_result = StageT4PassResult(
        hard_fail_rate=round(hard_fail_rate, 2),
        passed_hard_fail=pass_hard,
        overall_length_in_band_rate=round(overall_len_rate, 2),
        short_length_in_band_rate=round(short_rate, 2),
        long_length_in_band_rate=round(long_rate, 2),
        passed_length=pass_len,
        first_person_rate=round(first_person_rate, 2),
        passed_first_person=pass_fp,
        mean_key_point=round(mean_key_point, 4),
        passed_key_point=pass_kp,
        mean_better_move=round(mean_better_move, 4),
        passed_better_move=pass_bm,
        mean_reward=round(mean_reward, 4),
        avg_word_count=round(avg_words, 2),
        passed_all_gates=pass_all,
    )

    summary = {
        "total_evaluated": total,
        "hard_fail_count": hard_fails,
        "hard_fail_rate": pass_result.hard_fail_rate,
        "check_failure_counts": check_fail_counts,
        "first_person_rate": pass_result.first_person_rate,
        "overall_in_band_length_rate": pass_result.overall_length_in_band_rate,
        "short_in_band_length_rate": pass_result.short_length_in_band_rate,
        "long_in_band_length_rate": pass_result.long_length_in_band_rate,
        "avg_word_count": pass_result.avg_word_count,
        "mean_key_point": pass_result.mean_key_point,
        "mean_better_move": pass_result.mean_better_move,
        "mean_reward": pass_result.mean_reward,
        "passed_all_gates": pass_result.passed_all_gates,
    }

    return summary, pass_result


def generate_grpo_v2_report(
    summary: dict[str, Any],
    pass_result: StageT4PassResult,
    experiment_id: str = "RL-B2",
    model_name: str = "Second-1.0-2B-GRPO-v2",
) -> str:
    """Generate Markdown report for Stage T4 GRPO v2 ablation.

    Format metrics in clean tables comparing against SFT-A1 baseline.
    Use ASD-STE100 technical English.
    Return Markdown report text.
    """
    status_str = "PASSED" if pass_result.passed_all_gates else "FAILED"
    check_counts = summary.get("check_failure_counts", {})

    report_lines = [
        f"# Stage T4 GRPO v2 Evaluation Report: {experiment_id}",
        "",
        f"**Model**: `{model_name}`  ",
        f"**Starting Policy**: `SFT-A1` (`r=32`)  ",
        f"**Evaluation Set**: `eval/devset-1.0.jsonl` ({summary.get('total_evaluated', 0)} items)  ",
        f"**Gate Status**: **{status_str}**",
        "",
        "## 1. Stage T4 Pass Gates vs SFT-A1 Baseline",
        "",
        "| Gate Dimension | Stage T4 Target | SFT-A1 Baseline | GRPO v2 Result | Gate Outcome |",
        "|---|---|---|---|---|",
        f"| **Hard Fail Rate** | &le; 5.0% | 14.67% (22 items) | **{pass_result.hard_fail_rate}%** ({summary.get('hard_fail_count', 0)} items) | {'PASS' if pass_result.passed_hard_fail else 'FAIL'} |",
        f"| **Length In-Band (Overall)** | &ge; 80.0% | 99.33% | **{pass_result.overall_length_in_band_rate}%** | {'PASS' if pass_result.passed_length else 'FAIL'} |",
        f"| **First-Person Tone Rate** | &le; 2.0% | 0.00% | **{pass_result.first_person_rate}%** | {'PASS' if pass_result.passed_first_person else 'FAIL'} |",
        f"| **Key Point Score** | &ge; 0.70 | 0.9867 | **{pass_result.mean_key_point:.4f}** | {'PASS' if pass_result.passed_key_point else 'FAIL'} |",
        f"| **Better Move Score** | &ge; 0.75 | 0.9600 | **{pass_result.mean_better_move:.4f}** | {'PASS' if pass_result.passed_better_move else 'FAIL'} |",
        f"| **Mean Word Count** | 40\u201380 words | 64.0 words | **{pass_result.avg_word_count:.1f} words** | TRACKED |",
        f"| **Mean Reward v2** | Informational | 0.6920 | **{pass_result.mean_reward:.4f}** | TRACKED |",
        "",
        "## 2. Hard Check Failure Breakdown",
        "",
        "| Hard Check | Failed Items |",
        "|---|---|",
    ]

    if check_counts:
        for cname, cnt in sorted(check_counts.items(), key=lambda x: (-x[1], x[0])):
            report_lines.append(f"| `{cname}` | {cnt} |")
    else:
        report_lines.append("| None (All Passed) | 0 |")

    report_lines.extend([
        "",
        "## 3. Gate Decision",
        "",
    ])

    if pass_result.passed_all_gates:
        report_lines.append(
            "The model satisfies all Stage T4 pass gates on the development set. "
            "It is approved to proceed to Stage T5 and Stage T6 final verification."
        )
    else:
        report_lines.append(
            "The model did not satisfy all Stage T4 pass gates. "
            "Review check failure details and run the next ablation or enable the CMDP multiplier."
        )

    report_lines.append("")
    return "\n".join(report_lines)


def append_experiment_ledger(
    ledger_path: Path,
    experiment_id: str,
    pass_result: StageT4PassResult,
    config_diff: dict[str, Any] | None = None,
) -> None:
    """Append one experiment entry to eval/results/experiment_ledger.jsonl."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "exp_id": experiment_id,
        "stage": "T4",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_checkpoint": "SFT-A1",
        "config_diff": config_diff or {"experiment_id": experiment_id},
        "eval_split": "devset-1.0",
        "thinking_mode": "off",
        "metrics": {
            "hard_fail_rate": round(pass_result.hard_fail_rate / 100.0, 4),
            "mean_reward_v2": pass_result.mean_reward,
            "in_band_pct": round(pass_result.overall_length_in_band_rate / 100.0, 4),
            "avg_words": pass_result.avg_word_count,
            "key_point": pass_result.mean_key_point,
            "better_move": pass_result.mean_better_move,
            "first_person_pct": round(pass_result.first_person_rate / 100.0, 4),
        },
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entry point for Stage T4 GRPO v2 evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO v2 completions on the development set."
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
        default="RL-B2",
        help="Experiment identifier (e.g. RL-B1, RL-B2, RL-B3, RL-B4)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory to save Markdown report",
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default=str(DEFAULT_LEDGER_PATH),
        help="Path to experiment ledger JSONL",
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
            text = item.get("completion") or item.get("output") or item.get("comment") or ""
            completions_by_id[cid] = text

    results: list[dict[str, Any]] = []
    for d in dev_items:
        cid = str(d.get("id"))
        comp = completions_by_id.get(cid, "")
        evaluated = evaluate_grpo_v2_item(d, comp)
        results.append(evaluated)

    summary, pass_res = compute_grpo_v2_summary(results)
    report_md = generate_grpo_v2_report(
        summary=summary,
        pass_result=pass_res,
        experiment_id=args.experiment_id,
    )

    rep_dir = Path(args.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    report_file = rep_dir / f"stage_t4_{args.experiment_id.lower()}_report.md"
    report_file.write_text(report_md, encoding="utf-8")

    append_experiment_ledger(
        ledger_path=Path(args.ledger),
        experiment_id=args.experiment_id,
        pass_result=pass_res,
    )

    print(f"Stage T4 Evaluation: {args.experiment_id}")
    print(f"Hard Fail Rate: {pass_res.hard_fail_rate}% (target <= 5.0%, SFT-A1 was 14.67%)")
    print(f"Overall In-Band Length: {pass_res.overall_length_in_band_rate}% (target >= 80.0%)")
    print(f"First-Person Tone: {pass_res.first_person_rate}% (target <= 2.0%)")
    print(f"Key Point Score: {pass_res.mean_key_point} (target >= 0.70)")
    print(f"Better Move Score: {pass_res.mean_better_move} (target >= 0.75)")
    print(f"Mean Word Count: {pass_res.avg_word_count} words")
    print(f"Mean Reward v2: {pass_res.mean_reward}")
    print(f"All Gates Passed: {pass_res.passed_all_gates}")
    print(f"Report written to: {report_file}")


if __name__ == "__main__":
    main()
