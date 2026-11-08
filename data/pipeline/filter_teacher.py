"""Rejection sampling and verification module for Stage T2 teacher distillation.

This module evaluates candidate completions against Stage T1 reward gates.
It verifies hard rule constraints on both thinking traces and coaching prose.
It filters multi-sample completions using strict quality thresholds.
It emits verified student training records for SFT v2.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess

from data.pipeline.synthesize_teacher import (
    emit_student_records,
    parse_teacher_completion,
)
from rewards.checks import (
    CheckResult,
    check_coach_voice,
    check_grounded_moves,
    check_legal_moves,
    check_opening_claims,
    check_piece_on_square,
    check_relational_claims,
)
from rewards.scorer import RewardResult, compute_reward

LOGGER = logging.getLogger(__name__)

# Acceptance gate thresholds per Section 6.4 of TRAINING_PLAN.md
MIN_REWARD_GATE: float = 0.82
MIN_KEY_POINT_GATE: float = 0.75
MIN_BETTER_MOVE_GATE: float = 1.00
MIN_NOT_COPIED_GATE: float = 0.85

SHORT_MIN_WORDS: int = 40
SHORT_MAX_WORDS: int = 75
LONG_MIN_WORDS: int = 95
LONG_MAX_WORDS: int = 175


def evaluate_candidate_completion(
    prompt_item: dict[str, Any],
    raw_completion: str,
) -> dict[str, Any]:
    """Evaluate one candidate completion against Stage T2 rejection gates.

    Parse thinking trace and explanation.
    Run hard checks on the thinking trace.
    Compute reward and soft scores on the explanation prose.
    Verify all acceptance gates.
    Return evaluation result dictionary.
    """
    parsed = parse_teacher_completion(raw_completion)
    trace = parsed.get("trace", "").strip()
    explanation = parsed.get("explanation", "").strip()

    fen = prompt_item.get("fen", chess.STARTING_FEN)
    task = prompt_item.get("task", "explain_move")
    level = prompt_item.get("level", "club")
    detail = prompt_item.get("detail", "short")
    verdict = prompt_item.get("verdict")
    move_uci = prompt_item.get("move_uci")
    facts = prompt_item.get("facts", [])
    user_msg = prompt_item.get("user_message", "")

    failed_gates: list[str] = []

    # 1. Verify thinking block format
    has_thinking_block = bool(trace)
    if not has_thinking_block:
        failed_gates.append("missing_thinking_tags")

    # 2. Build grounding source for grounded move checks
    grounding_parts: list[str] = []
    if user_msg:
        grounding_parts.append(user_msg)
    if facts:
        grounding_parts.extend(facts)
    grounding_source = "\n".join(grounding_parts)

    # 3. Verify hard checks on thinking trace
    trace_hard_checks: dict[str, CheckResult] = {}
    if trace:
        trace_hard_checks["trace_legal_moves"] = check_legal_moves(
            text=trace,
            fen=fen,
            move_uci=move_uci,
            facts=facts,
        )
        trace_hard_checks["trace_grounded_moves"] = check_grounded_moves(
            text=trace,
            prompt_or_facts=grounding_source,
        )
        trace_hard_checks["trace_piece_on_square"] = check_piece_on_square(
            text=trace,
            fen=fen,
            move_uci=move_uci,
        )
        trace_hard_checks["trace_coach_voice"] = check_coach_voice(text=trace)
        trace_hard_checks["trace_opening_claims"] = check_opening_claims(
            text=trace,
            facts=facts,
        )
        trace_hard_checks["trace_relational_claims"] = check_relational_claims(
            text=trace,
            fen=fen,
            move_uci=move_uci,
            facts=facts,
        )

    trace_passed_hard = all(r.passed for r in trace_hard_checks.values())
    if not trace_passed_hard:
        failed_gates.append("trace_hard_violation")

    # 4. Compute reward on final explanation prose
    reward_res = compute_reward(
        text=explanation,
        fen=fen,
        level=level,
        detail=detail,
        verdict=verdict,
        move_uci=move_uci,
        facts=facts,
        prompt_text=user_msg,
        reward_version="v2",
    )

    if not reward_res.passed_all_hard:
        failed_gates.append("explanation_hard_violation")

    # 5. Check length gate (40 to 75 for short, 95 to 175 for long)
    word_count = len(explanation.split())
    if detail == "long":
        length_in_band = LONG_MIN_WORDS <= word_count <= LONG_MAX_WORDS
    else:
        length_in_band = SHORT_MIN_WORDS <= word_count <= SHORT_MAX_WORDS

    length_score = reward_res.soft_scores.get("length", 0.0)
    if not (length_in_band and length_score >= 1.0):
        failed_gates.append("length_gate")

    # 6. Check key point gate (>= 0.75)
    key_point_score = reward_res.soft_scores.get("key_point", 0.0)
    if key_point_score < MIN_KEY_POINT_GATE:
        failed_gates.append("key_point_gate")

    # 7. Check better move gate (1.0 for bad moves)
    better_move_score = reward_res.soft_scores.get("better_move", 0.0)
    if better_move_score < MIN_BETTER_MOVE_GATE:
        failed_gates.append("better_move_gate")

    # 8. Check not copied gate (>= 0.85)
    not_copied_score = reward_res.soft_scores.get("not_copied", 0.0)
    if not_copied_score < MIN_NOT_COPIED_GATE:
        failed_gates.append("not_copied_gate")

    # 9. Check total reward gate (>= 0.82)
    if reward_res.reward < MIN_REWARD_GATE:
        failed_gates.append("reward_gate")

    is_accepted = (len(failed_gates) == 0)

    return {
        "accepted": is_accepted,
        "reward": reward_res.reward,
        "word_count": word_count,
        "length_in_band": length_in_band,
        "failed_gates": failed_gates,
        "parsed": parsed,
        "trace_hard_checks": {k: v.passed for k, v in trace_hard_checks.items()},
        "explanation_hard_checks": {
            k: v.passed for k, v in reward_res.hard_checks.items()
        },
        "soft_scores": reward_res.soft_scores,
    }


def filter_teacher_completions(
    prompts: list[dict[str, Any]],
    completions_by_id: dict[str, list[str]],
    max_candidates_per_prompt: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Filter teacher completions across prompt records with multi-sample rejection.

    Evaluate up to max_candidates_per_prompt for each prompt.
    Stop early when a candidate passes all acceptance gates.
    Emit non-thinking and thinking student records for accepted candidates.
    Return tuple of (non_thinking_records, thinking_records, summary_stats).
    """
    accepted_non_thinking: list[dict[str, Any]] = []
    accepted_thinking: list[dict[str, Any]] = []

    gate_failures = Counter()
    total_evaluated = 0
    accepted_count = 0

    for item in prompts:
        p_id = item.get("id", "")
        candidates = completions_by_id.get(p_id, [])
        if not candidates:
            continue

        accepted_for_prompt = False
        candidates_to_test = candidates[:max_candidates_per_prompt]

        for cand_text in candidates_to_test:
            total_evaluated += 1
            eval_res = evaluate_candidate_completion(item, cand_text)

            if eval_res["accepted"]:
                accepted_for_prompt = True
                accepted_count += 1
                rec_nt, rec_th = emit_student_records(item, eval_res["parsed"])
                rec_nt["teacher_reward"] = eval_res["reward"]
                rec_th["teacher_reward"] = eval_res["reward"]
                accepted_non_thinking.append(rec_nt)
                accepted_thinking.append(rec_th)
                break
            else:
                for fg in eval_res["failed_gates"]:
                    gate_failures[fg] += 1

    total_prompts = len(prompts)
    acceptance_rate = (accepted_count / total_prompts) if total_prompts else 0.0

    summary_stats = {
        "total_prompts": total_prompts,
        "total_completions_evaluated": total_evaluated,
        "accepted_prompts": accepted_count,
        "acceptance_rate": round(acceptance_rate, 4),
        "gate_failure_breakdown": dict(gate_failures),
    }

    LOGGER.info("Filter summary: %s", json.dumps(summary_stats, indent=2))
    return accepted_non_thinking, accepted_thinking, summary_stats


def save_filtered_datasets(
    records: list[dict[str, Any]],
    output_path: Path | str,
) -> int:
    """Save verified records to JSONL file.

    Create parent directory if it does not exist.
    Return number of written records.
    """
    out_file = Path(output_path)
    if not out_file.is_absolute() and not out_file.parent.exists():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOGGER.info("Saved %d records to %s", len(records), out_file)
    return len(records)


def main() -> None:
    """CLI entry point for teacher rejection filtering."""
    parser = argparse.ArgumentParser(
        description="Filter teacher model completions with Stage T1 acceptance gates."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/processed/teacher_prompts.jsonl",
        help="Path to input teacher prompts JSONL",
    )
    parser.add_argument(
        "--completions",
        type=str,
        required=True,
        help="Path to generated teacher completions JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save filtered student records",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=4,
        help="Maximum candidate completions evaluated per prompt",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    prompts_path = Path(args.prompts)
    if not prompts_path.is_file() and (PROJECT_ROOT / prompts_path).is_file():
        prompts_path = PROJECT_ROOT / prompts_path

    completions_path = Path(args.completions)
    if not completions_path.is_file() and (PROJECT_ROOT / completions_path).is_file():
        completions_path = PROJECT_ROOT / completions_path

    prompts: list[dict[str, Any]] = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    completions_by_id: dict[str, list[str]] = defaultdict(list)
    with open(completions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            p_id = entry.get("id") or entry.get("prompt_id")
            completion_text = entry.get("completion") or entry.get("text") or ""
            if p_id and completion_text:
                completions_by_id[p_id].append(completion_text)

    nt_recs, th_recs, stats = filter_teacher_completions(
        prompts=prompts,
        completions_by_id=completions_by_id,
        max_candidates_per_prompt=args.max_candidates,
    )

    out_dir = Path(args.output_dir)
    save_filtered_datasets(nt_recs, out_dir / "teacher_verified_non_thinking.jsonl")
    save_filtered_datasets(th_recs, out_dir / "teacher_verified_thinking.jsonl")
    print(f"Filtering complete. Accepted {len(nt_recs)} examples ({stats['acceptance_rate']:.1%})")


if __name__ == "__main__":
    main()
