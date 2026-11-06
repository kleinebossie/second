"""Unit tests for Phase 6 GRPO evaluation module.

This module tests checkpoint ranking, Phase 6 pass rule verification, and sample review.
It tests anti-hacking guards, Markdown report generation, judge evaluation, and CLI execution.
All tests use ASD-STE100 Simplified Technical English docstrings.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import zipfile

import pytest

from eval.metrics import AggregateMetrics, EvalMetrics, ItemMetric
from training.evaluate_grpo import (
    CheckpointRankingEntry,
    Phase6EvalContext,
    Phase6PassResult,
    audit_anti_hacking,
    check_phase6_pass_rule,
    create_grpo_review_note,
    extract_step_number,
    generate_phase6_report,
    main,
    rank_checkpoints,
    run_phase6_evaluation,
    select_20_outputs_for_review,
)


def _make_dummy_aggregate(
    total: int = 100,
    hard_fail_rate: float = 10.0,
    format_fail_rate: float = 1.0,
    mean_reward: float = 0.65,
    avg_word_count: float = 45.0,
) -> AggregateMetrics:
    """Create a dummy AggregateMetrics object for testing."""
    return AggregateMetrics(
        total_items=total,
        hard_fail_rate=hard_fail_rate,
        hard_check_fail_rates={
            "legal_moves": 2.0,
            "grounded_moves": 3.0,
            "piece_on_square": 1.0,
            "verdict_tone": 1.0,
            "format": format_fail_rate,
            "beginner_jargon": 2.0,
        },
        mean_reward=mean_reward,
        mean_soft_score_all=0.82,
        mean_soft_score_passing=0.88,
        soft_score_breakdown={
            "key_point": 0.85,
            "better_move": 0.80,
            "length": 0.92,
            "level_fit": 0.88,
            "not_copied": 0.91,
            "no_repeats": 0.96,
        },
        avg_word_count_overall=avg_word_count,
        avg_word_count_short=38.0,
        avg_word_count_long=115.0,
        pct_in_band_overall=92.0,
        pct_in_band_short=94.0,
        pct_in_band_long=88.0,
        verdict_items_count=80,
        verdict_coverage_rate=96.0,
        verdict_accuracy_all=88.0,
        verdict_accuracy_covered=91.5,
        verdict_tone_pass_rate=99.0,
        breakdowns={
            "task": {
                "explain_move": {"count": 80.0, "hard_fail_rate": hard_fail_rate, "mean_reward": mean_reward, "avg_word_count": avg_word_count},
                "explain_position": {"count": 20.0, "hard_fail_rate": hard_fail_rate, "mean_reward": mean_reward, "avg_word_count": avg_word_count},
            },
            "level": {
                "beginner": {"count": 50.0, "hard_fail_rate": hard_fail_rate, "mean_reward": mean_reward, "avg_word_count": avg_word_count},
                "club": {"count": 50.0, "hard_fail_rate": hard_fail_rate, "mean_reward": mean_reward, "avg_word_count": avg_word_count},
            },
            "detail": {
                "short": {"count": 70.0, "hard_fail_rate": hard_fail_rate, "mean_reward": mean_reward, "avg_word_count": avg_word_count},
                "long": {"count": 30.0, "hard_fail_rate": hard_fail_rate, "mean_reward": mean_reward, "avg_word_count": avg_word_count},
            },
        },
    )


def _make_dummy_item(
    item_id: str,
    task: str = "explain_move",
    level: str = "club",
    detail: str = "short",
    passed: bool = True,
    failed_check: str | None = None,
    word_count: int = 40,
    completion: str = "White plays Ng5 to attack f7 directly.",
) -> ItemMetric:
    """Create a dummy ItemMetric object for testing."""
    failed = [failed_check] if (failed_check and not passed) else ([] if passed else ["format"])
    hard_checks = {
        "legal_moves": "legal_moves" not in failed,
        "grounded_moves": "grounded_moves" not in failed,
        "piece_on_square": "piece_on_square" not in failed,
        "verdict_tone": "verdict_tone" not in failed,
        "format": "format" not in failed,
        "beginner_jargon": "beginner_jargon" not in failed,
    }
    return ItemMetric(
        id=item_id,
        task=task,
        level=level,
        detail=detail,
        word_count=word_count,
        in_length_band=True,
        passed_all_hard=passed,
        failed_checks=failed,
        hard_checks=hard_checks,
        soft_scores={
            "key_point": 0.85,
            "better_move": 0.82,
            "length": 0.90,
            "level_fit": 0.85,
            "not_copied": 0.92,
            "no_repeats": 1.0,
        },
        reward=0.82 if passed else -1.0,
        verdict="Good" if task == "explain_move" else None,
        verdict_accurate=True if task == "explain_move" else None,
        verdict_covered=task == "explain_move",
        verdict_tone_passed=True,
        completion=completion,
    )


def test_extract_step_number() -> None:
    """Verify that extract_step_number parses integers from checkpoint names."""
    assert extract_step_number("checkpoint-200") == 200
    assert extract_step_number("checkpoint_50") == 50
    assert extract_step_number("step-1234") == 1234
    assert extract_step_number("checkpoint-final") == 0
    assert extract_step_number("model") == 0


def test_rank_checkpoints_sorting_and_selection() -> None:
    """Verify checkpoint ranking by reward, hard-fail rate, and step."""
    m1 = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.60, hard_fail_rate=15.0), items=[])
    m2 = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.75, hard_fail_rate=10.0), items=[])
    m3 = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.75, hard_fail_rate=8.0), items=[])
    m4 = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.50, hard_fail_rate=20.0), items=[])

    chk_val = {
        "checkpoint-100": m1,
        "checkpoint-200": m2,
        "checkpoint-300": m3,
        "checkpoint-50": m4,
    }

    rankings, top_names = rank_checkpoints(chk_val, top_k=3)

    assert len(rankings) == 4
    assert len(top_names) == 3

    # checkpoint-300 has same reward as 200 (0.75), but lower hard-fail rate (8.0 vs 10.0)
    assert rankings[0].checkpoint_name == "checkpoint-300"
    assert rankings[0].is_best is True
    assert rankings[0].step == 300

    assert rankings[1].checkpoint_name == "checkpoint-200"
    assert rankings[1].is_best is False

    assert rankings[2].checkpoint_name == "checkpoint-100"
    assert rankings[3].checkpoint_name == "checkpoint-50"

    assert top_names == ["checkpoint-300", "checkpoint-200", "checkpoint-100"]


def test_rank_checkpoints_step_tie_breaker() -> None:
    """Verify that higher step wins when reward and hard-fail are identical."""
    m1 = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.70, hard_fail_rate=10.0), items=[])
    m2 = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.70, hard_fail_rate=10.0), items=[])

    chk_val = {
        "checkpoint-100": m1,
        "checkpoint-200": m2,
    }

    rankings, top_names = rank_checkpoints(chk_val, top_k=2)

    assert rankings[0].checkpoint_name == "checkpoint-200"
    assert rankings[0].is_best is True
    assert rankings[1].checkpoint_name == "checkpoint-100"
    assert rankings[1].is_best is False


def test_rank_checkpoints_empty_and_test_map() -> None:
    """Verify ranking with empty dictionary and with test metrics mapping."""
    empty_rankings, empty_names = rank_checkpoints({})
    assert empty_rankings == []
    assert empty_names == []

    v_metric = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.68, hard_fail_rate=8.0), items=[])
    t_metric = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.65, hard_fail_rate=9.0), items=[])

    rankings, names = rank_checkpoints(
        checkpoint_metrics={"checkpoint-150": v_metric},
        test_metrics_map={"checkpoint-150": t_metric},
        top_k=1,
    )
    assert len(rankings) == 1
    assert rankings[0].test_mean_reward == 0.65
    assert rankings[0].test_hard_fail_rate == 9.0
    assert names == ["checkpoint-150"]


def test_check_phase6_pass_rule_all_passing() -> None:
    """Verify that pass rule succeeds when all 4 conditions are satisfied."""
    grpo_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.72, hard_fail_rate=8.0, format_fail_rate=1.0), items=[])
    sft_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.65, hard_fail_rate=11.3, format_fail_rate=0.0), items=[])
    base_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.10, hard_fail_rate=100.0, format_fail_rate=100.0), items=[])

    res = check_phase6_pass_rule(grpo_metrics, sft_metrics, base_metrics)

    assert res.reward_improved_over_sft is True
    assert res.hard_fail_lower_than_base is True
    assert res.hard_fail_not_worse_than_sft is True
    assert res.passed_format is True
    assert res.passed_all is True


def test_check_phase6_pass_rule_fails_when_reward_not_improved() -> None:
    """Verify that pass rule fails when GRPO reward is lower than SFT reward."""
    grpo_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.60, hard_fail_rate=8.0, format_fail_rate=1.0), items=[])
    sft_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.65, hard_fail_rate=11.3, format_fail_rate=0.0), items=[])
    base_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.10, hard_fail_rate=100.0, format_fail_rate=100.0), items=[])

    res = check_phase6_pass_rule(grpo_metrics, sft_metrics, base_metrics)

    assert res.reward_improved_over_sft is False
    assert res.passed_all is False


def test_check_phase6_pass_rule_fails_when_hard_fail_higher_than_sft() -> None:
    """Verify that pass rule fails when GRPO hard-fail rate exceeds SFT rate."""
    grpo_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.75, hard_fail_rate=15.0, format_fail_rate=1.0), items=[])
    sft_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.65, hard_fail_rate=11.3, format_fail_rate=0.0), items=[])
    base_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.10, hard_fail_rate=100.0, format_fail_rate=100.0), items=[])

    res = check_phase6_pass_rule(grpo_metrics, sft_metrics, base_metrics)

    assert res.hard_fail_not_worse_than_sft is False
    assert res.passed_all is False


def test_check_phase6_pass_rule_fails_when_format_under_95() -> None:
    """Verify that pass rule fails when format pass rate is 95 percent or lower."""
    grpo_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.75, hard_fail_rate=8.0, format_fail_rate=6.0), items=[])
    sft_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.65, hard_fail_rate=11.3, format_fail_rate=0.0), items=[])
    base_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.10, hard_fail_rate=100.0, format_fail_rate=100.0), items=[])

    res = check_phase6_pass_rule(grpo_metrics, sft_metrics, base_metrics)

    assert res.grpo_format_pass_rate == 94.0
    assert res.passed_format is False
    assert res.passed_all is False


def test_create_grpo_review_note_clean_and_defects() -> None:
    """Verify structured review note contents for passing and failing items."""
    clean_item = _make_dummy_item("item_1", level="beginner", detail="short", word_count=35)
    note_clean = create_grpo_review_note(clean_item)

    assert "Format: PASS." in note_clean
    assert "Voice: PASS." in note_clean
    assert "Length: PASS." in note_clean
    assert "Grounding: PASS." in note_clean

    failing_item = ItemMetric(
        id="item_fail",
        task="explain_move",
        level="beginner",
        detail="short",
        word_count=15,
        in_length_band=False,
        passed_all_hard=False,
        failed_checks=["format", "legal_moves", "beginner_jargon"],
        hard_checks={
            "legal_moves": False,
            "grounded_moves": True,
            "piece_on_square": True,
            "verdict_tone": True,
            "format": False,
            "beginner_jargon": False,
        },
        soft_scores={},
        reward=-1.0,
        verdict="Good",
        verdict_accurate=False,
        verdict_covered=True,
        verdict_tone_passed=True,
        completion="In this position white plays +2.5 eval move.",
    )
    note_fail = create_grpo_review_note(failing_item)

    assert "Format: FAIL." in note_fail
    assert "Voice: FAIL." in note_fail
    assert "Length: FAIL." in note_fail
    assert "Grounding: FAIL." in note_fail
    assert "illegal move" in note_fail
    assert "Watch for length collapse" in note_fail
    assert "Template opening detected" in note_fail


def test_select_20_outputs_for_review_exact_count_and_strata() -> None:
    """Verify that select_20_outputs_for_review returns exactly 20 diverse items."""
    tasks = ["explain_move", "explain_position"]
    levels = ["beginner", "club"]
    details = ["short", "long"]

    items: list[ItemMetric] = []
    records: list[dict[str, Any]] = []

    count = 0
    for task in tasks:
        for level in levels:
            for detail in details:
                for is_pass in [True, False]:
                    for _ in range(3):
                        count += 1
                        i_id = f"test_{count:03d}"
                        fail_reason = "legal_moves" if not is_pass else None
                        items.append(
                            _make_dummy_item(
                                item_id=i_id,
                                task=task,
                                level=level,
                                detail=detail,
                                passed=is_pass,
                                failed_check=fail_reason,
                            )
                        )
                        records.append({
                            "id": i_id,
                            "task": task,
                            "level": level,
                            "detail": detail,
                            "move_san": "Ng5",
                            "verdict": "Good",
                        })

    assert len(items) == 48
    selected = select_20_outputs_for_review(items, records)

    assert len(selected) == 20

    required_keys = {
        "id", "task", "level", "detail", "verdict", "move_san",
        "word_count", "reward", "passed_all_hard", "failed_checks",
        "soft_scores", "completion", "review_note",
        "length_collapsed", "robotic_opening",
    }
    for item_dict in selected:
        assert required_keys.issubset(item_dict.keys())
        assert isinstance(item_dict["review_note"], str)

    selected_tasks = {x["task"] for x in selected}
    selected_levels = {x["level"] for x in selected}
    selected_details = {x["detail"] for x in selected}
    selected_passes = {x["passed_all_hard"] for x in selected}

    assert selected_tasks == {"explain_move", "explain_position"}
    assert selected_levels == {"beginner", "club"}
    assert selected_details == {"short", "long"}
    assert True in selected_passes
    assert False in selected_passes


def test_select_20_outputs_for_review_small_list() -> None:
    """Verify that fewer than 20 items returns all available items."""
    items = [_make_dummy_item(f"item_{i}") for i in range(8)]
    records = [{"id": f"item_{i}", "move_san": "e4"} for i in range(8)]

    selected = select_20_outputs_for_review(items, records)
    assert len(selected) == 8


def test_audit_anti_hacking_flags() -> None:
    """Verify anti-hacking audit signals for collapse, robotic text, and judge concordance."""
    # Test collapse and robotic triggers
    items: list[dict[str, Any]] = [
        {"completion": "In this position White plays.", "length_collapsed": True, "robotic_opening": True},
        {"completion": "In this position Black attacks.", "length_collapsed": True, "robotic_opening": True},
        {"completion": "White plays Ng5 to win.", "length_collapsed": False, "robotic_opening": False},
        {"completion": "Black plays Be7 to castle.", "length_collapsed": False, "robotic_opening": False},
    ]

    audit = audit_anti_hacking(items)
    assert audit["length_collapse_count"] == 2
    assert audit["length_collapse_rate"] == 50.0
    assert audit["length_collapse_flag"] is True
    assert audit["robotic_opening_count"] == 2
    assert audit["robotic_opening_rate"] == 50.0
    assert audit["robotic_opening_flag"] is True

    # Test judge discordance
    mock_judge = {
        "pairwise": {
            "summaries": {
                "sft-1.0-2b": {
                    "candidate_win_rate": 30.0,
                    "candidate_win_tie_rate": 45.0,
                }
            }
        }
    }
    grpo_m = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.75), items=[])
    sft_m = EvalMetrics(aggregate=_make_dummy_aggregate(mean_reward=0.60), items=[])

    audit_discordant = audit_anti_hacking(
        reviewed_20=items,
        judge_results=mock_judge,
        sft_test_metrics=sft_m,
        grpo_test_metrics=grpo_m,
    )
    assert audit_discordant["judge_concordance_flag"] is False
    assert "Watch for reward hacking" in audit_discordant["judge_concordance_note"]


def test_generate_phase6_report_with_and_without_judge() -> None:
    """Verify markdown report generation with and without Gemini judge results."""
    val_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=137), items=[])
    test_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=300, mean_reward=0.72, hard_fail_rate=8.0), items=[])
    sft_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=300, mean_reward=0.65, hard_fail_rate=11.3), items=[])
    base_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=300, mean_reward=0.10, hard_fail_rate=100.0), items=[])

    pass_result = check_phase6_pass_rule(test_metrics, sft_metrics, base_metrics)

    reviewed_20 = [
        {
            "id": "sample_001",
            "task": "explain_move",
            "level": "beginner",
            "detail": "short",
            "verdict": "Good",
            "move_san": "Ng5",
            "word_count": 35,
            "reward": 0.85,
            "passed_all_hard": True,
            "failed_checks": [],
            "soft_scores": {"key_point": 1.0},
            "completion": "White plays Ng5 to attack the f7 pawn.",
            "review_note": "Format: PASS. Voice: PASS. Length: PASS. Grounding: PASS.",
            "length_collapsed": False,
            "robotic_opening": False,
        }
    ]

    rankings = [
        CheckpointRankingEntry(
            checkpoint_name="checkpoint-300",
            step=300,
            val_mean_reward=0.74,
            val_hard_fail_rate=7.5,
            val_avg_word_count=42.0,
            val_pct_in_band=94.0,
            is_best=True,
        )
    ]

    step_logs = [
        {"step": 50, "mean_reward": 0.55, "hard_fail_rate": 15.0, "avg_word_count": 40.0},
        {"step": 100, "mean_reward": 0.65, "hard_fail_rate": 10.0, "avg_word_count": 42.0},
    ]

    mock_judge = {
        "model": "gemini-3.5-flash-lite",
        "rubric": {
            "summary": {
                "mean_correctness": 4.5,
                "mean_clarity": 4.6,
                "mean_usefulness": 4.3,
                "mean_level_fit": 4.4,
                "mean_overall": 4.45,
            }
        },
        "pairwise": {
            "summaries": {
                "base-qwen-2b": {
                    "total_pairs": 300,
                    "candidate_wins": 260,
                    "opponent_wins": 20,
                    "ties": 20,
                    "candidate_win_rate": 86.7,
                    "candidate_win_tie_rate": 93.3,
                    "swap_agreement_rate": 90.0,
                },
                "sft-1.0-2b": {
                    "total_pairs": 300,
                    "candidate_wins": 180,
                    "opponent_wins": 60,
                    "ties": 60,
                    "candidate_win_rate": 60.0,
                    "candidate_win_tie_rate": 80.0,
                    "swap_agreement_rate": 85.0,
                },
            }
        },
    }

    ctx = Phase6EvalContext(
        model_name="grpo-1.0-2b",
        best_checkpoint="checkpoint-300",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        sft_test_metrics=sft_metrics,
        base_qwen_metrics=base_metrics,
        pass_result=pass_result,
        reviewed_20=reviewed_20,
        checkpoint_rankings=rankings,
        step_logs=step_logs,
        judge_results=mock_judge,
        training_metadata={"learning_rate": 5e-6},
    )

    report_with_judge = generate_phase6_report(ctx)

    assert "# Phase 6 Evaluation Report: grpo-1.0-2b" in report_with_judge
    assert "## 1. Executive Summary and Phase 6 Verification" in report_with_judge
    assert "## 2. GRPO Training Configuration and Step Progression" in report_with_judge
    assert "## 3. Validation Split Automatic Metrics" in report_with_judge
    assert "## 4. Frozen Test Set Comparison" in report_with_judge
    assert "## 5. Gemini Flash Judge Results on Best Checkpoint" in report_with_judge
    assert "## 6. Manual Review of 20 Outputs and Anti-Hacking Audit" in report_with_judge
    assert "## 7. Anti-Hacking Audit Findings" in report_with_judge
    assert "checkpoint-300" in report_with_judge
    assert "86.7%" in report_with_judge
    assert "4.45 / 5.0" in report_with_judge

    # Without judge results
    ctx_no_judge = Phase6EvalContext(
        model_name="grpo-no-judge",
        best_checkpoint="checkpoint-300",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        sft_test_metrics=sft_metrics,
        base_qwen_metrics=base_metrics,
        pass_result=pass_result,
        reviewed_20=reviewed_20,
        checkpoint_rankings=rankings,
        step_logs=[],
        judge_results=None,
    )
    report_no_judge = generate_phase6_report(ctx_no_judge)
    assert "Gemini Flash judge evaluation was skipped" in report_no_judge


def test_run_phase6_evaluation_missing_baselines(tmp_path: Path) -> None:
    """Verify that FileNotFoundError is raised when baselines are missing."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    sft_file = tmp_path / "sft.jsonl"
    base_file = tmp_path / "base.jsonl"

    record = {"id": "item_1", "task": "explain_move", "prompt": "FEN: 8/8/8/8/8/8/8/8 w - - 0 1", "completion": "Move."}
    val_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    test_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    # Missing SFT file
    with pytest.raises(FileNotFoundError, match="SFT baseline file not found"):
        run_phase6_evaluation(
            val_outputs_path=val_file,
            test_outputs_path=test_file,
            sft_outputs_path=sft_file,
            base_qwen_path=base_file,
            results_dir=tmp_path / "res",
            reports_dir=tmp_path / "rep",
        )

    # Missing Base Qwen file
    sft_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Base Qwen baseline file not found"):
        run_phase6_evaluation(
            val_outputs_path=val_file,
            test_outputs_path=test_file,
            sft_outputs_path=sft_file,
            base_qwen_path=base_file,
            results_dir=tmp_path / "res",
            reports_dir=tmp_path / "rep",
        )


def test_run_phase6_evaluation_end_to_end_skip_judge(tmp_path: Path) -> None:
    """Verify complete evaluation execution with skip_judge enabled."""
    prompt_str = (
        "task: explain_move\n"
        "level: club\n"
        "detail: short\n\n"
        "FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4\n"
        "White to move, move 4.\n"
        "Move played: Ng5.\n"
        "Verdict: Good.\n"
        "White's winning chances: 54% → 53%.\n"
        "Best move: d3.\n"
        "Best line: d3 Bc5 c3 d6.\n"
        "Facts:\n"
        "- Ng5 attacks the pawn on f7 together with the bishop on c4.\n"
        "- Black's best reply is d5.\n"
        "- Material: equal.\n"
    )

    clean_item = {
        "id": "grpo_test_1",
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "prompt": prompt_str,
        "completion": (
            "The move Ng5 is a good move. "
            "White attacks the vulnerable f7 square. "
            "Black must defend against the knight and bishop threat. "
            "This maintains active piece play in the opening."
        ),
    }

    val_file = tmp_path / "val_outputs.jsonl"
    test_file = tmp_path / "test_outputs.jsonl"
    sft_file = tmp_path / "sft_outputs.jsonl"
    base_file = tmp_path / "base_qwen.jsonl"

    for f in (val_file, test_file, sft_file, base_file):
        f.write_text(json.dumps(clean_item) + "\n", encoding="utf-8")

    res_dir = tmp_path / "results"
    rep_dir = tmp_path / "reports"

    summary = run_phase6_evaluation(
        val_outputs_path=val_file,
        test_outputs_path=test_file,
        sft_outputs_path=sft_file,
        base_qwen_path=base_file,
        results_dir=res_dir,
        reports_dir=rep_dir,
        model_name="grpo-test",
        skip_judge=True,
    )

    assert summary["model_name"] == "grpo-test"
    assert (res_dir / "grpo-test_val_eval.json").is_file()
    assert (res_dir / "grpo-test_eval.json").is_file()
    assert (rep_dir / "grpo-test.md").is_file()
    assert summary["reviewed_20_count"] == 1
    assert summary["judge_results"] is None


def test_run_phase6_evaluation_with_mocked_judge(tmp_path: Path) -> None:
    """Verify evaluation execution with a mocked Gemini judge."""
    prompt_str = (
        "task: explain_move\n"
        "level: club\n"
        "detail: short\n\n"
        "FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4\n"
        "White to move, move 4.\n"
        "Move played: Ng5.\n"
        "Verdict: Good.\n"
        "White's winning chances: 54% → 53%.\n"
        "Best move: d3.\n"
        "Best line: d3 Bc5 c3 d6.\n"
        "Facts:\n"
        "- Ng5 attacks the pawn on f7 together with the bishop on c4.\n"
        "- Black's best reply is d5.\n"
        "- Material: equal.\n"
    )

    item = {
        "id": "grpo_judge_1",
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "prompt": prompt_str,
        "completion": "The move Ng5 is a good move. White attacks f7.",
    }

    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    sft_file = tmp_path / "sft.jsonl"
    base_file = tmp_path / "base.jsonl"

    for f in (val_file, test_file, sft_file, base_file):
        f.write_text(json.dumps(item) + "\n", encoding="utf-8")

    res_dir = tmp_path / "results"
    rep_dir = tmp_path / "reports"

    mock_judge_results = {
        "model": "gemini-3.5-flash-lite",
        "rubric": {
            "summary": {
                "mean_correctness": 4.8,
                "mean_clarity": 4.7,
                "mean_usefulness": 4.6,
                "mean_level_fit": 4.5,
                "mean_overall": 4.65,
            }
        },
        "pairwise": {
            "summaries": {
                "base-qwen-2b": {
                    "total_pairs": 1,
                    "candidate_wins": 1,
                    "opponent_wins": 0,
                    "ties": 0,
                    "candidate_win_rate": 100.0,
                    "candidate_win_tie_rate": 100.0,
                    "swap_agreement_rate": 100.0,
                }
            }
        },
    }

    with patch("training.evaluate_grpo.get_gemini_api_key", return_value="fake-api-key"):
        with patch("training.evaluate_grpo.run_judge_eval", return_value=mock_judge_results) as mock_run:
            summary = run_phase6_evaluation(
                val_outputs_path=val_file,
                test_outputs_path=test_file,
                sft_outputs_path=sft_file,
                base_qwen_path=base_file,
                results_dir=res_dir,
                reports_dir=rep_dir,
                model_name="grpo-judge-test",
                run_judge=True,
                skip_judge=False,
            )

    assert mock_run.called
    assert summary["judge_results"] is not None
    assert (res_dir / "grpo-judge-test_judge.json").is_file()
    assert (rep_dir / "grpo-judge-test.md").is_file()


def test_main_cli_with_from_zip(tmp_path: Path) -> None:
    """Verify that CLI handles --from-zip extraction and evaluation."""
    prompt_str = (
        "task: explain_move\n"
        "level: club\n"
        "detail: short\n\n"
        "FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4\n"
        "White to move, move 4.\n"
        "Move played: Ng5.\n"
        "Verdict: Good.\n"
        "White's winning chances: 54% → 53%.\n"
        "Best move: d3.\n"
        "Best line: d3 Bc5 c3 d6.\n"
        "Facts:\n"
        "- Ng5 attacks the pawn on f7 together with the bishop on c4.\n"
        "- Black's best reply is d5.\n"
        "- Material: equal.\n"
    )

    record = {
        "id": "zip_item_1",
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "prompt": prompt_str,
        "completion": "The move Ng5 is a good move. White attacks f7.",
    }

    zip_src = tmp_path / "zip_src"
    zip_src.mkdir()
    (zip_src / "grpo-1.0-2b_val_outputs.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (zip_src / "grpo-1.0-2b_outputs.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (zip_src / "checkpoint-300_val.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (zip_src / "grpo_training_summary.json").write_text(
        json.dumps({
            "total_steps": 300,
            "step_logs": [
                {"step": 50, "mean_reward": 0.55, "hard_fail_rate": 15.0, "avg_word_count": 40.0},
                {"step": 100, "mean_reward": 0.65, "hard_fail_rate": 10.0, "avg_word_count": 42.0},
            ],
        }),
        encoding="utf-8",
    )

    zip_file = tmp_path / "phase6_grpo_results.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.write(zip_src / "grpo-1.0-2b_val_outputs.jsonl", "grpo-1.0-2b_val_outputs.jsonl")
        zf.write(zip_src / "grpo-1.0-2b_outputs.jsonl", "grpo-1.0-2b_outputs.jsonl")
        zf.write(zip_src / "checkpoint-300_val.jsonl", "checkpoint-300_val.jsonl")
        zf.write(zip_src / "grpo_training_summary.json", "grpo_training_summary.json")

    res_dir = tmp_path / "extracted_results"
    rep_dir = tmp_path / "extracted_reports"
    sft_file = tmp_path / "sft.jsonl"
    base_file = tmp_path / "base.jsonl"
    sft_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    base_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    cli_args = [
        "evaluate_grpo.py",
        "--from-zip", str(zip_file),
        "--sft-outputs", str(sft_file),
        "--base-qwen", str(base_file),
        "--results-dir", str(res_dir),
        "--reports-dir", str(rep_dir),
        "--model-name", "grpo-zip-test",
        "--skip-judge",
    ]

    with patch.object(sys, "argv", cli_args):
        main()

    assert (res_dir / "grpo-zip-test_val_eval.json").is_file()
    assert (res_dir / "grpo-zip-test_eval.json").is_file()
    assert (rep_dir / "grpo-zip-test.md").is_file()


def test_main_cli_missing_zip_exits(tmp_path: Path) -> None:
    """Verify that CLI exits with code 1 if zip file does not exist."""
    missing_zip = tmp_path / "nonexistent.zip"
    cli_args = [
        "evaluate_grpo.py",
        "--from-zip", str(missing_zip),
    ]

    with patch.object(sys, "argv", cli_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
