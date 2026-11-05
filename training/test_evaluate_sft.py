"""Unit tests for Phase 5 evaluation module.

This module tests Phase 5 pass rule verification and sample selection.
It tests Markdown report generation, end-to-end evaluation, and zip bundle extraction.
All tests use ASD-STE100 Simplified Technical English docstrings.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

import pytest

from eval.metrics import AggregateMetrics, EvalMetrics, ItemMetric
from training.evaluate_sft import (
    MANUAL_REVIEW_NOTES,
    Phase5EvalContext,
    Phase5PassResult,
    check_phase5_pass_rule,
    create_structured_review_note,
    generate_phase5_report,
    main,
    run_phase5_evaluation,
    select_30_outputs_for_review,
)


def _make_dummy_aggregate(
    total: int = 100,
    hard_fail_rate: float = 20.0,
    format_fail_rate: float = 2.0,
    mean_reward: float = 0.5,
) -> AggregateMetrics:
    """Create a dummy AggregateMetrics object for testing."""
    return AggregateMetrics(
        total_items=total,
        hard_fail_rate=hard_fail_rate,
        hard_check_fail_rates={
            "legal_moves": 5.0,
            "grounded_moves": 5.0,
            "piece_on_square": 2.0,
            "verdict_tone": 1.0,
            "format": format_fail_rate,
            "beginner_jargon": 3.0,
        },
        mean_reward=mean_reward,
        mean_soft_score_all=0.75,
        mean_soft_score_passing=0.85,
        soft_score_breakdown={
            "key_point": 0.8,
            "better_move": 0.7,
            "length": 0.9,
            "level_fit": 0.85,
            "not_copied": 0.9,
            "no_repeats": 0.95,
        },
        avg_word_count_overall=45.0,
        avg_word_count_short=38.0,
        avg_word_count_long=110.0,
        pct_in_band_overall=90.0,
        pct_in_band_short=92.0,
        pct_in_band_long=85.0,
        verdict_items_count=80,
        verdict_coverage_rate=95.0,
        verdict_accuracy_all=85.0,
        verdict_accuracy_covered=89.5,
        verdict_tone_pass_rate=99.0,
        breakdowns={
            "task": {
                "explain_move": {"count": 80.0, "hard_fail_rate": hard_fail_rate, "mean_reward": mean_reward, "avg_word_count": 45.0},
                "explain_position": {"count": 20.0, "hard_fail_rate": hard_fail_rate, "mean_reward": mean_reward, "avg_word_count": 45.0},
            }
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
            "key_point": 0.8,
            "better_move": 0.8,
            "length": 0.9,
            "level_fit": 0.8,
            "not_copied": 0.9,
            "no_repeats": 1.0,
        },
        reward=0.75 if passed else -1.0,
        verdict="Good" if task == "explain_move" else None,
        verdict_accurate=True if task == "explain_move" else None,
        verdict_covered=task == "explain_move",
        verdict_tone_passed=True,
        completion="White plays Ng5 to attack f7 directly.",
    )


def test_check_phase5_pass_rule_passes_when_all_conditions_met() -> None:
    """Verify that pass rule succeeds when format pass > 95% and hard-fail is lower."""
    test_metrics = EvalMetrics(
        aggregate=_make_dummy_aggregate(hard_fail_rate=15.0, format_fail_rate=2.0),
        items=[],
    )
    base_metrics = EvalMetrics(
        aggregate=_make_dummy_aggregate(hard_fail_rate=35.0, format_fail_rate=50.0),
        items=[],
    )
    result = check_phase5_pass_rule(test_metrics, base_metrics)

    assert result.passed_format is True
    assert result.format_pass_rate == 98.0
    assert result.passed_hard_fail is True
    assert result.passed_all is True


def test_check_phase5_pass_rule_fails_when_format_under_95() -> None:
    """Verify that pass rule fails when format pass rate is 95% or lower."""
    test_metrics = EvalMetrics(
        aggregate=_make_dummy_aggregate(hard_fail_rate=10.0, format_fail_rate=5.5),
        items=[],
    )
    base_metrics = EvalMetrics(
        aggregate=_make_dummy_aggregate(hard_fail_rate=35.0, format_fail_rate=50.0),
        items=[],
    )
    result = check_phase5_pass_rule(test_metrics, base_metrics)

    assert result.format_pass_rate == 94.5
    assert result.passed_format is False
    assert result.passed_hard_fail is True
    assert result.passed_all is False


def test_check_phase5_pass_rule_fails_when_hard_fail_not_lower() -> None:
    """Verify that pass rule fails when SFT hard-fail rate is higher than base."""
    test_metrics = EvalMetrics(
        aggregate=_make_dummy_aggregate(hard_fail_rate=45.0, format_fail_rate=2.0),
        items=[],
    )
    base_metrics = EvalMetrics(
        aggregate=_make_dummy_aggregate(hard_fail_rate=40.0, format_fail_rate=50.0),
        items=[],
    )
    result = check_phase5_pass_rule(test_metrics, base_metrics)

    assert result.passed_format is True
    assert result.passed_hard_fail is False
    assert result.passed_all is False


def test_create_structured_review_note_clean_item() -> None:
    """Verify structured note for an item passing all checks."""
    item = _make_dummy_item("item_1", level="beginner", detail="short", passed=True)
    note = create_structured_review_note(item)

    assert "Format: PASS." in note
    assert "Voice: PASS." in note
    assert "Length: PASS." in note
    assert "Grounding: PASS." in note


def test_create_structured_review_note_failing_item() -> None:
    """Verify structured note for an item failing format and legal moves."""
    item = ItemMetric(
        id="item_fail",
        task="explain_move",
        level="beginner",
        detail="short",
        word_count=12,
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
        completion="### Header\nIllegal eval +2.5 move.",
    )
    note = create_structured_review_note(item)

    assert "Format: FAIL." in note
    assert "Voice: FAIL." in note
    assert "Length: FAIL." in note
    assert "Grounding: FAIL." in note
    assert "illegal move" in note


def test_select_30_outputs_for_review_returns_exact_30() -> None:
    """Verify that select_30_outputs_for_review returns exactly 30 diverse items."""
    tasks = ["explain_move", "explain_position"]
    levels = ["beginner", "club"]
    details = ["short", "long"]

    items: list[ItemMetric] = []
    records: list[dict[str, Any]] = []

    count = 0
    # Generate 48 items covering all strata with passing and failing examples
    for task in tasks:
        for level in levels:
            for detail in details:
                for is_pass in [True, False]:
                    for rep in range(3):
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
    selected = select_30_outputs_for_review(items, records)

    assert len(selected) == 30

    # Verify required keys in each selected dictionary
    required_keys = {
        "id", "task", "level", "detail", "verdict", "move_san",
        "word_count", "reward", "passed_all_hard", "failed_checks",
        "soft_scores", "completion", "review_note",
    }
    for item_dict in selected:
        assert required_keys.issubset(item_dict.keys())
        assert isinstance(item_dict["review_note"], str)
        assert len(item_dict["review_note"]) > 0

    # Verify representation of both tasks, levels, and details
    selected_tasks = {x["task"] for x in selected}
    selected_levels = {x["level"] for x in selected}
    selected_details = {x["detail"] for x in selected}
    selected_passes = {x["passed_all_hard"] for x in selected}

    assert selected_tasks == {"explain_move", "explain_position"}
    assert selected_levels == {"beginner", "club"}
    assert selected_details == {"short", "long"}
    assert True in selected_passes
    assert False in selected_passes


def test_select_30_outputs_for_review_small_list() -> None:
    """Verify that fewer than 30 items returns all available items."""
    items = [_make_dummy_item(f"item_{i}") for i in range(12)]
    records = [{"id": f"item_{i}", "move_san": "e4"} for i in range(12)]

    selected = select_30_outputs_for_review(items, records)
    assert len(selected) == 12


def test_generate_phase5_report() -> None:
    """Verify markdown content generated for Phase 5 report."""
    val_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=137), items=[])
    test_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=300), items=[])
    base_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=300, hard_fail_rate=100.0), items=[])

    pass_result = check_phase5_pass_rule(test_metrics, base_metrics)
    reviewed_30 = [
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
        }
    ]

    report = generate_phase5_report(
        model_name="sft-1.0-2b",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        base_qwen_metrics=base_metrics,
        pass_result=pass_result,
        reviewed_30=reviewed_30,
        training_metadata={"epochs": 2, "train_loss": 0.42},
    )

    assert "# Phase 5 Evaluation Report: sft-1.0-2b" in report
    assert "## 1. Executive Summary and Pass Rule Verification" in report
    assert "## 2. Training Configuration Summary" in report
    assert "## 3. Validation Split Automatic Metrics" in report
    assert "## 4. Frozen Test Set Comparison" in report
    assert "## 5. Manual Review of 30 Outputs" in report
    assert "sample_001" in report
    assert "Format Pass Rate" in report


def test_generate_phase5_report_with_context() -> None:
    """Verify report generation when using Phase5EvalContext dataclass."""
    val_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=137), items=[])
    test_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=300), items=[])
    base_metrics = EvalMetrics(aggregate=_make_dummy_aggregate(total=300, hard_fail_rate=100.0), items=[])
    pass_result = check_phase5_pass_rule(test_metrics, base_metrics)

    context = Phase5EvalContext(
        model_name="sft-ctx-test",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        base_qwen_metrics=base_metrics,
        pass_result=pass_result,
        reviewed_30=[],
        training_metadata={"epochs": 2},
    )

    report = generate_phase5_report(context)
    assert "# Phase 5 Evaluation Report: sft-ctx-test" in report
    assert "## 1. Executive Summary and Pass Rule Verification" in report


def test_create_structured_review_note_manual_observation() -> None:
    """Verify that structured review notes include specific human observations."""
    item = _make_dummy_item("test_008", task="explain_move", level="beginner", detail="short")
    note = create_structured_review_note(item)

    assert "Observation:" in note
    assert "reads the move number 15 and alternative move Ra7" in note


def test_run_phase5_evaluation_raises_when_baseline_missing(tmp_path: Path) -> None:
    """Verify that run_phase5_evaluation raises FileNotFoundError when baseline is missing."""
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"
    missing_base_file = tmp_path / "nonexistent_baseline.jsonl"

    record = {"id": "dummy_1", "task": "explain_move", "prompt": "FEN: 8/8/8/8/8/8/8/8 w - - 0 1", "completion": "Move."}
    val_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    test_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Base Qwen baseline file not found"):
        run_phase5_evaluation(
            val_outputs_path=val_file,
            test_outputs_path=test_file,
            base_qwen_path=missing_base_file,
            results_dir=tmp_path / "res",
            reports_dir=tmp_path / "rep",
        )


def test_run_phase5_evaluation_end_to_end(tmp_path: Path) -> None:
    """Verify full end-to-end evaluation runner execution."""
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
        "id": "test_e2e_1",
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
    base_file = tmp_path / "base_qwen.jsonl"

    val_file.write_text(json.dumps(clean_item) + "\n", encoding="utf-8")
    test_file.write_text(json.dumps(clean_item) + "\n", encoding="utf-8")
    base_file.write_text(json.dumps(clean_item) + "\n", encoding="utf-8")

    res_dir = tmp_path / "results"
    rep_dir = tmp_path / "reports"

    summary = run_phase5_evaluation(
        val_outputs_path=val_file,
        test_outputs_path=test_file,
        base_qwen_path=base_file,
        results_dir=res_dir,
        reports_dir=rep_dir,
        model_name="sft-test",
    )

    assert summary["model_name"] == "sft-test"
    assert (res_dir / "sft-test_val_eval.json").is_file()
    assert (res_dir / "sft-test_eval.json").is_file()
    assert (rep_dir / "sft-test.md").is_file()
    assert summary["reviewed_30_count"] == 1


def test_main_cli_with_from_zip(tmp_path: Path) -> None:
    """Verify that CLI handles --from-zip extraction and execution."""
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
        "completion": (
            "The move Ng5 is a good move. "
            "White attacks the vulnerable f7 square. "
            "Black must defend against the knight and bishop threat. "
            "This maintains active piece play in the opening."
        ),
    }

    # Create zip file
    zip_source_dir = tmp_path / "zip_source"
    zip_source_dir.mkdir()
    (zip_source_dir / "sft-1.0-2b_val_outputs.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (zip_source_dir / "sft-1.0-2b_outputs.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (zip_source_dir / "sft_training_summary.json").write_text(json.dumps({"total_steps": 400}), encoding="utf-8")

    zip_file = tmp_path / "phase5_sft_results.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.write(zip_source_dir / "sft-1.0-2b_val_outputs.jsonl", "sft-1.0-2b_val_outputs.jsonl")
        zf.write(zip_source_dir / "sft-1.0-2b_outputs.jsonl", "sft-1.0-2b_outputs.jsonl")
        zf.write(zip_source_dir / "sft_training_summary.json", "sft_training_summary.json")

    res_dir = tmp_path / "extracted_results"
    rep_dir = tmp_path / "extracted_reports"
    base_file = tmp_path / "base_qwen.jsonl"
    base_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    cli_args = [
        "evaluate_sft.py",
        "--from-zip", str(zip_file),
        "--base-qwen", str(base_file),
        "--results-dir", str(res_dir),
        "--reports-dir", str(rep_dir),
        "--model-name", "sft-zip-test",
    ]

    with patch.object(sys, "argv", cli_args):
        main()

    assert (res_dir / "sft-zip-test_val_eval.json").is_file()
    assert (res_dir / "sft-zip-test_eval.json").is_file()
    assert (rep_dir / "sft-zip-test.md").is_file()
