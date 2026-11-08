"""Unit tests for Stage T3 SFT v2 evaluation runner.

This module tests devset item evaluation, metric aggregation, and report generation.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from training.evaluate_sft_v2 import (
    GATE_MAX_HARD_FAIL_RATE,
    GATE_MIN_IN_BAND_LENGTH,
    GATE_MIN_KEY_POINT,
    StageT3PassResult,
    compute_sft_v2_summary,
    evaluate_sft_v2_item,
    generate_sft_v2_report,
)


def test_evaluate_sft_v2_item_passing() -> None:
    """Test evaluation of a valid coaching explanation."""
    dev_item = {
        "id": "dev_001",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "verdict": "Good",
        "move_uci": "f1c4",
        "facts": ["Move attacks: pawn on f7.", "Phase: Opening."],
        "user_message": "Move played: Bc4. Verdict: Good. Facts:\nMove attacks: pawn on f7.\nPhase: Opening.",
    }
    completion = (
        "Playing Bc4 is a good developing move in this open position. "
        "It activates your light-squared bishop, controls key central squares, and directly attacks the vulnerable pawn on f7. "
        "This develops your pieces smoothly and prepares kingside castling."
    )

    res = evaluate_sft_v2_item(dev_item, completion)
    assert res["id"] == "dev_001"
    assert res["in_length_band"] is True
    assert res["passed_all_hard"] is True
    assert res["reward"] > 0.80


def test_compute_sft_v2_summary_passing_gates() -> None:
    """Test summary calculation when all Stage T3 gates pass."""
    results = [
        {
            "detail": "short",
            "in_length_band": True,
            "passed_all_hard": True,
            "reward": 0.88,
            "soft_scores": {"key_point": 0.85},
        }
        for _ in range(50)
    ]
    summary, pass_res = compute_sft_v2_summary(results)

    assert pass_res.hard_fail_rate == 0.0
    assert pass_res.passed_hard_fail is True
    assert pass_res.overall_length_in_band_rate == 100.0
    assert pass_res.passed_length is True
    assert pass_res.mean_key_point == 0.85
    assert pass_res.passed_key_point is True
    assert pass_res.passed_all_gates is True


def test_compute_sft_v2_summary_failing_hard_gate() -> None:
    """Test summary calculation when hard fail gate fails."""
    results = [
        {
            "detail": "short",
            "in_length_band": True,
            "passed_all_hard": (i >= 10),  # 10 fails out of 100 = 10% > 6%
            "reward": 0.50 if i < 10 else 0.85,
            "soft_scores": {"key_point": 0.75},
        }
        for i in range(100)
    ]
    summary, pass_res = compute_sft_v2_summary(results)

    assert pass_res.hard_fail_rate == 10.0
    assert pass_res.passed_hard_fail is False
    assert pass_res.passed_all_gates is False


def test_generate_sft_v2_report() -> None:
    """Test generating a Markdown report."""
    pass_res = StageT3PassResult(
        hard_fail_rate=4.0,
        passed_hard_fail=True,
        overall_length_in_band_rate=82.0,
        short_length_in_band_rate=85.0,
        long_length_in_band_rate=78.0,
        passed_length=True,
        mean_key_point=0.72,
        passed_key_point=True,
        mean_reward=0.89,
        passed_all_gates=True,
    )
    summary = {"total_evaluated": 150}

    report = generate_sft_v2_report(
        summary=summary,
        pass_result=pass_res,
        experiment_id="SFT-A1",
    )
    assert "# Stage T3 SFT v2 Evaluation Report: SFT-A1" in report
    assert "Gate Status**: **PASSED**" in report
    assert "Hard Fail Rate" in report
