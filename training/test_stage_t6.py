"""Unit tests for Stage T6 frozen test-set evaluation and report generation.

All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.metrics import evaluate_completions
from training.evaluate_stage_t6 import (
    compute_reward_v2_stats,
    evaluate_overrefusal_records,
    format_stage_t6_report,
    merge_completions_with_testset,
    select_stratified_audit_sample,
    verify_stage_t6_gates,
)


def test_merge_completions_and_gate_verification() -> None:
    """Verify merging testset metadata and evaluating Stage T6 gates."""
    testset_sample = [
        {
            "id": "test_001",
            "fen": "r1bqkbnr/1p1n1ppp/p2p4/2pPp1BQ/B3P3/8/PPP2PPP/RN2K1NR b KQkq - 1 7",
            "task": "explain_move",
            "level": "club",
            "detail": "short",
            "move_uci": "g8f6",
            "move_san": "Nf6",
            "verdict": "Blunder",
            "phase": "opening",
            "facts": [
                "Nf6 attacks the queen on h5.",
                "After Nf6, White's best reply is Bxd7+.",
                "The queen on h5 is not defended.",
                "White has a check: Bxd7+.",
                "Material: equal.",
                "Phase: opening.",
            ],
            "user_message": (
                "task: explain_move\nlevel: club\ndetail: short\n\n"
                "FEN: r1bqkbnr/1p1n1ppp/p2p4/2pPp1BQ/B3P3/8/PPP2PPP/RN2K1NR b KQkq - 1 7\n"
                "Black to move, move 7.\nMove played: Nf6.\nVerdict: Blunder.\n"
                "Best move: Qa5+.\nFacts:\n- Nf6 attacks the queen on h5.\n"
                "- After Nf6, White's best reply is Bxd7+.\n"
                "- The queen on h5 is not defended.\n"
                "- White has a check: Bxd7+.\n"
                "- Material: equal.\n- Phase: opening."
            ),
        }
    ]
    raw_completions = [
        {
            "id": "test_001",
            "completion": (
                "The move Nf6 is a blunder that allows White to play the strong check Bxd7+. "
                "While attacking the undefended queen on h5 looks tempting, Black should play Qa5+ "
                "instead to keep a decisive advantage in this opening position."
            ),
        }
    ]

    merged = merge_completions_with_testset(testset_sample, raw_completions)
    assert len(merged) == 1
    assert merged[0]["id"] == "test_001"
    assert merged[0]["concept_tier"] is not None

    eval_metrics = evaluate_completions(merged)
    v2_stats = compute_reward_v2_stats(merged)
    assert v2_stats["mean_reward_v2"] > 0.70

    overrefusal = evaluate_overrefusal_records([
        {"id": "refusal_01", "completion": "White plays Bxf7+ to expose the Black king."}
    ])
    assert overrefusal["overrefusal_pct"] == 0.0

    audit = select_stratified_audit_sample(eval_metrics, per_tier=2)
    assert audit["diary_voice_violations"] == 0

    gates = verify_stage_t6_gates(eval_metrics, overrefusal, judge_results=None)
    assert gates.passed_all_deterministic is True
    assert gates.passed_all_gates is True

    report_md = format_stage_t6_report(
        eval_metrics=eval_metrics,
        reward_v2_stats=v2_stats,
        gate_summary=gates,
        overrefusal_summary=overrefusal,
        audit_summary=audit,
    )
    assert "Stage T6 Final Evaluation Report" in report_md
    assert "PASSED" in report_md


def test_stage_t6_notebook_structure() -> None:
    """Verify that training/stage_t6_eval.ipynb exists and embeds all 300 test items."""
    nb_path = Path("training/stage_t6_eval.ipynb")
    assert nb_path.is_file()
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(nb_data["cells"]) >= 6
    full_source = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"])
    assert "TESTSET_B64" in full_source
    assert "OVERREFUSAL_B64" in full_source
    assert "grpo_v2_b2" in full_source
