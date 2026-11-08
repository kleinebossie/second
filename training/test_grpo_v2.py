"""Unit tests for Stage T4 Reward v2, GRPO v2 telemetry, and evaluation.

This module tests:
- Graded hard penalties and substantiation floor in rewards/scorer.py
- Multiplicative target length scaling M_len(w)
- Multi-objective reward streams (MO-GRPO / GDPO) in rewards/api.py
- GRPO v2 configs, live collapse telemetry, and CMDP primal-dual updates
- Stage T4 devset evaluation and bundle packaging
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import zipfile

import pytest

from rewards.api import (
    reward_grounding_fn,
    reward_pedagogy_fn,
    reward_style_fn,
)
from rewards.checks import CheckResult
from rewards.scorer import (
    compute_graded_hard_penalty,
    compute_length_multiplier,
    compute_reward,
)
from training.build_colab_bundle import build_phase4_grpo_v2_bundle
from training.evaluate_grpo_v2 import (
    compute_grpo_v2_summary,
    evaluate_grpo_v2_item,
    generate_grpo_v2_report,
)
from training.grpo_data import (
    MonitoredGRPOReward,
    build_multi_objective_reward_funcs,
    detect_anomalies,
    load_grpo_config,
)

ITALIAN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def test_compute_length_multiplier_short() -> None:
    """Verify smooth target length multiplier M_len(w) for short detail mode."""
    assert compute_length_multiplier("", "short") == 0.2
    assert compute_length_multiplier("word " * 10, "short") == 0.2
    assert compute_length_multiplier("word " * 15, "short") == 0.2
    assert compute_length_multiplier("word " * 25, "short") == pytest.approx(0.6)
    assert compute_length_multiplier("word " * 35, "short") == 1.0
    assert compute_length_multiplier("word " * 60, "short") == 1.0
    assert compute_length_multiplier("word " * 75, "short") == 1.0
    assert compute_length_multiplier("word " * 110, "short") == 0.3
    assert compute_length_multiplier("word " * 130, "short") == 0.3


def test_compute_length_multiplier_long() -> None:
    """Verify smooth target length multiplier M_len(w) for long detail mode."""
    assert compute_length_multiplier("word " * 30, "long") == 0.2
    assert compute_length_multiplier("word " * 40, "long") == 0.2
    assert compute_length_multiplier("word " * 65, "long") == pytest.approx(0.6)
    assert compute_length_multiplier("word " * 90, "long") == 1.0
    assert compute_length_multiplier("word " * 140, "long") == 1.0
    assert compute_length_multiplier("word " * 180, "long") == 1.0
    assert compute_length_multiplier("word " * 240, "long") == 0.3
    assert compute_length_multiplier("word " * 260, "long") == 0.3


def test_compute_graded_hard_penalty() -> None:
    """Verify graded hard penalties for fatal hallucinations vs style faults."""
    ok = CheckResult(passed=True, reason="")
    fail = CheckResult(passed=False, reason="failed")

    # 1 style fault -> -0.35
    res_style1 = {"coach_voice": fail, "legal_moves": ok}
    assert compute_graded_hard_penalty(res_style1) == -0.35

    # 2 style faults -> -0.70
    res_style2 = {"coach_voice": fail, "format": fail, "legal_moves": ok}
    assert compute_graded_hard_penalty(res_style2) == -0.70

    # 1 fatal hallucination -> -0.50
    res_fatal1 = {"piece_on_square": fail, "coach_voice": ok}
    assert compute_graded_hard_penalty(res_fatal1) == -0.50

    # 1 fatal + 1 style -> -0.85
    res_mixed = {"piece_on_square": fail, "coach_voice": fail}
    assert compute_graded_hard_penalty(res_mixed) == -0.85

    # 2 fatal hallucinations -> capped at -1.00
    res_fatal2 = {"legal_moves": fail, "piece_on_square": fail}
    assert compute_graded_hard_penalty(res_fatal2) == -1.00


def test_reward_v2_substantiation_floor() -> None:
    """Verify that vague responses with zero concrete references are capped at 0.15."""
    vague_text = (
        "This move is a terrible blunder that loses the game immediately because "
        "it ignores the opponent plan and weakens the overall position on the board "
        "instead of finding a much better alternative option for the player."
    )
    res_v2 = compute_reward(
        text=vague_text,
        fen=ITALIAN_FEN,
        level="club",
        detail="short",
        verdict="Blunder",
        best_move_san="d3",
        facts=["After Ng5, Black's best reply is d5."],
        reward_version="v2",
    )
    assert res_v2.passed_all_hard is True
    assert res_v2.reward <= 0.15


def test_multi_objective_reward_streams() -> None:
    """Verify grounding, pedagogy, and style reward streams for MO-GRPO."""
    prompt = (
        "task: explain_move\nlevel: club\ndetail: short\n\n"
        f"FEN: {ITALIAN_FEN}\nWhite to move, move 4.\n"
        "Move played: Ng5.\nVerdict: Good.\nBest move: d3.\n"
        "Facts:\n- Ng5 attacks the pawn on f7 together with the bishop on c4.\n"
        "- After Ng5, Black's best reply is d5."
    )
    good_comp = (
        "Playing Ng5 is a good aggressive move in this opening position. "
        "It jumps the knight forward to attack the vulnerable pawn on f7 together with the bishop on c4. "
        "Black should respond actively with d5 to challenge the center."
    )
    bad_comp = "I played Qxe5 because my rook on e4 was pinned."

    g_scores = reward_grounding_fn([prompt, prompt], [good_comp, bad_comp])
    p_scores = reward_pedagogy_fn([prompt, prompt], [good_comp, bad_comp])
    s_scores = reward_style_fn([prompt, prompt], [good_comp, bad_comp])

    assert g_scores[0] == 0.0
    assert g_scores[1] < 0.0
    assert p_scores[0] > 0.40
    assert p_scores[1] == 0.0
    assert s_scores[0] > 0.25
    assert s_scores[1] == 0.0


def test_load_stage_t4_grpo_v2_configs() -> None:
    """Verify all Stage T4 GRPO v2 YAML configurations validate cleanly."""
    cfg_b2 = load_grpo_config("training/configs/grpo_v2_b2_dapo.yaml")
    assert cfg_b2.loss_type == "dapo"
    assert cfg_b2.reward_version == "v2"
    assert cfg_b2.multi_objective_aggregation == "sum_then_normalize"
    assert cfg_b2.per_device_train_batch_size == 8
    assert cfg_b2.gradient_accumulation_steps == 8

    cfg_b3 = load_grpo_config("training/configs/grpo_v2_b3_drgrpo.yaml")
    assert cfg_b3.loss_type == "dr_grpo"
    assert cfg_b3.max_steps == 150

    cfg_b4 = load_grpo_config("training/configs/grpo_v2_b4_gdpo.yaml")
    assert cfg_b4.multi_objective_aggregation == "normalize_then_sum"


def test_detect_anomalies_v2_telemetry() -> None:
    """Verify group_length_std, zero_var_group_pct, first_person_pct, and motif_mention_pct."""
    records = []
    # Group 1: identical word counts (0 std) and identical rewards (zero var)
    for i in range(8):
        records.append({
            "word_count": 45,
            "passed_all_hard": True,
            "reward": 0.85,
            "opening_prefix": f"opening {i} words here",
            "has_first_person": (i == 0),
            "has_motif_or_best_move": True,
        })
    anomalies = detect_anomalies(records, group_size=8)
    assert anomalies["group_length_std"] == 0.0
    assert anomalies["low_length_diversity"] is True
    assert anomalies["zero_var_group_pct"] == 100.0
    assert anomalies["first_person_pct"] == 12.5
    assert anomalies["motif_mention_pct"] == 100.0


def test_cmdp_primal_dual_update() -> None:
    """Verify CMDP dual variable lambda_rule updates on high violation batch."""
    monitor = MonitoredGRPOReward(
        log_interval_batches=5,
        reward_version="v2",
        use_cmdp_primal_dual=True,
        lambda_rule_init=1.0,
        eta_lambda=0.5,
        target_violation_rate=0.04,
    )
    prompt = f"task: explain_move\nFEN: {ITALIAN_FEN}\nMove played: Ng5.\nVerdict: Good.\nFacts:\n- Material: equal."
    bad_comp = "I thought my knight was safe."
    scores = monitor([prompt], [bad_comp])
    assert scores[0] == -0.35
    # 100% violation rate > 4% target -> lambda_rule increases above 1.0
    assert monitor.lambda_rule > 1.0


def test_evaluate_grpo_v2_runner_and_bundle(tmp_path: Path) -> None:
    """Verify Stage T4 evaluator, notebook syntax, and phase 4 Colab bundle."""
    nb_path = Path("training/grpo_v2.ipynb")
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) == 10
    for idx, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        clean_lines = [
            line if not line.lstrip().startswith("!") else f"# {line}"
            for line in source.splitlines()
        ]
        ast.parse("\n".join(clean_lines), filename=f"grpo_v2_cell_{idx}")

    dev_item = {
        "id": "dev_test_1",
        "fen": ITALIAN_FEN,
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "verdict": "Good",
        "move_uci": "f3g5",
        "facts": ["Ng5 attacks the pawn on f7 together with the bishop on c4."],
        "user_message": "Move played: Ng5.\nVerdict: Good.\nBest move: d3.\nFacts:\n- Ng5 attacks the pawn on f7 together with the bishop on c4.",
    }
    comp = (
        "Playing Ng5 is a good attacking move in this opening position. "
        "It jumps the knight forward to target the vulnerable pawn on f7 alongside the bishop on c4. "
        "This creates immediate tactical pressure against the opponent king."
    )
    item_res = evaluate_grpo_v2_item(dev_item, comp)
    assert item_res["passed_all_hard"] is True
    assert item_res["reward"] > 0.75

    summary, pass_res = compute_grpo_v2_summary([item_res])
    assert pass_res.passed_all_gates is True
    report = generate_grpo_v2_report(summary, pass_res, experiment_id="RL-B2")
    assert "Stage T4 GRPO v2 Evaluation Report: RL-B2" in report

    bundle_zip = tmp_path / "test_phase4.zip"
    built = build_phase4_grpo_v2_bundle(output_path=bundle_zip)
    assert built.is_file()
    with zipfile.ZipFile(built, "r") as zf:
        names = set(zf.namelist())
        assert "data/processed/rl_prompts_v2.jsonl" in names
        assert "training/configs/grpo_v2_b2_dapo.yaml" in names
        assert "training/grpo_v2.ipynb" in names
        assert "training/evaluate_grpo_v2.py" in names
        assert "eval/testset-1.0.jsonl" not in names
