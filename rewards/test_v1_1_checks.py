"""Unit tests for Phase 11 Reward v3 checks and evaluation metrics.

This module verifies boilerplate rejection, positional fact grounding, and v3 scoring.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.metrics import HARD_CHECK_NAMES_V3, evaluate_completions, evaluate_item, get_hard_check_names
from eval.run_eval import generate_markdown_report
from rewards.api import (
    reward_grounding_fn,
    reward_pedagogy_fn,
    reward_style_fn,
    score_completion,
)
from rewards.checks import (
    check_beginner_jargon,
    check_positional_fact_grounding,
    check_vague_boilerplate,
)
from rewards.scorer import compute_reward
from training.grpo_data import MonitoredGRPOReward, build_multi_objective_reward_funcs


def test_rejects_unsupported_coordination_and_harmony_boilerplate() -> None:
    """Reject unsupported coordination, harmony, overall, and vague weakness cliches."""
    facts = [
        "After exd4, White's best reply is Nxd4.",
        "After exd4, White's recapture Nxd4 centralizes the knight.",
        "Best move e4 keeps central space and tension instead of trading on d4.",
        "Material: equal.",
        "Phase: opening.",
    ]

    res_coord = check_vague_boilerplate(
        "This move blocks coordination and leaves your pieces tangled.",
        facts=facts,
    )
    assert not res_coord.passed
    assert "coordination" in res_coord.reason.lower()

    res_harmony = check_vague_boilerplate(
        "Playing e4 instead maintains harmony and gives a stronger overall position.",
        facts=facts,
    )
    assert not res_harmony.passed

    res_weak = check_vague_boilerplate(
        "This move creates unnecessary weaknesses in the position.",
        facts=facts,
    )
    assert not res_weak.passed

    # Legitimate square-anchored weakness passes
    res_weak_sq = check_vague_boilerplate(
        "Trading on d4 creates an unnecessary weakness on c6 and gives White central space.",
        facts=facts,
    )
    assert res_weak_sq.passed

    # Coordination allowed if explicitly present in facts
    facts_with_coord = facts + ["This move blocks rook coordination."]
    res_coord_ok = check_vague_boilerplate(
        "This move blocks coordination on the back rank.",
        facts=facts_with_coord,
    )
    assert res_coord_ok.passed


def test_rejects_ungrounded_generic_filler() -> None:
    """Reject empty generic filler when no concrete square, move, or positional concept appears."""
    facts = [
        "After a3, Black's best reply is Nf6.",
        "Best move d4 claims central space on d4.",
        "Material: equal.",
    ]
    res_bad = check_vague_boilerplate(
        "This is a weak positional move that does not improve the position.",
        facts=facts,
    )
    assert not res_bad.passed

    res_good = check_vague_boilerplate(
        "a3 does not improve the position because d4 claims central space on d4.",
        facts=facts,
    )
    assert res_good.passed


def test_positional_fact_grounding_on_6_exd4() -> None:
    """Enforce positional concept grounding on 6... exd4 when no tactic exists."""
    exd4_pos_facts = [
        "After exd4, White's best reply is Nxd4.",
        "After exd4, White's recapture Nxd4 centralizes the knight.",
        "Best move e4 keeps central space and tension instead of trading on d4.",
        "Unlike exd4, e4 prevents White from centralizing pieces.",
        "exd4 releases central tension on d4.",
        "Material: equal.",
        "Phase: opening.",
    ]

    # Bare move mention without positional reasoning must fail
    res_fail = check_positional_fact_grounding(
        "exd4 is an inaccuracy. Playing e4 is better here.",
        facts=exd4_pos_facts,
        verdict="Inaccuracy",
    )
    assert not res_fail.passed

    # Beginner-safe central space or centralization concept must pass
    res_pass_space = check_positional_fact_grounding(
        "exd4 releases tension too early. Pushing e4 keeps more space in the center.",
        facts=exd4_pos_facts,
        verdict="Inaccuracy",
    )
    assert res_pass_space.passed

    res_pass_center = check_positional_fact_grounding(
        "Trading on d4 lets White bring the knight to the center with Nxd4. Push e4 instead.",
        facts=exd4_pos_facts,
        verdict="Inaccuracy",
    )
    assert res_pass_center.passed


def test_positional_grounding_skips_when_tactical_facts_exist() -> None:
    """Pass positional grounding immediately when tactical facts exist in prompt."""
    tactical_facts = [
        "After a3, Black's best reply is Qxb2.",
        "The pawn on b2 is not defended.",
        "Best move O-O castles to keep the king safe.",
        "Material: equal.",
    ]
    res = check_positional_fact_grounding(
        "a3 leaves the b2 pawn hanging. Play O-O instead.",
        facts=tactical_facts,
        verdict="Blunder",
    )
    assert res.passed


def test_beginner_level_positional_grounding_compatible_with_jargon_check() -> None:
    """Confirm beginner positional explanations pass both v3 grounding and beginner jargon checks."""
    facts = [
        "After a3, Black's best reply is Rd2.",
        "Best move Rad1 places the rook on an open file.",
        "The d-file is open.",
        "Material: equal.",
    ]
    beginner_text = (
        "Playing a3 is slow. Rad1 puts your rook on the open d-file. "
        "This controls the open line right away."
    )
    assert check_positional_fact_grounding(beginner_text, facts=facts).passed
    assert check_beginner_jargon(beginner_text, level="beginner").passed


def test_reward_v3_backwards_compatibility_and_scoring() -> None:
    """Verify v2 stays unchanged while v3 runs eleven checks and graded penalties."""
    fen = "rnb1k2r/ppppqppp/5n2/4p3/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R b KQkq - 0 6"
    facts = [
        "After exd4, White's best reply is Nxd4.",
        "After exd4, White's recapture Nxd4 centralizes the knight.",
        "Best move e4 keeps central space and tension instead of trading on d4.",
        "Unlike exd4, e4 prevents White from centralizing pieces.",
        "Material: equal.",
        "Phase: opening.",
    ]
    prompt = (
        "task: explain_move\nlevel: club\ndetail: short\n\n"
        f"FEN: {fen}\nBlack to move, move 6.\nMove played: exd4.\n"
        "Verdict: Inaccuracy.\nBlack's winning chances: 49% → 43%.\n"
        "Best move: e4.\nBest line: e4 Ng1 Bxc3 Bxc3.\nFacts:\n"
        + "\n".join(f"- {f}" for f in facts)
    )

    bad_boilerplate = (
        "exd4 is an inaccuracy because it blocks coordination and creates unnecessary weaknesses. "
        "Instead, e4 is better here and keeps a solid overall position for Black in this opening."
    )
    # v2 does not enforce v3 boilerplate or positional grounding checks
    res_v2 = compute_reward(
        text=bad_boilerplate,
        fen=fen,
        level="club",
        detail="short",
        verdict="Inaccuracy",
        move_uci="e5d4",
        best_move_san="e4",
        facts=facts,
        prompt_text=prompt,
        reward_version="v2",
    )
    assert res_v2.passed_all_hard

    # v3 rejects boilerplate and grades penalty (-0.35 for 1 style check, -0.70 for 2)
    res_v3_bad = compute_reward(
        text=bad_boilerplate,
        fen=fen,
        level="club",
        detail="short",
        verdict="Inaccuracy",
        move_uci="e5d4",
        best_move_san="e4",
        facts=facts,
        prompt_text=prompt,
        reward_version="v3",
    )
    assert len(res_v3_bad.hard_checks) == 11
    assert not res_v3_bad.passed_all_hard
    assert not res_v3_bad.hard_checks["vague_boilerplate"].passed
    assert res_v3_bad.reward < 0.0

    # Missing positional grounding alone incurs -0.35
    res_v3_ungrounded = compute_reward(
        text=(
            "Playing exd4 is an inaccuracy that misses a stronger option in this opening. "
            "Instead, Black should play e4 here to keep the game on a better path for Black."
        ),
        fen=fen,
        level="club",
        detail="short",
        verdict="Inaccuracy",
        move_uci="e5d4",
        best_move_san="e4",
        facts=facts,
        prompt_text=prompt,
        reward_version="v3",
    )
    assert not res_v3_ungrounded.passed_all_hard
    assert not res_v3_ungrounded.hard_checks["positional_grounding"].passed
    assert res_v3_ungrounded.reward == -0.35

    # Strong grounded response passes all 11 checks with positive reward
    good_text = (
        "Playing exd4 is an inaccuracy because it releases central tension too early "
        "and lets White centralize the knight with Nxd4. Instead, pushing e4 keeps "
        "central space and stops White from claiming the d4 square."
    )
    res_v3_good = score_completion(prompt, good_text, reward_version="v3")
    assert res_v3_good.passed_all_hard
    assert len(res_v3_good.hard_checks) == 11
    assert res_v3_good.reward > 0.6

    # GDPO stream helpers respect reward_version="v3"
    assert reward_grounding_fn([prompt], [bad_boilerplate], reward_version="v3")[0] < 0.0
    assert reward_grounding_fn([prompt], [good_text], reward_version="v3")[0] == 0.0
    assert reward_pedagogy_fn([prompt], [good_text], reward_version="v3")[0] > 0.0
    assert reward_style_fn([prompt], [good_text], reward_version="v3")[0] > 0.0


def test_eval_metrics_supports_reward_v3() -> None:
    """Verify eval.metrics reports all 11 hard checks when reward_version is v3."""
    assert len(HARD_CHECK_NAMES_V3) == 11
    item = {
        "id": "v3_item_1",
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "move_uci": "e1g1",
        "verdict": "Best",
        "facts": [
            "After O-O, Black's best reply is d6.",
            "O-O castles to safeguard the king.",
            "Material: equal.",
            "Phase: opening.",
        ],
        "completion": (
            "O-O is the best move because castling safeguards the king early in the opening. "
            "White tucks the king away safely and prepares to develop the remaining pieces."
        ),
    }
    m = evaluate_item(item, reward_version="v3")
    assert m.passed_all_hard
    assert set(m.hard_checks.keys()) == set(HARD_CHECK_NAMES_V3)

    metrics = evaluate_completions([item], reward_version="v3")
    assert set(metrics.aggregate.hard_check_fail_rates.keys()) == set(HARD_CHECK_NAMES_V3)
    assert metrics.aggregate.hard_fail_rate == 0.0


def test_reward_v3_positional_grounding_on_real_test_301_and_loopholes() -> None:
    """Verify v3 catches ungrounded prose on real test_301 and closes boilerplate loopholes."""
    testset_path = Path(__file__).resolve().parent.parent / "eval" / "testset-1.1.jsonl"
    assert testset_path.exists()
    records = [json.loads(line) for line in testset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_301 = next(r for r in records if r["id"] == "test_301")

    # Using COMPLETE real facts list from test_301 (not hand-trimmed)
    ungrounded_res = compute_reward(
        text="exd4 is an inaccuracy. Playing e4 is better here for Black.",
        fen=test_301["fen"],
        level=test_301["level"],
        detail=test_301["detail"],
        verdict=test_301["verdict"],
        move_uci=test_301["move_uci"],
        facts=test_301["facts"],
        prompt_text=test_301["user_message"],
        reward_version="v3",
    )
    assert not ungrounded_res.passed_all_hard
    assert not ungrounded_res.hard_checks["positional_grounding"].passed

    grounded_res = compute_reward(
        text=(
            "exd4 is an inaccuracy because it releases central tension and lets White "
            "centralize the knight with Nxd4. Playing e4 keeps central space instead."
        ),
        fen=test_301["fen"],
        level=test_301["level"],
        detail=test_301["detail"],
        verdict=test_301["verdict"],
        move_uci=test_301["move_uci"],
        facts=test_301["facts"],
        prompt_text=test_301["user_message"],
        reward_version="v3",
    )
    assert grounded_res.passed_all_hard

    # Pawn-move loopholes in UNNECESSARY_WEAKNESS_RE and GENERIC_FILLER_RE
    pos_facts = [
        "After a3, Black's best reply is d5.",
        "Best move d4 claims central space on d4.",
        "Material: equal.",
    ]
    assert not check_vague_boilerplate(
        "a3 is a weak positional move that does not improve White's position.",
        facts=pos_facts,
    ).passed
    assert not check_vague_boilerplate(
        "Playing e4 creates unnecessary weaknesses.",
        facts=pos_facts,
    ).passed

    # Expanded coordination and harmony boilerplate patterns
    for bad_phrase in (
        "This move blocks rook coordination.",
        "Playing a3 reduces piece coordination.",
        "Playing e4 keeps a harmonious position.",
        "This move creates disharmony.",
        "Developing now improves the overall position.",
    ):
        assert not check_vague_boilerplate(bad_phrase, facts=pos_facts).passed, f"Should reject: {bad_phrase}"

    # Tightened king_safety and space_tension concept patterns
    castle_facts = [
        "After h3, Black's best reply is Nf6.",
        "Best move O-O castles to keep the king safe.",
        "Material: equal.",
    ]
    assert not check_positional_fact_grounding("Playing h3 is a safe move.", facts=castle_facts).passed

    space_facts = [
        "After a3, Black's best reply is Nc6.",
        "Best move d4 claims central space on d4.",
        "Material: equal.",
    ]
    assert not check_positional_fact_grounding("White needs more control and a quick push.", facts=space_facts).passed

    # Background "White has not castled." alone does not force king_safety
    bg_facts = [
        "After h3, Black's best reply is Nf6.",
        "White has not castled.",
        "Material: equal.",
    ]
    assert check_positional_fact_grounding("h3 prevents Bg4.", facts=bg_facts).passed


def test_reward_v3_round3_tightened_loopholes_gdpo_and_report() -> None:
    """Verify Rule 3/4 loopholes, tightened positional patterns, v3 GDPO forwarding, and report output."""
    pos_facts = [
        "After a3, Black's best reply is d5.",
        "Best move d4 claims central space on d4.",
        "Material: equal.",
    ]
    # Rule 4: generic filler fails even when mentioning two moves with no square or positional concept
    assert not check_vague_boilerplate(
        "a3 is a passive move compared to O-O.",
        facts=pos_facts,
    ).passed
    assert not check_vague_boilerplate(
        "a3 keeps a flexible position while Nc3 was also legal.",
        facts=pos_facts,
    ).passed

    # Rule 3: trade clause square does not exempt vague weakness claims
    assert not check_vague_boilerplate(
        "Trading on d4 creates unnecessary weaknesses.",
        facts=pos_facts,
    ).passed
    assert not check_vague_boilerplate(
        "This leaves structural weaknesses behind.",
        facts=pos_facts,
    ).passed
    assert check_vague_boilerplate(
        "Trading on d4 creates an unnecessary weakness on c3.",
        facts=pos_facts,
    ).passed

    # Expanded coordination / harmony / overall patterns
    for bad_phrase in (
        "This harms coordination of the pieces.",
        "White maintains harmony across the board.",
        "Black keeps a harmonious setup here.",
        "The overall piece position favors White.",
        "This hurts the position overall.",
    ):
        assert not check_vague_boilerplate(bad_phrase, facts=pos_facts).passed, f"Should reject: {bad_phrase}"

    # Tightened centralization and space_tension patterns reject bare center/middle/trade
    cent_facts = ["After Qxd4, White's recapture Qxd4 centralizes the queen.", "Material: equal."]
    assert not check_positional_fact_grounding(
        "After Qxd4, White fights in the middle area.",
        facts=cent_facts,
    ).passed
    assert check_positional_fact_grounding(
        "After Qxd4, White puts the queen in the center.",
        facts=cent_facts,
    ).passed

    assert not check_positional_fact_grounding(
        "White should focus on the center rather than a trade.",
        facts=pos_facts,
    ).passed
    assert check_positional_fact_grounding(
        "Playing d4 takes central space instead of trading.",
        facts=pos_facts,
    ).passed

    # activity_dev accepts test_311 phrasing ("brings the knight out")
    act_facts = ["Nc3 develops the knight toward the center.", "Material: equal."]
    assert check_positional_fact_grounding(
        "Nc3 brings the knight out and prepares to castle.",
        facts=act_facts,
    ).passed

    # build_multi_objective_reward_funcs forwards v3 to all 3 streams
    monitor = MonitoredGRPOReward(reward_version="v3")
    streams = build_multi_objective_reward_funcs(monitor)
    assert [fn.__name__ for fn in streams] == ["reward_grounding_fn", "reward_pedagogy_fn", "reward_style_fn"]

    fen_exd4 = "r1bqk2r/pp3ppp/2n1pn2/3p4/1b1P4/2NB1N2/PP3PPP/R1BQK2R b KQkq - 0 7"
    prompt_exd4 = (
        "task: explain_move\nlevel: club\ndetail: short\n\n"
        f"FEN: {fen_exd4}\nBlack to move, move 7.\nMove played: a6.\nVerdict: Inaccuracy.\n"
        "Black's winning chances: 48% → 42%.\nBest move: e5.\nBest line: e5 dxe5.\nFacts:\n"
        "- After a6, White's best reply is O-O.\n"
        "- Best move e5 claims central space on e5.\n"
        "- Material: equal.\n- Phase: opening."
    )
    boilerplate_text = (
        "a6 is an inaccuracy that blocks piece coordination. "
        "Black should play e5 for a stronger overall position."
    )
    g_val = streams[0]([prompt_exd4], [boilerplate_text])[0]
    p_val = streams[1]([prompt_exd4], [boilerplate_text])[0]
    s_val = streams[2]([prompt_exd4], [boilerplate_text])[0]
    assert g_val < 0.0
    assert p_val == 0.0
    assert s_val == 0.0

    # get_hard_check_names + generate_markdown_report reflect reward_version
    assert len(get_hard_check_names("v1")) == 6
    assert len(get_hard_check_names("v2")) == 9
    assert len(get_hard_check_names("v3")) == 11

    m_res = evaluate_completions(
        [
            {
                "id": "rep_1",
                "task": "explain_move",
                "level": "club",
                "detail": "short",
                "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "move_uci": "e1g1",
                "verdict": "Best",
                "facts": ["O-O castles to safeguard the king.", "Material: equal.", "Phase: opening."],
                "completion": "O-O castles to safeguard the king and connects the rooks cleanly.",
            }
        ],
        reward_version="v3",
    )
    md_v3 = generate_markdown_report("Second-1.1", m_res, reward_version="v3")
    assert "`v3`" in md_v3
    assert "vague_boilerplate" in md_v3
    assert "positional_grounding" in md_v3
    assert "-0.50 per fatal hallucination check" in md_v3
    assert "| `key_point` | 0.42 |" in md_v3
    assert "| `better_move` | 0.23 |" in md_v3
    assert "| `length` | multiplier |" in md_v3

    # Recapture word alone must not pass centralization; activity recapture requires activity_dev
    from rewards.claims import _extract_positional_categories

    testset_path = Path(__file__).resolve().parent.parent / "eval" / "testset-1.1.jsonl"
    test_301 = next(
        json.loads(line)
        for line in testset_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("id") == "test_301"
    )
    assert not check_positional_fact_grounding(
        "exd4 is an inaccuracy because White recaptures with Nxd4. Instead, e4 is better.",
        facts=test_301["facts"],
        verdict=test_301["verdict"],
    ).passed
    test_310_facts = [
        "After dxc5, Black's best reply is Bxc5.",
        "After dxc5, Black's recapture Bxc5 increases bishop activity.",
        "Material: equal.",
    ]
    assert _extract_positional_categories(test_310_facts) == {"activity_dev"}

    # Coordination evasion variants & movement-verb square exclusion ("reaches e4")
    for coord_bad in (
        "This move lacks coordination cleanly.",
        "This move limits coordination on the queenside.",
        "This move undermines coordination cleanly.",
        "Black plays without coordination here.",
        "Piece coordination suffers after this push.",
    ):
        assert not check_vague_boilerplate(coord_bad, facts=pos_facts).passed, f"Should reject: {coord_bad}"

    assert not check_vague_boilerplate(
        "This move creates unnecessary weaknesses and reaches e4.",
        facts=pos_facts,
    ).passed
    assert not check_vague_boilerplate(
        "This move is a weak move and reaches e4.",
        facts=pos_facts,
    ).passed
