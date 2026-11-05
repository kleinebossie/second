"""Unit tests for the rewards API module.

This module tests prompt parsing, completion scoring, GRPO functions, and verdict accuracy.
It also evaluates adversarial and reward-hacking explanation outputs.
"""

from __future__ import annotations

import pytest

from rewards.api import (
    PromptInfo,
    grpo_reward_fn,
    parse_prompt,
    score_completion,
    verdict_accuracy,
)

# Sample explain_move user prompt
SAMPLE_MOVE_USER_PROMPT = """task: explain_move
level: club
detail: short

FEN: rn1qkb1r/pp2pp1p/2pp2p1/5bB1/3Pn3/5N1P/PPPNQPP1/R3KB1R b KQkq - 3 7
Black to move, move 7.
Move played: Qa5.
Verdict: Mistake.
Black's winning chances: 71% → 60%.
Best move: Nxg5.
Best line: Nxg5 Nxg5 h6 Nge4 Bg7 c3.
Facts:
- Qa5 attacks the knight on d2 together with the knight on e4.
- After Qa5, White's best reply is Bf4.
- White can capture the knight on e4 with Qxe4.
- Black threatens Qxd2+.
- Black is up a pawn.
- Phase: opening."""

# Sample Qwen chat template for explain_move
SAMPLE_MOVE_CHAT_PROMPT = f"""<|im_start|>system
You are Second, a chess assistant. You explain chess positions and moves clearly using the provided facts.<|im_end|>
<|im_start|>user
{SAMPLE_MOVE_USER_PROMPT}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

# Sample explain_position user prompt
SAMPLE_POSITION_USER_PROMPT = """task: explain_position
level: beginner
detail: short

FEN: rn2kb1r/pbpp1p1p/1p2p2p/6q1/3PP1B1/P1N5/1PP2PPP/R2QK1NR b KQkq - 3 7
Black to move, move 7.
Black's winning chances: 46%.
Best move: Nc6.
Best line: Nc6 Bf3 Bg7 Nge2 h5 h4.
Facts:
- Top candidate moves: Nc6, h5, Bg7.
- Black has a check: Qd2+.
- Black can capture the bishop on g4 with Qxg4.
- Material: equal.
- Phase: opening."""

SAMPLE_POSITION_CHAT_PROMPT = f"""<|im_start|>system
You are Second, a chess assistant.<|im_end|>
<|im_start|>user
{SAMPLE_POSITION_USER_PROMPT}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


def test_parse_prompt_explain_move() -> None:
    """Verify field extraction from an explain_move prompt."""
    info = parse_prompt(SAMPLE_MOVE_USER_PROMPT)

    assert isinstance(info, PromptInfo)
    assert info.task == "explain_move"
    assert info.level == "club"
    assert info.detail == "short"
    assert (
        info.fen
        == "rn1qkb1r/pp2pp1p/2pp2p1/5bB1/3Pn3/5N1P/PPPNQPP1/R3KB1R b KQkq - 3 7"
    )
    assert info.side_to_move == "black"
    assert info.move_played_san == "Qa5"
    assert info.move_uci == "d8a5"
    assert info.verdict == "Mistake"
    assert info.best_move_san == "Nxg5"
    assert info.best_line == ["Nxg5", "Nxg5", "h6", "Nge4", "Bg7", "c3"]
    assert len(info.facts) == 6
    assert info.facts[0] == "Qa5 attacks the knight on d2 together with the knight on e4."


def test_parse_prompt_chat_template() -> None:
    """Verify that chat template input produces identical PromptInfo results."""
    info_plain = parse_prompt(SAMPLE_MOVE_USER_PROMPT)
    info_chat = parse_prompt(SAMPLE_MOVE_CHAT_PROMPT)
    assert info_chat == info_plain


def test_parse_prompt_explain_position() -> None:
    """Verify field extraction from an explain_position prompt."""
    info = parse_prompt(SAMPLE_POSITION_USER_PROMPT)

    assert info.task == "explain_position"
    assert info.level == "beginner"
    assert info.detail == "short"
    assert (
        info.fen
        == "rn2kb1r/pbpp1p1p/1p2p2p/6q1/3PP1B1/P1N5/1PP2PPP/R2QK1NR b KQkq - 3 7"
    )
    assert info.side_to_move == "black"
    assert info.move_played_san is None
    assert info.move_uci is None
    assert info.verdict is None
    assert info.best_move_san == "Nc6"
    assert info.best_line == ["Nc6", "Bf3", "Bg7", "Nge2", "h5", "h4"]
    assert len(info.facts) == 5

    info_chat = parse_prompt(SAMPLE_POSITION_CHAT_PROMPT)
    assert info_chat == info


def test_parse_prompt_missing_or_invalid_fen_raises() -> None:
    """Verify that missing or invalid FEN strings raise a clear ValueError."""
    with pytest.raises(ValueError, match="missing required FEN"):
        parse_prompt("task: explain_move\nlevel: club\n")

    with pytest.raises(ValueError, match="empty FEN"):
        parse_prompt("task: explain_move\nFEN: \n")

    with pytest.raises(ValueError, match="Invalid FEN"):
        parse_prompt("task: explain_move\nFEN: not_a_real_fen\n")


def test_score_completion_with_think_prefix() -> None:
    """Verify that score_completion strips thinking blocks correctly."""
    completion_body = (
        "This move is a mistake because Black should have played Nxg5. "
        "Qa5 allows White to defend and keep an active position."
    )
    completion_with_think = (
        "<think>\n"
        "Let us examine candidate moves and threats.\n"
        "The best move is Nxg5.\n"
        "</think>\n"
        f"{completion_body}"
    )

    res_clean = score_completion(SAMPLE_MOVE_USER_PROMPT, completion_body)
    res_think = score_completion(SAMPLE_MOVE_USER_PROMPT, completion_with_think)

    assert res_clean.passed_all_hard is True
    assert res_think.passed_all_hard is True
    assert res_clean.reward == res_think.reward


def test_grpo_reward_fn_string_and_conversational() -> None:
    """Verify grpo_reward_fn on string inputs, message lists, and faulty samples."""
    completion_text = (
        "This move is a mistake. Black should have played Nxg5 instead to capture the bishop. "
        "The queen move allows White to respond with Bf4."
    )

    prompts = [
        SAMPLE_MOVE_USER_PROMPT,
        [
            {"role": "system", "content": "You are Second."},
            {"role": "user", "content": SAMPLE_MOVE_USER_PROMPT},
        ],
        "task: explain_move\nlevel: beginner\nno fen here",
    ]
    completions = [
        completion_text,
        [{"role": "assistant", "content": completion_text}],
        completion_text,
    ]

    rewards = grpo_reward_fn(prompts, completions)

    assert len(rewards) == 3
    assert rewards[0] > 0.0
    assert rewards[1] == rewards[0]
    # Faulty sample returns -1.0 without raising
    assert rewards[2] == -1.0


def test_verdict_accuracy_agreement() -> None:
    """Verify verdict accuracy evaluation on agreement and severity cases."""
    # Exact match on bad group
    assert verdict_accuracy("This is a blunder.", "Blunder") is True

    # Same bad group with different severity (mistake vs inaccuracy)
    assert verdict_accuracy("This move is a mistake.", "Inaccuracy") is True

    # Same bad group with different severity (blunder vs mistake)
    assert verdict_accuracy("This is a major blunder that loses material.", "Mistake") is True

    # Exact match on good group
    assert verdict_accuracy("Best move in this position.", "Best") is True

    # Same good group with different severity (good move vs best move)
    assert verdict_accuracy("A very good move that controls the center.", "Best") is True


def test_verdict_accuracy_disagreement_and_missing() -> None:
    """Verify verdict accuracy on disagreements, missing words, and missing verdicts."""
    # Praise on blunder
    assert verdict_accuracy("Great move! Very strong play.", "Blunder") is False

    # Criticism on good move
    assert verdict_accuracy("A terrible mistake that loses.", "Good") is False

    # Neutral explanation without verdict words returns None
    assert verdict_accuracy("White develops the knight to f3.", "Good") is None

    # Empty verdict returns None
    assert verdict_accuracy("This is a blunder.", None) is None

    # Negated criticism with positive move term agrees with good verdict
    assert (
        verdict_accuracy(
            "White avoids any mistake with this great move.",
            "Good",
        )
        is True
    )

    # Negated criticism without positive move term states no verdict class
    assert (
        verdict_accuracy(
            "White avoids any mistake here.",
            "Good",
        )
        is None
    )


# --- Adversarial Tests (Target Reward < 0.60) ---


def test_adversarial_generic_sentence_with_piece_names_and_instead() -> None:
    """Verify that generic sentences with piece names and 'instead' fail to get high reward."""
    # Adv (a): Generic sentence mentioning pieces and 'instead' without position content
    adv_text = (
        "The knight moves to help the king and queen instead. "
        "The pawn moves forward to protect the pieces. "
        "The bishop and rook stay active in the game."
    )
    res = score_completion(SAMPLE_MOVE_USER_PROMPT, adv_text)
    assert res.reward < 0.60
    assert res.soft_scores["key_point"] == 0.0
    assert res.soft_scores["better_move"] == 0.0


def test_adversarial_pasted_fact_lines() -> None:
    """Verify that fact lines pasted as prose do not receive a high reward."""
    # Adv (b): Pasting prompt facts verbatim as prose
    adv_text = (
        "Qa5 attacks the knight on d2 together with the knight on e4. "
        "After Qa5, White best reply is Bf4. "
        "White can capture the knight on e4 with Qxe4."
    )
    res = score_completion(SAMPLE_MOVE_USER_PROMPT, adv_text)
    assert res.reward < 0.60
    assert res.soft_scores["not_copied"] == 0.0


def test_adversarial_move_list() -> None:
    """Verify that listing move sequences without explanation receives a low or negative reward."""
    # Adv (c): Listing chess moves without prose
    adv_text = (
        "Nxg5 Nxg5 h6 Nge4 Bg7 c3 Qa5 Bf4 d5 Nd2 O-O "
        "Nf3 Qb6 Qc1 Kh7 Be2 Nd7 O-O Re8 Re1 e5 dxe5 Nxe5 Nxe5 Bxe5 Bxe5"
    )
    res = score_completion(SAMPLE_MOVE_USER_PROMPT, adv_text)
    assert res.reward < 0.60


def test_adversarial_repeated_sentence() -> None:
    """Verify that repeated sentences receive penalties and score below 0.60."""
    # Adv (d): The same sentence repeated multiple times
    adv_text = (
        "The knight on e4 is in danger of being lost. "
        "The knight on e4 is in danger of being lost. "
        "The knight on e4 is in danger of being lost."
    )
    res = score_completion(SAMPLE_MOVE_USER_PROMPT, adv_text)
    assert res.reward < 0.60
    assert res.soft_scores["no_repeats"] == 0.0
