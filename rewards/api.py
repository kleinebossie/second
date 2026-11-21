"""Public API for prompt parsing, scoring, and reinforcement learning.

This module provides prompt-driven evaluation functions for GRPO and benchmarks.
It parses structured prompt texts and scores assistant completions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Final

import chess

from .checks import BAD_VERDICTS, GOOD_VERDICTS, check_verdict_tone
from .parser import (
    NEGATIVE_TONE_PHRASES,
    POSITIVE_TONE_PHRASES,
    parse_verdict_words,
)
from .scorer import RewardResult, compute_reward

logger: logging.Logger = logging.getLogger(__name__)

# Precompiled regex patterns for prompt extraction
USER_TURN_RE: Final[re.Pattern[str]] = re.compile(
    r"<\|im_start\|>user\r?\n?(.*?)(?:<\|im_end\|>|$)",
    re.DOTALL,
)
THINK_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*<think>.*?</think>\s*",
    re.DOTALL,
)
TASK_RE: Final[re.Pattern[str]] = re.compile(
    r"^task:\s*([^\n\r]+)",
    re.MULTILINE,
)
LEVEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^level:\s*([^\n\r]+)",
    re.MULTILINE,
)
DETAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"^detail:\s*([^\n\r]+)",
    re.MULTILINE,
)
FEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^FEN:\s*([^\n\r]*)",
    re.MULTILINE,
)
MOVE_PLAYED_RE: Final[re.Pattern[str]] = re.compile(
    r"^Move played:\s*([^\n\r]+)",
    re.MULTILINE,
)
VERDICT_RE: Final[re.Pattern[str]] = re.compile(
    r"^Verdict:\s*([^\n\r]+)",
    re.MULTILINE,
)
BEST_MOVE_RE: Final[re.Pattern[str]] = re.compile(
    r"^Best move:\s*([^\n\r]+)",
    re.MULTILINE,
)
BEST_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^Best line:\s*([^\n\r]+)",
    re.MULTILINE,
)

# Lowercase verdict sets for accuracy checks
LOWER_BAD_VERDICTS: Final[set[str]] = {v.lower() for v in BAD_VERDICTS}
LOWER_GOOD_VERDICTS: Final[set[str]] = {v.lower() for v in GOOD_VERDICTS}


@dataclass(frozen=True)
class PromptInfo:
    """Structured fields parsed from a prompt string.

    The dataclass stores task parameters, position data, and facts.
    """

    task: str
    level: str
    detail: str
    fen: str
    side_to_move: str
    move_played_san: str | None
    move_uci: str | None
    verdict: str | None
    best_move_san: str | None
    best_line: list[str]
    facts: list[str]


def _extract_user_message(prompt: str) -> str:
    """Extract user message content from plain text or chat templates.

    If chat markers exist, return the content between user tags.
    Otherwise return the cleaned prompt text.
    """
    if "<|im_start|>user" in prompt:
        match = USER_TURN_RE.search(prompt)
        if match:
            return match.group(1).strip()
    return prompt.strip()


def parse_prompt(user_message: str) -> PromptInfo:
    """Parse structured fields from a user prompt or chat prompt string.

    If the prompt contains chat template tokens, extract the user turn.
    If the FEN line is missing or invalid, raise a ValueError.
    Return a PromptInfo instance with all extracted fields.
    """
    text = _extract_user_message(user_message)

    # 1. Parse header fields
    task_match = TASK_RE.search(text)
    task = task_match.group(1).strip() if task_match else "explain_move"

    level_match = LEVEL_RE.search(text)
    level = level_match.group(1).strip() if level_match else "club"

    detail_match = DETAIL_RE.search(text)
    detail = detail_match.group(1).strip() if detail_match else "short"

    # 2. Parse FEN line
    fen_match = FEN_RE.search(text)
    if not fen_match:
        raise ValueError("Prompt is missing required FEN line.")
    fen = fen_match.group(1).strip()
    if not fen:
        raise ValueError("Prompt contains an empty FEN line.")

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN in prompt: {fen}") from exc

    side_to_move = "white" if board.turn == chess.WHITE else "black"

    # 3. Parse move played and convert to UCI
    move_played_san: str | None = None
    move_uci: str | None = None
    if task == "explain_move":
        move_match = MOVE_PLAYED_RE.search(text)
        if move_match:
            raw_move = move_match.group(1).strip().rstrip(".")
            if raw_move:
                move_played_san = raw_move
                try:
                    move_obj = board.parse_san(move_played_san)
                    move_uci = move_obj.uci()
                except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                    move_uci = None

    # 4. Parse verdict
    verdict: str | None = None
    if task == "explain_move":
        verdict_match = VERDICT_RE.search(text)
        if verdict_match:
            raw_verdict = verdict_match.group(1).strip().rstrip(".")
            if raw_verdict:
                verdict = raw_verdict

    # 5. Parse best move
    best_move_san: str | None = None
    best_match = BEST_MOVE_RE.search(text)
    if best_match:
        raw_best = best_match.group(1).strip().rstrip(".")
        if raw_best:
            best_move_san = raw_best

    # 6. Parse best line
    best_line: list[str] = []
    line_match = BEST_LINE_RE.search(text)
    if line_match:
        raw_line = line_match.group(1).strip().rstrip(".")
        best_line = [m for m in raw_line.split() if m]

    # 7. Parse facts
    facts: list[str] = []
    lines = text.splitlines()
    facts_idx: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == "Facts:":
            facts_idx = i
            break

    if facts_idx is not None:
        for line in lines[facts_idx + 1 :]:
            stripped = line.strip()
            if stripped.startswith("- "):
                facts.append(stripped[2:].strip())

    return PromptInfo(
        task=task,
        level=level,
        detail=detail,
        fen=fen,
        side_to_move=side_to_move,
        move_played_san=move_played_san,
        move_uci=move_uci,
        verdict=verdict,
        best_move_san=best_move_san,
        best_line=best_line,
        facts=facts,
    )


def score_completion(
    prompt: str,
    completion: str,
    reward_version: str = "v1",
) -> RewardResult:
    """Score a model completion against a prompt string.

    Strip a leading thinking block from the completion.
    Parse positional and factual metadata from the prompt.
    Pass the extracted user turn as prompt_text to compute_reward.
    Return the computed RewardResult.
    """
    cleaned_completion = THINK_BLOCK_RE.sub("", completion).strip()
    user_message = _extract_user_message(prompt)
    info = parse_prompt(user_message)

    return compute_reward(
        text=cleaned_completion,
        fen=info.fen,
        level=info.level,
        detail=info.detail,
        verdict=info.verdict,
        move_uci=info.move_uci,
        best_move_san=info.best_move_san or "",
        facts=info.facts,
        prompt_text=user_message,
        reward_version=reward_version,
    )


def _extract_content(item: Any, role_target: str) -> str:
    """Extract message text from string or message dict collections."""
    if isinstance(item, str):
        return item

    if isinstance(item, list):
        for msg in reversed(item):
            if isinstance(msg, dict) and msg.get("role") == role_target:
                return str(msg.get("content", ""))
        if item and isinstance(item[-1], dict):
            return str(item[-1].get("content", ""))
        if item:
            return str(item[-1])
        return ""

    if isinstance(item, dict):
        return str(item.get("content", ""))

    return str(item)


def _score_batch(
    prompts: list[Any],
    completions: list[Any],
    default_reward_version: str,
    extractor: Any,
    fn_name: str,
    fallback_value: float,
    kwargs: dict[str, Any],
) -> list[float]:
    """Score a batch of prompt/completion pairs with a stream extractor."""
    reward_version = str(kwargs.get("reward_version", default_reward_version))
    rewards: list[float] = []
    for prompt_item, completion_item in zip(prompts, completions):
        try:
            prompt_str = _extract_content(prompt_item, role_target="user")
            completion_str = _extract_content(
                completion_item, role_target="assistant"
            )
            result = score_completion(
                prompt_str, completion_str, reward_version=reward_version
            )
            rewards.append(float(extractor(result)))
        except Exception as exc:
            logger.warning("Failed to score sample in %s: %s", fn_name, exc)
            rewards.append(fallback_value)
    return rewards


def grpo_reward_fn(
    prompts: list[Any],
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """Calculate rewards for a batch of prompts and completions in GRPO.

    Accept plain strings or message dictionary lists.
    Extract the last user prompt and assistant response from conversations.
    If evaluation of one sample fails, log a warning and return negative one.
    Return a list of scalar rewards.
    """
    return _score_batch(
        prompts,
        completions,
        default_reward_version="v1",
        extractor=lambda result: result.reward,
        fn_name="grpo_reward_fn",
        fallback_value=-1.0,
        kwargs=kwargs,
    )


def reward_grounding_fn(
    prompts: list[Any],
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """Compute grounding stream reward for MO-GRPO / GDPO.

    Return 0.0 when all hard checks pass.
    Return graded negative penalty when any hard check fails.
    """
    return _score_batch(
        prompts,
        completions,
        default_reward_version="v2",
        extractor=lambda result: 0.0 if result.passed_all_hard else result.reward,
        fn_name="reward_grounding_fn",
        fallback_value=-1.0,
        kwargs=kwargs,
    )


def reward_pedagogy_fn(
    prompts: list[Any],
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """Compute pedagogy stream reward for MO-GRPO / GDPO.

    Combine key_point and better_move multiplied by the target length multiplier.
    Return 0.0 when any hard check fails.
    """
    from .scorer import V2_WEIGHT_BETTER_MOVE, V2_WEIGHT_KEY_POINT

    def _extract_ped(result: RewardResult) -> float:
        if not result.passed_all_hard:
            return 0.0
        kp = result.soft_scores.get("key_point", 0.0)
        bm = result.soft_scores.get("better_move", 0.0)
        m_len = result.soft_scores.get("length_multiplier", 1.0)
        ped_score = (V2_WEIGHT_KEY_POINT * kp + V2_WEIGHT_BETTER_MOVE * bm) * m_len
        return round(float(ped_score), 4)

    return _score_batch(
        prompts,
        completions,
        default_reward_version="v2",
        extractor=_extract_ped,
        fn_name="reward_pedagogy_fn",
        fallback_value=0.0,
        kwargs=kwargs,
    )


def reward_style_fn(
    prompts: list[Any],
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """Compute style stream reward for MO-GRPO / GDPO.

    Combine level_fit, not_copied, and no_repeats multiplied by the length multiplier.
    Return 0.0 when any hard check fails.
    """
    from .scorer import (
        V2_WEIGHT_LEVEL_FIT,
        V2_WEIGHT_NO_REPEATS,
        V2_WEIGHT_NOT_COPIED,
    )

    def _extract_style(result: RewardResult) -> float:
        if not result.passed_all_hard:
            return 0.0
        lf = result.soft_scores.get("level_fit", 0.0)
        nc = result.soft_scores.get("not_copied", 0.0)
        nr = result.soft_scores.get("no_repeats", 0.0)
        m_len = result.soft_scores.get("length_multiplier", 1.0)
        style_score = (
            V2_WEIGHT_LEVEL_FIT * lf
            + V2_WEIGHT_NOT_COPIED * nc
            + V2_WEIGHT_NO_REPEATS * nr
        ) * m_len
        return round(float(style_score), 4)

    return _score_batch(
        prompts,
        completions,
        default_reward_version="v2",
        extractor=_extract_style,
        fn_name="reward_style_fn",
        fallback_value=0.0,
        kwargs=kwargs,
    )



def verdict_accuracy(text: str, verdict: str | None) -> bool | None:
    """Check whether explanation text agrees with the move verdict.

    If verdict is None or text contains no verdict class, return None.
    If verdict is an Inaccuracy, Mistake, or Blunder, group it as bad.
    If verdict is a Best, Good, or Only move, group it as good.

    Exact class matches count as correct.
    Matches within the same group also count as correct.
    Players and small models often interchange severity terms such as mistake and blunder.
    Both terms agree on the error direction of the move.

    If text only contains opposite group terms, return False.
    Return None when text contains no verdict terms.
    """
    if not verdict or not text:
        return None

    norm_verdict = verdict.strip().lower()
    if norm_verdict in LOWER_BAD_VERDICTS:
        target_group = "bad"
    elif norm_verdict in LOWER_GOOD_VERDICTS:
        target_group = "good"
    else:
        return None

    words = parse_verdict_words(text)
    pos_words = [w for w in words if w in POSITIVE_TONE_PHRASES]
    neg_words = [w for w in words if w in NEGATIVE_TONE_PHRASES]

    if not pos_words and not neg_words:
        return None

    if target_group == "bad":
        # Agree if text uses negative error terms
        if neg_words:
            return True
        # Disagree if text only uses positive terms
        return False

    # target_group == "good"
    # Disagree if text uses unnegated negative terms directed at the move
    if neg_words and not check_verdict_tone(text, verdict).passed:
        return False

    # Agree if text uses positive move terms
    if pos_words:
        return True

    # When negative terms were negated and no positive terms exist,
    # the text states no verdict class.
    return None
