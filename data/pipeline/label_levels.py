"""Level labeling for chess commentary.

This module classifies chess explanations into beginner or club difficulty levels.
Beginner explanations require short sentences, no advanced chess jargon, and no engine evaluations.
Club explanations allow deeper chess terminology and varied sentence structures.
"""

from __future__ import annotations

import re
from typing import Final

from rewards.checks import ENGINE_JARGON_WORDS, EVAL_NUMBER_RE
from rewards.scorer import CHESS_JARGON_WORDS

# Combined chess concepts and engine jargon prohibited for beginner level
JARGON_WORDS: Final[list[str]] = list(
    dict.fromkeys(CHESS_JARGON_WORDS + ENGINE_JARGON_WORDS)
)


def calculate_average_sentence_length(text: str) -> float:
    """Calculate the average number of words per sentence.

    Split text by sentence terminators.
    Ignore empty sentences.
    Return zero when no sentences exist.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return sum(lengths) / len(lengths)


def has_jargon_or_eval(text: str) -> bool:
    """Check whether text contains advanced chess jargon or evaluation numbers.

    Search for words in the jargon vocabulary.
    Search for signed or decimal evaluation numbers.
    Return true if any match is found.
    """
    text_lower = text.lower()
    for word in JARGON_WORDS:
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, text_lower):
            return True

    if EVAL_NUMBER_RE.search(text):
        return True

    return False


def label_level(comment: str) -> str:
    """Assign beginner or club level to a chess comment.

    When average sentence length is at most 15 words and no jargon is present, return beginner.
    Otherwise return club.
    """
    avg_len = calculate_average_sentence_length(comment)
    if avg_len <= 15.0 and not has_jargon_or_eval(comment):
        return "beginner"
    return "club"
