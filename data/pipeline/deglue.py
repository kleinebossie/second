"""De-gluing and vocabulary validation for chess comments.

This module separates glued words from lost line breaks.
It preserves chess notation, numbers, and compound words.
It also removes reader-directed questions from comments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

# Path to the vendored vocabulary and frequency database
VOCAB_PATH: Final[Path] = Path(__file__).resolve().parent / "deglue_vocab.json"

# Common valid two-letter English words allowed in splits
VALID_2LETTER_WORDS: Final[set[str]] = {
    "to", "of", "in", "is", "it", "on", "he", "as", "at", "by",
    "we", "or", "an", "my", "so", "up", "if", "me", "no", "go",
    "do", "us", "am", "be",
}

# Standard chess SAN move regular expression
SAN_MOVE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|"
    r"[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|"
    r"O-O(?:-O)?|"
    r"\d+\.+(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8])?)$",
    re.IGNORECASE,
)

# Move prefix with move notation e.g. 14...Nxe5 or 11.Ne5
MOVE_WITH_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^\d+\.+(?:[a-zA-Z0-9+=#]+)?$",
    re.IGNORECASE,
)

# Ordinal numbers regular expression e.g. 1st, 2nd, 3rd, 19th
ORDINAL_RE: Final[re.Pattern[str]] = re.compile(
    r"^\d+(?:st|nd|rd|th)$",
    re.IGNORECASE,
)

# Reader-directed question patterns
READER_QUESTION_PATTERNS: Final[tuple[str, ...]] = (
    r"what\s+do\s+you\s+think",
    r"who\s+do\s+you\s+think",
    r"do\s+you\s+agree",
    r"any\s+ideas",
    r"can\s+you\s+see",
    r"what\s+would\s+you",
    r"how\s+would\s+you",
    r"would\s+you\s+agree",
    r"any\s+suggestions",
    r"do\s+you\s+see",
    r"can\s+you\s+find",
    r"what\s+about\s+you",
    r"what\s+do\s+you\s+say",
    r"see\s+what\s+you\s+think",
    r"dear\s+reader",
    r"question\s+for\s+the\s+reader",
    r"do\s+you\s+the\s+jury\s+agree",
)

READER_RE: Final[re.Pattern[str]] = re.compile(
    r"(" + "|".join(READER_QUESTION_PATTERNS) + r")",
    re.IGNORECASE,
)

TRAILING_QUESTION_CLAUSE_RE: Final[re.Pattern[str]] = re.compile(
    r"[,;\u2013\u2014\-]\s*(?:" + "|".join(READER_QUESTION_PATTERNS) + r")[^.!?]*\?*\s*$",
    re.IGNORECASE,
)


def _load_vocab_data() -> tuple[dict[str, int], set[str]]:
    """Load word frequencies and vocabulary from the vendored file.

    Return a tuple of frequency dictionary and vocabulary set.
    """
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
            frequencies = data.get("frequencies", {})
            vocabulary = set(data.get("vocabulary", []))
            return frequencies, vocabulary
    return {}, set()


WORD_FREQUENCIES, VOCABULARY = _load_vocab_data()


def deglue_token(token: str) -> str | None:
    """Separate a single glued token into two words.

    If the token is in the vocabulary, return None.
    If the token has a valid split, return the two words.
    Otherwise, return None.
    """
    token_lower = token.lower()
    if token_lower in VOCABULARY:
        return None

    token_len = len(token_lower)
    if token_len < 3:
        return None

    candidates: list[tuple[str, str, int, int]] = []
    for split_index in range(1, token_len):
        left_part = token_lower[:split_index]
        right_part = token_lower[split_index:]

        # Enforce length rules
        if len(left_part) == 1 and left_part not in ("a", "i"):
            continue
        if len(right_part) == 1 and right_part not in ("a", "i"):
            continue
        if len(left_part) == 2 and left_part not in VALID_2LETTER_WORDS:
            continue
        if len(right_part) == 2 and right_part not in VALID_2LETTER_WORDS:
            continue

        left_count = WORD_FREQUENCIES.get(left_part, 0)
        right_count = WORD_FREQUENCIES.get(right_part, 0)

        # High confidence check: both parts known, count >= 2, and at least one part >= 10
        if left_count >= 2 and right_count >= 2 and (left_count >= 10 or right_count >= 10):
            score = left_count * right_count
            candidates.append((left_part, right_part, score, split_index))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[2], reverse=True)
    best_candidate = candidates[0]

    # Check for ambiguity if multiple splits exist
    if len(candidates) > 1:
        second_candidate = candidates[1]
        if best_candidate[2] < 3 * second_candidate[2]:
            return None

    split_index = best_candidate[3]
    left_result = token[:split_index]
    right_result = token[split_index:]

    # Keep capitalization of the first letter
    if token[0].isupper():
        left_result = left_result[0].upper() + left_result[1:].lower()
    else:
        left_result = left_result.lower()

    if len(token) > split_index and token[split_index].isupper():
        right_result = right_result[0].upper() + right_result[1:].lower()
    else:
        right_result = right_result.lower()

    return f"{left_result} {right_result}"


def deglue_comment(text: str) -> str:
    """Separate glued words in a chess comment.

    Do not split chess notation, numbers, or compound words.
    Keep the capitalization of the first letter.
    Return the cleaned comment.
    """
    # Add space after punctuation preceded by a letter and followed immediately by a letter
    spaced_text = re.sub(r"(?<=[A-Za-z])([.!?])([A-Za-z])", r"\1 \2", text)
    spaced_text = re.sub(r"(?<=[A-Za-z])([,:;])([A-Za-z])", r"\1 \2", spaced_text)

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        start_index = match.start()
        end_index = match.end()

        # Check for adjacent hyphens or dashes
        if start_index > 0 and spaced_text[start_index - 1] in "-\u2010\u2011\u2012\u2013\u2014\u2015":
            return token
        if end_index < len(spaced_text) and spaced_text[end_index] in "-\u2010\u2011\u2012\u2013\u2014\u2015":
            return token

        # Check for adjacent contraction apostrophes
        if start_index > 1 and spaced_text[start_index - 1] in "'’" and spaced_text[start_index - 2].isalpha():
            return token
        if end_index < len(spaced_text) - 1 and spaced_text[end_index] in "'’" and spaced_text[end_index + 1].isalpha():
            return token

        split_result = deglue_token(token)
        if split_result is not None:
            return split_result
        return token

    return re.sub(r"\b[A-Za-z]+\b", replace_token, spaced_text)


def remove_reader_questions(text: str) -> str:
    """Remove reader-directed question sentences from a comment.

    Remove sentences that ask questions to the reader.
    Keep rhetorical chess questions.
    Return the comment without reader questions.
    """
    # Remove trailing reader question clauses separated by comma or dash
    cleaned = TRAILING_QUESTION_CLAUSE_RE.sub(".", text)

    # Split remaining text by sentence boundary
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    kept_sentences: list[str] = []

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue
        if READER_RE.search(sentence_clean):
            # Check if this sentence is a question or reader address
            if "?" in sentence_clean or any(
                pattern in sentence_clean.lower()
                for pattern in ("what do you think", "dear reader", "question for the reader", "any suggestions")
            ):
                continue
        kept_sentences.append(sentence_clean)

    result = " ".join(kept_sentences).strip()
    return re.sub(r"\s+", " ", result).strip()


def is_unknown_token(token: str) -> bool:
    """Check if a token is unknown or suspicious.

    Return False if the token is a known word, number, or move.
    Return True if the token is unknown.
    """
    clean_token = token.lower().strip(".,;:!?()[]\"'—–\u2010\u2011\u2012\u2013\u2014\u2015")
    if not clean_token:
        return False
    if clean_token.isdigit():
        return False
    if ORDINAL_RE.match(clean_token):
        return False
    if SAN_MOVE_RE.match(clean_token):
        return False
    if MOVE_WITH_PREFIX_RE.match(clean_token):
        return False
    if clean_token in VOCABULARY:
        return False
    if clean_token.endswith("'s") and clean_token[:-2] in VOCABULARY:
        return False
    if clean_token.endswith("’s") and clean_token[:-2] in VOCABULARY:
        return False
    if re.match(r"^[a-h][1-8]$", clean_token):
        return False
    return True


def count_unknown_tokens(text: str) -> int:
    """Count suspicious tokens in a comment.

    Return the total number of unknown tokens.
    """
    tokens = text.split()
    return sum(1 for token in tokens if is_unknown_token(token))


def has_too_many_unknown_tokens(text: str, max_unknown: int = 2) -> bool:
    """Check if a comment has too many unknown tokens.

    If unknown token count exceeds max_unknown, return True.
    Otherwise, return False.
    """
    return count_unknown_tokens(text) > max_unknown
