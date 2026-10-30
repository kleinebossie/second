"""Second chess shared core module.

This module provides fact extraction and prompt formatting functions.
It maintains exact parity with the TypeScript core implementation.
"""

from __future__ import annotations

from .facts import (
    classify_verdict,
    extract_facts,
    win_chance,
)
from .openings import (
    OPENING_BOOK,
    lookup_opening,
)
from .prompt import (
    SYSTEM_MESSAGE,
    build_prompt_messages,
    build_user_message,
    format_qwen_chat_prompt,
)

__all__ = [
    "SYSTEM_MESSAGE",
    "extract_facts",
    "build_user_message",
    "build_prompt_messages",
    "format_qwen_chat_prompt",
    "win_chance",
    "classify_verdict",
    "OPENING_BOOK",
    "lookup_opening",
]
