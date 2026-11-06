"""Training module for Project Second.

This module provides data loading, tokenization, and configuration helpers for fine-tuning.
"""

from .grpo_data import (
    GRPOConfigParams,
    MonitoredGRPOReward,
    detect_anomalies,
    load_grpo_config,
    load_rl_prompt_records,
    sample_outputs_for_inspection,
)
from .sft_data import (
    AnswerOnlyDataCollator,
    SFTConfigParams,
    build_non_thinking_chat_prompt,
    collate_sft_batch,
    load_sft_config,
    load_sft_records,
    resolve_text_linear_modules,
    tokenize_sft_record,
)

__all__ = [
    "AnswerOnlyDataCollator",
    "GRPOConfigParams",
    "MonitoredGRPOReward",
    "SFTConfigParams",
    "build_non_thinking_chat_prompt",
    "collate_sft_batch",
    "detect_anomalies",
    "load_grpo_config",
    "load_rl_prompt_records",
    "load_sft_config",
    "load_sft_records",
    "resolve_text_linear_modules",
    "sample_outputs_for_inspection",
    "tokenize_sft_record",
]
