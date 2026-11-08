"""GRPO data loading, configuration validation, and reward monitoring.

This module provides data loading and configuration validation for Phase 6 GRPO RL.
It wraps rule checks into a monitored reward function.
The reward function logs diagnostics and detects reward hacking.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import random
from typing import Any

from eval.metrics import clean_completion_text, count_words
from rewards.api import (
    _extract_content,
    _extract_user_message,
    reward_grounding_fn,
    reward_pedagogy_fn,
    reward_style_fn,
    score_completion,
)
from training.sft_data import (
    ALLOWED_BASE_MODELS,
    _read_yaml_file,
    build_non_thinking_chat_prompt,
)

logger: logging.Logger = logging.getLogger(__name__)

ALLOWED_LOSS_TYPES: set[str] = {"grpo", "bnpo", "dr_grpo", "dapo"}
ALLOWED_REWARD_VERSIONS: set[str] = {"v1", "v2"}
ALLOWED_MULTI_OBJECTIVE_AGGREGATIONS: set[str] = {
    "sum_then_normalize",
    "normalize_then_sum",
}


@dataclass
class GRPOConfigParams:
    """Parameters for Group Relative Policy Optimization training."""

    model_name: str
    version: str
    size: str
    base_model: str
    text_only: bool
    enable_thinking: bool
    sft_adapter_dir: str
    rl_prompts_path: str
    val_data_path: str
    testset_path: str
    num_generations: int
    max_completion_length: int
    max_prompt_length: int
    temperature: float
    learning_rate: float
    beta: float
    max_steps: int
    warmup_steps: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    save_total_limit: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gpu: str
    checkpoint_dir: str
    loss_type: str = "dapo"
    reward_version: str = "v1"
    multi_objective_aggregation: str = "sum_then_normalize"
    use_cmdp_primal_dual: bool = False
    devset_path: str = "eval/devset-1.0.jsonl"
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def grpo(self) -> dict[str, Any]:
        """Return the GRPO section of the raw configuration."""
        return self.raw_config.get("grpo", {})

    @property
    def mode(self) -> dict[str, Any]:
        """Return the mode section of the raw configuration."""
        return self.raw_config.get("mode", {})


def load_grpo_config(
    config_path: str | Path = "training/configs/second-1.0-2b.yaml",
) -> GRPOConfigParams:
    """Load and validate the GRPO configuration.

    If the configuration file does not exist, raise FileNotFoundError.
    If any parameter fails validation, raise ValueError.
    Return validated GRPOConfigParams.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw = _read_yaml_file(path)
    mode = raw.get("mode", {})
    grpo = raw.get("grpo", {})

    base_model = raw.get("base_model", "")
    text_only = mode.get("text_only")
    enable_thinking = mode.get("enable_thinking")

    num_generations = grpo.get("num_generations")
    max_completion_length = grpo.get("max_completion_length")
    temperature = grpo.get("temperature")
    learning_rate = grpo.get("learning_rate")
    beta = grpo.get("beta")
    max_steps = grpo.get("max_steps")
    save_steps = grpo.get("save_steps")
    per_device_train_batch_size = grpo.get("per_device_train_batch_size")
    gradient_accumulation_steps = grpo.get("gradient_accumulation_steps")
    loss_type = str(grpo.get("loss_type", "dapo"))
    reward_version = str(grpo.get("reward_version", "v1"))
    multi_objective_aggregation = str(
        grpo.get("multi_objective_aggregation", "sum_then_normalize")
    )
    use_cmdp_primal_dual = bool(grpo.get("use_cmdp_primal_dual", False))

    if base_model not in ALLOWED_BASE_MODELS:
        raise ValueError(
            f"Invalid base_model: expected one of {sorted(ALLOWED_BASE_MODELS)}, got '{base_model}'"
        )
    if text_only is not True:
        raise ValueError(f"Invalid text_only: expected True, got {text_only}")
    if enable_thinking is not False:
        raise ValueError(
            f"Invalid enable_thinking: expected False, got {enable_thinking}"
        )
    if num_generations != 8:
        raise ValueError(f"Invalid num_generations: expected 8, got {num_generations}")
    if max_completion_length != 300:
        raise ValueError(
            f"Invalid max_completion_length: expected 300, got {max_completion_length}"
        )
    if temperature is None or not (0.5 <= float(temperature) <= 1.5):
        raise ValueError(
            f"Invalid temperature: expected 0.5 <= temperature <= 1.5, got {temperature}"
        )
    if learning_rate is None or not (0 < float(learning_rate) <= 0.001):
        raise ValueError(
            f"Invalid learning_rate: expected 0 < learning_rate <= 0.001, got {learning_rate}"
        )
    if beta is None or not (0.0 <= float(beta) <= 0.5):
        raise ValueError(
            f"Invalid beta: expected 0.0 <= beta <= 0.5, got {beta}"
        )
    min_allowed_steps = 150 if reward_version == "v2" else 300
    if not (isinstance(max_steps, int) and min_allowed_steps <= max_steps <= 600):
        raise ValueError(
            f"Invalid max_steps: expected {min_allowed_steps} <= max_steps <= 600, got {max_steps}"
        )
    if not (isinstance(save_steps, int) and 25 <= save_steps <= 50):
        raise ValueError(
            f"Invalid save_steps: expected 25 <= save_steps <= 50, got {save_steps}"
        )
    if loss_type not in ALLOWED_LOSS_TYPES:
        raise ValueError(
            f"Invalid loss_type: expected one of {sorted(ALLOWED_LOSS_TYPES)}, got '{loss_type}'"
        )
    if reward_version not in ALLOWED_REWARD_VERSIONS:
        raise ValueError(
            f"Invalid reward_version: expected one of {sorted(ALLOWED_REWARD_VERSIONS)}, got '{reward_version}'"
        )
    if multi_objective_aggregation not in ALLOWED_MULTI_OBJECTIVE_AGGREGATIONS:
        raise ValueError(
            f"Invalid multi_objective_aggregation: expected one of {sorted(ALLOWED_MULTI_OBJECTIVE_AGGREGATIONS)}, got '{multi_objective_aggregation}'"
        )

    if not isinstance(per_device_train_batch_size, int) or per_device_train_batch_size <= 0:
        raise ValueError(
            f"Invalid per_device_train_batch_size: expected positive integer, got {per_device_train_batch_size}"
        )
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps <= 0:
        raise ValueError(
            f"Invalid gradient_accumulation_steps: expected positive integer, got {gradient_accumulation_steps}"
        )

    effective_batch = per_device_train_batch_size * gradient_accumulation_steps
    if effective_batch % num_generations != 0:
        raise ValueError(
            f"Invalid batch size: per_device_train_batch_size * gradient_accumulation_steps "
            f"({effective_batch}) must be divisible by num_generations ({num_generations})"
        )

    return GRPOConfigParams(
        model_name=str(raw.get("model_name", "Second-1.0-2B")),
        version=str(raw.get("version", "1.0")),
        size=str(raw.get("size", "2B")),
        base_model=base_model,
        text_only=bool(text_only),
        enable_thinking=bool(enable_thinking),
        sft_adapter_dir=str(grpo.get("sft_adapter_dir", "")),
        rl_prompts_path=str(grpo.get("rl_prompts_path", "data/processed/rl_prompts.jsonl")),
        val_data_path=str(grpo.get("val_data_path", "data/processed/sft_val.jsonl")),
        testset_path=str(grpo.get("testset_path", "eval/testset-1.0.jsonl")),
        num_generations=int(num_generations),
        max_completion_length=int(max_completion_length),
        max_prompt_length=int(grpo.get("max_prompt_length", 512)),
        temperature=float(temperature),
        learning_rate=float(learning_rate),
        beta=float(beta),
        max_steps=int(max_steps),
        warmup_steps=int(grpo.get("warmup_steps", 50)),
        save_steps=int(save_steps),
        eval_steps=int(grpo.get("eval_steps", 50)),
        logging_steps=int(grpo.get("logging_steps", 10)),
        save_total_limit=int(grpo.get("save_total_limit", 3)),
        per_device_train_batch_size=int(per_device_train_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        gpu=str(grpo.get("gpu", "L4")),
        checkpoint_dir=str(grpo.get("checkpoint_dir", "")),
        loss_type=loss_type,
        reward_version=reward_version,
        multi_objective_aggregation=multi_objective_aggregation,
        use_cmdp_primal_dual=use_cmdp_primal_dual,
        devset_path=str(grpo.get("devset_path", "eval/devset-1.0.jsonl")),
        raw_config=raw,
    )


def load_rl_prompt_records(
    path: str | Path,
    max_prompts: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> list[dict[str, Any]]:
    """Load prompt records for reinforcement learning from a JSONL file.

    If the file does not exist, raise FileNotFoundError.
    Format each prompt with the non-thinking chat template.
    Remove heavy Stockfish evaluation fields to save memory.
    If shuffle is True, randomize records with the provided seed.
    If max_prompts is positive, return at most that number of records.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"RL prompt file not found: {path_obj}")

    records: list[dict[str, Any]] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)

            # Ensure user_message exists
            if "user_message" not in rec or not rec["user_message"]:
                if "chat_prompt" in rec and rec["chat_prompt"]:
                    rec["user_message"] = _extract_user_message(rec["chat_prompt"])
                else:
                    raise ValueError("Record missing chat_prompt and user_message.")

            # Ensure chat_prompt exists
            if "chat_prompt" not in rec or not rec["chat_prompt"]:
                rec["chat_prompt"] = build_non_thinking_chat_prompt(
                    rec["user_message"]
                )

            # Set prompt column for TRL GRPOTrainer
            rec["prompt"] = rec["chat_prompt"]

            # Remove heavy Stockfish PV data to conserve memory
            rec.pop("eval_before", None)
            rec.pop("eval_after", None)

            records.append(rec)
            if not shuffle and max_prompts is not None and max_prompts > 0:
                if len(records) >= max_prompts:
                    break

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    if max_prompts is not None and max_prompts > 0:
        records = records[:max_prompts]

    return records


def _compute_std(values: list[float]) -> float:
    """Compute population standard deviation for a list of numbers."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / n
    return math.sqrt(variance)


def detect_anomalies(
    records: list[dict[str, Any]],
    group_size: int = 8,
) -> dict[str, Any]:
    """Detect reward hacking and training collapse in scored completions.

    Flag length collapse when the average word count is below 15 words.
    Flag low group length diversity when group_length_std is below 4.0 words.
    Flag robotic text when the top four-word prefix exceeds 40 percent.
    Flag high hard-fail when the hard-fail rate exceeds 50 percent.
    Track zero_var_group_pct, first_person_pct, and motif_mention_pct.
    Return a dictionary with anomaly flags and metric values.
    """
    total = len(records)
    if total == 0:
        return {
            "length_collapse": False,
            "low_length_diversity": False,
            "robotic_text": False,
            "high_hard_fail": False,
            "avg_word_count": 0.0,
            "group_length_std": 0.0,
            "zero_var_group_pct": 0.0,
            "first_person_pct": 0.0,
            "motif_mention_pct": 0.0,
            "hard_fail_rate": 0.0,
            "mean_reward": 0.0,
            "top_opening_prefix": "",
            "top_opening_share": 0.0,
        }

    total_words = sum(int(r.get("word_count", 0)) for r in records)
    avg_word_count = round(total_words / total, 2)

    hard_fails = sum(1 for r in records if not r.get("passed_all_hard", True))
    hard_fail_rate = round(100.0 * hard_fails / total, 2)

    total_reward = sum(float(r.get("reward", 0.0)) for r in records)
    mean_reward = round(total_reward / total, 4)

    first_person_count = sum(
        1 for r in records if bool(r.get("has_first_person", False))
    )
    first_person_pct = round(100.0 * first_person_count / total, 2)

    motif_count = sum(
        1 for r in records if bool(r.get("has_motif_or_best_move", False))
    )
    motif_mention_pct = round(100.0 * motif_count / total, 2)

    # Compute within-group standard deviations across groups of size G
    eff_group = max(1, group_size)
    group_len_stds: list[float] = []
    zero_var_groups = 0
    num_groups = 0

    for i in range(0, total, eff_group):
        grp = records[i : i + eff_group]
        if len(grp) >= 2:
            num_groups += 1
            grp_words = [float(r.get("word_count", 0)) for r in grp]
            grp_rewards = [float(r.get("reward", 0.0)) for r in grp]
            group_len_stds.append(_compute_std(grp_words))
            if _compute_std(grp_rewards) < 1e-4:
                zero_var_groups += 1

    group_length_std = (
        round(sum(group_len_stds) / len(group_len_stds), 2)
        if group_len_stds
        else 0.0
    )
    zero_var_group_pct = (
        round(100.0 * zero_var_groups / num_groups, 2)
        if num_groups > 0
        else 0.0
    )
    low_length_diversity = bool(num_groups > 0 and group_length_std < 4.0)

    non_empty = [
        r for r in records
        if r.get("word_count", 0) > 0 and r.get("opening_prefix")
    ]
    top_opening_prefix = ""
    top_opening_share = 0.0
    robotic_text = False

    if len(non_empty) >= 10:
        counter = Counter(r["opening_prefix"] for r in non_empty)
        most_common = counter.most_common(1)
        if most_common:
            top_opening_prefix = most_common[0][0]
            count = most_common[0][1]
            top_opening_share = round(count / len(non_empty), 4)
            if top_opening_share > 0.40:
                robotic_text = True
    elif non_empty:
        counter = Counter(r["opening_prefix"] for r in non_empty)
        most_common = counter.most_common(1)
        if most_common:
            top_opening_prefix = most_common[0][0]
            count = most_common[0][1]
            top_opening_share = round(count / len(non_empty), 4)

    length_collapse = avg_word_count < 15.0
    high_hard_fail = hard_fail_rate > 50.0

    return {
        "length_collapse": length_collapse,
        "low_length_diversity": low_length_diversity,
        "robotic_text": robotic_text,
        "high_hard_fail": high_hard_fail,
        "avg_word_count": avg_word_count,
        "group_length_std": group_length_std,
        "zero_var_group_pct": zero_var_group_pct,
        "first_person_pct": first_person_pct,
        "motif_mention_pct": motif_mention_pct,
        "hard_fail_rate": hard_fail_rate,
        "mean_reward": mean_reward,
        "top_opening_prefix": top_opening_prefix,
        "top_opening_share": top_opening_share,
    }


def sample_outputs_for_inspection(
    records: list[dict[str, Any]],
    count: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample records deterministically for human review.

    If the record count is less than or equal to count, return all records.
    Otherwise, select sample records using the random seed.
    """
    if len(records) <= count:
        return list(records)

    rng = random.Random(seed)
    return rng.sample(records, count)


def build_multi_objective_reward_funcs(
    monitor: MonitoredGRPOReward | None = None,
) -> list[Any]:
    """Return the three split reward callables for MO-GRPO / GDPO training.

    When a MonitoredGRPOReward instance is provided, wrap grounding with telemetry.
    Return [grounding_fn, reward_pedagogy_fn, reward_style_fn].
    """
    if monitor is None:
        return [reward_grounding_fn, reward_pedagogy_fn, reward_style_fn]

    def _monitored_grounding_fn(
        prompts: list[Any],
        completions: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        # Record telemetry via monitor while returning only grounding stream
        monitor(prompts, completions, **kwargs)
        return reward_grounding_fn(prompts, completions, **kwargs)

    _monitored_grounding_fn.__name__ = "reward_grounding_fn"
    return [_monitored_grounding_fn, reward_pedagogy_fn, reward_style_fn]


class MonitoredGRPOReward:
    """Reward function wrapper with diagnostics and anti-hacking monitoring.

    Score completions with rule checks for reinforcement learning.
    Log mean reward, collapse metrics, and failure rates across training steps.
    Support Reward v1 and Reward v2, plus optional CMDP primal-dual multiplier.
    Save sample completions for manual review.
    """

    __name__: str = "grpo_rule_reward"

    def __init__(
        self,
        log_interval_batches: int = 25,
        sample_size: int = 20,
        seed: int = 42,
        reward_version: str = "v1",
        group_size: int = 8,
        use_cmdp_primal_dual: bool = False,
        lambda_rule_init: float = 1.0,
        eta_lambda: float = 0.5,
        target_violation_rate: float = 0.04,
    ) -> None:
        """Initialize the reward monitor with logging intervals and reward version."""
        self.__name__: str = "grpo_rule_reward"
        self.log_interval_batches = log_interval_batches
        self.sample_size = sample_size
        self.seed = seed
        self.reward_version = reward_version
        self.group_size = group_size
        self.use_cmdp_primal_dual = use_cmdp_primal_dual
        self.lambda_rule: float = lambda_rule_init
        self.eta_lambda: float = eta_lambda
        self.target_violation_rate: float = target_violation_rate
        self.call_count: int = 0
        self.current_window: list[dict[str, Any]] = []
        self.step_summaries: list[dict[str, Any]] = []
        self.sampled_outputs: dict[int, list[dict[str, Any]]] = {}

    def summarize_window(self) -> dict[str, Any]:
        """Summarize diagnostics across the current batch window.

        Return anomaly flags, averages, and sample count for the window.
        """
        anomalies = detect_anomalies(
            self.current_window,
            group_size=self.group_size,
        )
        return {
            "step": self.call_count,
            "sample_count": len(self.current_window),
            "lambda_rule": round(self.lambda_rule, 4),
            **anomalies,
        }

    def __call__(
        self,
        prompts: list[Any],
        completions: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        """Score a batch of completions and record monitoring metrics.

        Extract user prompt text and assistant completion text.
        Calculate rule-based reward and check failures for each pair.
        Record word counts, opening prefixes, first-person usage, and motif mentions.
        If use_cmdp_primal_dual is enabled, scale negative penalties by lambda_rule
        and update lambda_rule toward target_violation_rate (0.04).
        If call count reaches the log interval, save summaries and samples.
        Return a list of scalar rewards.
        """
        from rewards.checks import check_coach_voice

        batch_rewards: list[float] = []
        batch_violations = 0
        motif_keywords = ("fork", "pin", "skewer", "checkmate", "mate", "discovered")

        for prompt_item, completion_item in zip(prompts, completions):
            prompt_str = _extract_content(prompt_item, role_target="user")
            completion_str = _extract_content(
                completion_item, role_target="assistant"
            )
            cleaned_text = clean_completion_text(completion_str)
            user_msg = _extract_user_message(prompt_str)

            words = [w.strip() for w in cleaned_text.split() if w.strip()]
            word_count = len(words)
            opening_prefix = " ".join(words[:4]).lower() if words else ""
            has_first_person = not check_coach_voice(cleaned_text).passed
            text_lower = cleaned_text.lower()

            try:
                result = score_completion(
                    prompt_str,
                    completion_str,
                    reward_version=self.reward_version,
                )
                reward_val = float(result.reward)
                passed_all_hard = bool(result.passed_all_hard)
                failed_checks = [
                    k for k, v in result.hard_checks.items() if not v.passed
                ]
                has_motif_or_best = (
                    result.soft_scores.get("better_move", 0.0) >= 1.0
                    or any(m in text_lower for m in motif_keywords)
                )
            except Exception as exc:
                logger.warning("Scoring failed in MonitoredGRPOReward: %s", exc)
                reward_val = -1.0
                passed_all_hard = False
                failed_checks = ["scoring_exception"]
                has_motif_or_best = False

            if not passed_all_hard:
                batch_violations += 1
                if self.use_cmdp_primal_dual and reward_val < 0.0:
                    reward_val = round(max(-3.0, reward_val * self.lambda_rule), 4)

            sample_record = {
                "prompt": user_msg,
                "completion": cleaned_text,
                "reward": reward_val,
                "passed_all_hard": passed_all_hard,
                "failed_checks": failed_checks,
                "word_count": word_count,
                "opening_prefix": opening_prefix,
                "has_first_person": has_first_person,
                "has_motif_or_best_move": has_motif_or_best,
            }

            self.current_window.append(sample_record)
            batch_rewards.append(reward_val)

        if self.use_cmdp_primal_dual and len(prompts) > 0:
            batch_violation_rate = batch_violations / len(prompts)
            updated_lambda = self.lambda_rule + self.eta_lambda * (
                batch_violation_rate - self.target_violation_rate
            )
            self.lambda_rule = max(0.2, min(3.0, updated_lambda))

        self.call_count += 1

        if self.call_count % self.log_interval_batches == 0:
            summary = self.summarize_window()
            self.step_summaries.append(summary)
            sampled = sample_outputs_for_inspection(
                self.current_window,
                count=self.sample_size,
                seed=self.seed + self.call_count,
            )
            self.sampled_outputs[self.call_count] = sampled
            self.current_window.clear()

        return batch_rewards


MonitoredGRPOReward.__name__ = "grpo_rule_reward"


