"""Run teacher selection spike on Google Colab or local GPU.

This script evaluates teacher model candidates on 50 devset items.
It measures hard check pass rate, soft scores, and output length.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline.synthesize_teacher import (
    build_teacher_messages,
    parse_teacher_completion,
)
from rewards.scorer import compute_reward

LOGGER = logging.getLogger(__name__)

DEFAULT_CANDIDATES: list[str] = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.6-27B",
]


def load_devset_items(path: Path, count: int = 50) -> list[dict[str, Any]]:
    """Load items from the stratified development set.

    If count is zero or negative, return an empty list.
    If the relative path does not exist, check the project root.
    Read JSONL lines until reaching count.
    Return list of item dictionaries.
    """
    if count <= 0:
        return []
    target_path = Path(path)
    if not target_path.is_file() and (PROJECT_ROOT / target_path).is_file():
        target_path = PROJECT_ROOT / target_path
    items: list[dict[str, Any]] = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if len(items) >= count:
                break
    return items


def generate_mock_completions(items: list[dict[str, Any]]) -> list[str]:
    """Generate mock teacher responses for unit testing and dry runs.

    Produce valid coaching prose with structured thinking tags.
    Return list of mock completion strings.
    """
    completions: list[str] = []
    for item in items:
        task = item.get("task", "explain_move")
        detail = item.get("detail", "short")
        facts = item.get("facts", [])
        fact_hint = facts[0] if facts else "Material is equal."
        user_msg = item.get("user_message", "")

        best_san = item.get("best_move_san")
        if not best_san:
            match = re.search(r"Best move:\s*([^\n.]+)", user_msg)
            best_san = match.group(1).strip() if match else "e4"

        if task == "explain_position":
            wc_pct = 50
            m_wc = re.search(r"winning chances:\s*(\d+)%", user_msg)
            if m_wc:
                wc_pct = int(m_wc.group(1))

            trace = (
                f"verdict_or_eval: Even ({wc_pct}% winning chances)\n"
                f"main_fact: {fact_hint}\n"
                f"best_move: {best_san}\n"
                f"coach_plan: Explain candidate moves and positional coordination"
            )
            if detail == "long":
                explanation = (
                    f"In this position, the game remains balanced with {wc_pct}% winning chances for your side. "
                    f"According to the position facts, {fact_hint} "
                    f"Maintaining piece harmony across the central files is essential before committing to pawn breaks. "
                    f"Both sides have active prospects, so careful piece placement will determine who seizes the initiative.\n\n"
                    f"The strongest continuation is {best_san}, which coordinates your pieces and maintains pressure across key squares. "
                    f"Prioritize active piece placement, central control, and king safety before launching direct operations on the board. "
                    f"This constructive plan improves your coordination while restricting opponent counterplay in the coming phase."
                )
            else:
                explanation = (
                    f"In this position, the game remains balanced with {wc_pct}% winning chances for your side. "
                    f"According to the position facts, {fact_hint} "
                    f"The strongest continuation is {best_san}, which coordinates your pieces, improves central control, and maintains pressure across key squares."
                )
        else:
            verdict = item.get("verdict") or "Good"
            move_san = item.get("move_san") or "this move"

            trace = (
                f"verdict_or_eval: {verdict} ({move_san})\n"
                f"main_fact: {fact_hint}\n"
                f"best_move: {best_san}\n"
                f"coach_plan: State verdict and explain refutation"
            )
            is_bad = verdict in ("Inaccuracy", "Mistake", "Blunder")
            if detail == "long":
                if is_bad:
                    explanation = (
                        f"{move_san} is classified as {verdict} in this position. "
                        f"The move affects the position directly and tactical concessions arise quickly for your side. "
                        f"According to the position facts, {fact_hint} "
                        f"Leaving key squares unguarded gives the opponent immediate initiative and active targets.\n\n"
                        f"A stronger option is {best_san}, which improves active piece play and prevents tactical counterplay. "
                        f"Look to coordinate your remaining pieces, protect vulnerable squares, and secure king safety before launching further operations on the board. "
                        f"Following this plan keeps your structure solid and avoids unnecessary material loss."
                    )
                else:
                    explanation = (
                        f"{move_san} is classified as {verdict} in this position. "
                        f"The move affects the position directly and maintains your strategic advantage across the board. "
                        f"According to the position facts, {fact_hint} "
                        f"Accurate moves like this build lasting pressure and restrict the opponent's active options.\n\n"
                        f"Playing {move_san} keeps active pressure on the board and supports your central structure. "
                        f"Follow up with solid piece coordination to consolidate your space, improve rooks, and maintain king safety throughout the next phase."
                    )
            else:
                if is_bad:
                    explanation = (
                        f"{move_san} is classified as {verdict} in this position. "
                        f"The move affects the position directly and creates tactical problems. "
                        f"According to the position facts, {fact_hint} "
                        f"A stronger option is {best_san}, which improves active piece play and prevents opponent counterplay."
                    )
                else:
                    explanation = (
                        f"{move_san} is classified as {verdict} in this position. "
                        f"The move affects the position directly and supports your plan. "
                        f"According to the position facts, {fact_hint} "
                        f"Playing {move_san} maintains a solid position, good central control, and active piece play for your side."
                    )

        mock = f"<think>\n{trace}\n</think>\n\n{explanation}"
        completions.append(mock)
    return completions


def generate_hf_completions(
    items: list[dict[str, Any]],
    model_name: str,
    load_in_4bit: bool = False,
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate completions using HuggingFace Transformers pipeline.

    Load model with bfloat16 or 4-bit bitsandbytes quantization.
    Generate completions for each prompt.
    Return list of completion strings.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("Loading model %s (4-bit=%s)...", model_name, load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    completions: list[str] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for idx, item in enumerate(items):
        user_msg = item.get("user_message", "")
        messages = build_teacher_messages(user_msg)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(response)
        LOGGER.info("Generated sample %d / %d", idx + 1, len(items))

    return completions


def evaluate_completions(
    items: list[dict[str, Any]],
    completions: list[str],
) -> dict[str, Any]:
    """Score model completions against Stage T1 checks and soft rubrics.

    Return dictionary with aggregate metrics and per-item scores.
    """
    total = len(items)
    if total == 0:
        return {
            "total": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "mean_reward": 0.0,
            "mean_word_count": 0.0,
            "in_band_rate": 0.0,
            "mean_key_point": 0.0,
            "mean_better_move": 0.0,
            "item_results": [],
        }

    results: list[dict[str, Any]] = []
    pass_count = 0
    total_reward = 0.0
    total_words = 0
    total_key_point = 0.0
    total_better_move = 0.0
    in_band_count = 0

    for item, raw_text in zip(items, completions):
        parsed = parse_teacher_completion(raw_text)
        explanation = parsed["explanation"]
        words = len(explanation.split())
        total_words += words

        detail = item.get("detail", "short")
        if detail == "short" and 35 <= words <= 80:
            in_band_count += 1
        elif detail == "long" and 80 <= words <= 180:
            in_band_count += 1

        best_san = item.get("best_move_san")
        if not best_san:
            match = re.search(r"Best move:\s*([^\n.]+)", item.get("user_message", ""))
            best_san = match.group(1).strip() if match else ""

        reward_res = compute_reward(
            text=explanation,
            fen=item.get("fen", ""),
            level=item.get("level", "club"),
            detail=detail,
            verdict=item.get("verdict"),
            move_uci=item.get("move_uci"),
            best_move_san=best_san,
            facts=item.get("facts"),
            prompt_text=item.get("user_message", ""),
            reward_version="v2",
        )

        passed = reward_res.passed_all_hard
        if passed:
            pass_count += 1
        total_reward += reward_res.reward
        total_key_point += reward_res.soft_scores.get("key_point", 0.0)
        total_better_move += reward_res.soft_scores.get("better_move", 0.0)

        results.append({
            "item_id": item.get("game_id", ""),
            "raw_text": raw_text,
            "trace": parsed["trace"],
            "explanation": explanation,
            "word_count": words,
            "passed_all_hard": passed,
            "reward": reward_res.reward,
            "failed_checks": [
                name for name, check_res in reward_res.hard_checks.items() if not check_res.passed
            ],
            "soft_scores": reward_res.soft_scores,
        })

    pass_rate = pass_count / total
    mean_reward = total_reward / total
    mean_words = total_words / total
    in_band_rate = in_band_count / total
    mean_kp = total_key_point / total
    mean_bm = total_better_move / total

    return {
        "total": total,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "mean_reward": mean_reward,
        "mean_word_count": mean_words,
        "in_band_rate": in_band_rate,
        "mean_key_point": mean_kp,
        "mean_better_move": mean_bm,
        "item_results": results,
    }


def run_spike(
    model_name: str,
    devset_path: Path,
    output_dir: Path,
    count: int = 50,
    load_in_4bit: bool = False,
    is_mock: bool = False,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Run teacher selection evaluation for one model candidate.

    If the output directory is relative, resolve it against the project root.
    Save generations and summary JSON to output directory.
    Return evaluation summary dictionary.
    """
    target_output_dir = Path(output_dir)
    if not target_output_dir.is_absolute():
        target_output_dir = PROJECT_ROOT / target_output_dir
    target_output_dir.mkdir(parents=True, exist_ok=True)
    items = load_devset_items(devset_path, count=count)
    LOGGER.info("Loaded %d items from %s", len(items), devset_path)

    start_time = time.time()
    if is_mock or model_name == "mock":
        completions = generate_mock_completions(items)
    else:
        completions = generate_hf_completions(
            items=items,
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
        )
    elapsed = time.time() - start_time

    eval_summary = evaluate_completions(items, completions)
    eval_summary["model_name"] = model_name
    eval_summary["elapsed_seconds"] = elapsed
    eval_summary["seconds_per_item"] = elapsed / len(items) if items else 0.0

    slug = model_name.replace("/", "_").replace(".", "_")
    summary_path = target_output_dir / f"{slug}_summary.json"
    gen_path = target_output_dir / f"{slug}_generations.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    with open(gen_path, "w", encoding="utf-8") as f:
        for res in eval_summary.get("item_results", []):
            f.write(json.dumps(res) + "\n")

    LOGGER.info("Saved summary to %s", summary_path)
    LOGGER.info("Saved generations to %s", gen_path)
    return eval_summary


def main() -> None:
    """CLI entry point for teacher selection spike."""
    parser = argparse.ArgumentParser(description="Run teacher selection spike on devset.")
    parser.add_argument("--model", type=str, default="mock", help="Candidate model name or HF ID")
    parser.add_argument("--devset", type=str, default="eval/devset-1.0.jsonl", help="Devset path")
    parser.add_argument("--count", type=int, default=50, help="Number of items to evaluate")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit bitsandbytes")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--output-dir", type=str, default="spikes/teacher_selection/results")
    parser.add_argument("--mock", action="store_true", help="Run mock generation for testing")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run_spike(
        model_name=args.model,
        devset_path=Path(args.devset),
        output_dir=Path(args.output_dir),
        count=args.count,
        load_in_4bit=args.load_in_4bit,
        is_mock=args.mock,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n--- Teacher Selection Spike Result ---")
    print(f"Model: {summary['model_name']}")
    print(f"Items Evaluated: {summary['total']}")
    print(f"Hard Checks Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Mean Reward: {summary['mean_reward']:.4f}")
    print(f"In-Band Length Rate: {summary['in_band_rate']:.1%}")
    print(f"Mean Word Count: {summary['mean_word_count']:.1f}")
    print(f"Mean Key Point Score: {summary['mean_key_point']:.4f}")
    print(f"Elapsed Time: {summary['elapsed_seconds']:.1f}s ({summary['seconds_per_item']:.2f}s/item)")


if __name__ == "__main__":
    main()
