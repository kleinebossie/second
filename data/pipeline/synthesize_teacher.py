"""Teacher synthesis module for generating high-quality chess explanations.

This module formats prompts for teacher models.
It extracts structured cognitive traces from completions.
It splits teacher completions into thinking and non-thinking student training records.
It generates teacher completions using Qwen/Qwen3.5-27B or mock execution.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewards.parser import parse_moves, parse_squares

LOGGER = logging.getLogger(__name__)

DEFAULT_TEACHER_MODEL: str = "Qwen/Qwen3.5-27B"
DEFAULT_PROMPTS_PATH: Path = Path("data/processed/teacher_prompts.jsonl")
DEFAULT_OUTPUT_PATH: Path = Path("data/processed/teacher_completions.jsonl")

TEACHER_SYSTEM_PROMPT: str = (
    "You are an expert chess coach. "
    "You explain chess positions and moves clearly for students. "
    "Follow these coaching rules strictly:\n"
    "1. For explain_move tasks: State the move verdict clearly in your opening sentence. "
    "Name the primary threat, hanging piece, or tactical motif from the Facts block.\n"
    "2. If the move is an Inaccuracy, Mistake, or Blunder: Use feigned discovery. "
    "First explain why the move looks tempting. "
    "Then show the exact refutation line using the best reply from the Facts. "
    "Always state the best move and explain what problem it solves.\n"
    "3. For explain_position tasks: Summarize the position evaluation and recommend the best move or plan. "
    "Explain key candidate moves and tactical or positional factors from the Facts.\n"
    "4. Never mention any move, square, piece, motif, or opening name outside the provided position and Facts. "
    "Never use first-person pronouns like 'I', 'me', or 'my'. "
    "Write in the second person ('you') or third person ('White', 'Black').\n"
    "5. Target length:\n"
    "   - For detail: short, write 2 to 4 sentences (40 to 70 words).\n"
    "   - For detail: long, write 4 to 7 sentences in 2 paragraphs (100 to 160 words).\n"
    "6. Output structure:\n"
    "   First output a structured cognitive trace in <think> tags:\n"
    "   <think>\n"
    "   verdict_or_eval: <Verdict or evaluation> (<brief reason>)\n"
    "   main_fact: <primary fact or tactic from Facts>\n"
    "   best_move: <best move or counterfactual plan>\n"
    "   coach_plan: <brief explanation plan>\n"
    "   </think>\n"
    "   Then output the final explanation prose immediately after </think>."
)

THINK_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def build_teacher_messages(user_message: str) -> list[dict[str, str]]:
    """Build chat messages for the teacher model prompt.

    Include the teacher system prompt and the student user message.
    Return list of message dictionaries.
    """
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def parse_teacher_completion(raw_text: str) -> dict[str, str]:
    """Parse raw completion text into structured trace and explanation.

    If <think> tags exist, extract the trace inside them.
    If </think> exists without an opening tag, extract preceding text as trace.
    Strip thinking tags from the final explanation prose.
    Return dictionary with trace and explanation fields.
    """
    if "</think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        trace = parts[0].replace("<think>", "").strip()
        explanation = parts[1].strip()
        return {
            "trace": trace,
            "explanation": explanation,
        }
    match = THINK_REGEX.search(raw_text)
    if match:
        trace = match.group(1).strip()
        explanation = raw_text[match.end():].strip()
    else:
        trace = ""
        explanation = raw_text.strip()

    return {
        "trace": trace,
        "explanation": explanation,
    }


def emit_student_records(
    item: dict[str, Any],
    parsed: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Emit non-thinking and thinking training records from parsed output.

    Build non-thinking record with empty think tags.
    Build thinking record with structured trace inside think tags.
    Return tuple of (non_thinking_record, thinking_record).
    """
    base = dict(item)
    user_msg = base.get("user_message", "")
    trace = parsed.get("trace", "").strip()
    explanation = parsed.get("explanation", "").strip()

    # 1. Non-thinking variant
    non_thinking_assistant = f"<think>\n\n</think>\n\n{explanation}"
    non_thinking_prompt = (
        f"<|im_start|>system\nYou are Second, a chess assistant. "
        f"You explain chess positions and moves clearly using the provided facts.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    rec_non_thinking = {
        **base,
        "comment": explanation,
        "explanation": explanation,
        "assistant_response": non_thinking_assistant,
        "chat_prompt": non_thinking_prompt,
        "thinking_mode": "off",
    }

    # 2. Thinking variant
    thinking_assistant = f"<think>\n{trace}\n</think>\n\n{explanation}" if trace else non_thinking_assistant
    thinking_prompt = (
        f"<|im_start|>system\nYou are Second, a chess assistant. "
        f"You explain chess positions and moves clearly using the provided facts.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    rec_thinking = {
        **base,
        "comment": thinking_assistant,
        "explanation": explanation,
        "trace": trace,
        "assistant_response": thinking_assistant,
        "chat_prompt": thinking_prompt,
        "thinking_mode": "standard",
    }

    return rec_non_thinking, rec_thinking


def _extract_mock_fact_ref(fact_str: str) -> str:
    """Extract a safe board reference from a fact string for mock generation.

    If a square exists, reference the square with unambiguous context.
    If a move exists, reference the move.
    Otherwise return a safe central reference.
    """
    sqs = parse_squares(fact_str)
    if sqs:
        return f"on the {sqs[0]} square"
    mvs = parse_moves(fact_str)
    if mvs:
        return f"involving {mvs[0]}"
    words = [
        w for w in re.findall(r"\b[a-zA-Z]{4,}\b", fact_str.lower())
        if w not in ("phase", "material", "after", "white", "black", "best", "reply")
    ]
    if len(words) >= 2:
        return f"with {words[0]} and {words[1]}"
    return "in the center"


def generate_mock_teacher_completion(item: dict[str, Any]) -> str:
    """Generate deterministic mock teacher completion for testing and offline runs.

    Synthesize coaching prose matching the item task, verdict, facts, and target length.
    Include structured <think> trace.
    Return formatted mock completion string.
    """
    task = item.get("task", "explain_move")
    detail = item.get("detail", "short")
    facts = item.get("facts", [])
    user_msg = item.get("user_message", "")
    verdict = item.get("verdict") or "Good"

    best_san = "Nf3"
    m_best = re.search(r"Best move:\s*([^\n.]+)", user_msg)
    if m_best:
        best_san = m_best.group(1).strip().rstrip("+#!?")

    played_san = "this move"
    m_played = re.search(r"Move played:\s*([^\n.]+)", user_msg)
    if m_played:
        played_san = m_played.group(1).strip().rstrip("+#!?")

    reply_san = ""
    for f in facts:
        m_rep = re.search(r"best reply is ([A-Za-z0-9+#=]+)", f)
        if m_rep:
            reply_san = m_rep.group(1).strip().rstrip("+#!?")
            break

    non_generic = [
        f for f in facts
        if not f.strip().startswith(("Phase:", "Material:"))
        and "in material" not in f.lower()
    ]
    hint1 = non_generic[0] if non_generic else "Piece coordination remains essential."

    ref1 = _extract_mock_fact_ref(non_generic[0]) if len(non_generic) > 0 else "on the central squares"
    ref2 = _extract_mock_fact_ref(non_generic[1]) if len(non_generic) > 1 else ""

    is_bad = verdict in ("Inaccuracy", "Mistake", "Blunder")

    if task == "explain_position":
        trace = (
            f"verdict_or_eval: Balanced position (level chances across the board)\n"
            f"main_fact: {hint1}\n"
            f"best_move: {best_san}\n"
            f"coach_plan: Outline position balance, discuss piece coordination, recommend best continuation"
        )
        extra_clause = f" You should also remain alert {ref2}." if ref2 else ""
        if detail == "long":
            explanation = (
                f"In this position, both sides hold balanced chances with active piece play across the board. "
                f"Your immediate priority is maintaining solid coordination and safeguarding vulnerable areas {ref1}. "
                f"Watch opponent piece maneuvers carefully.{extra_clause}\n\n"
                f"The strongest continuation here is {best_san}, which improves your tactical placement and reinforces your central structure. "
                f"Keep your king well protected while preparing coordinated piece play. "
                f"Patient maneuvering will allow you to generate active pressure without creating structural weaknesses in your camp. "
                f"Maintain flexible options and coordinate your army step by step to control the flow of the game."
            )
        else:
            explanation = (
                f"In this position, the game offers balanced prospects with active piece play on both sides. "
                f"You should prioritize piece coordination and maintain solid control {ref1}.{extra_clause} "
                f"The strongest continuation is {best_san}, which solidifies your structure and keeps your pieces harmonious across the board."
            )
    elif is_bad:
        trace = (
            f"verdict_or_eval: {verdict} ({played_san} concedes tactical initiative)\n"
            f"main_fact: {hint1}\n"
            f"best_move: {best_san}\n"
            f"coach_plan: Use feigned discovery, explain why played move is tempting, reveal refutation, state best move"
        )
        refutation = f"opponent replies with {reply_san}" if reply_san else f"opponent finds strong pressure {ref1}"
        if detail == "long":
            explanation = (
                f"Playing {played_san} is an unfortunate {verdict.lower()} that concedes a serious tactical problem. "
                f"At first glance, this choice looks tempting because it seems active and challenges territory. "
                f"However, the move overlooks tactical danger because the {refutation}, seizing the initiative directly {ref1}.\n\n"
                f"Instead of {played_san}, you should have played {best_san}. "
                f"Choosing {best_san} solves the immediate defensive problem, secures your position, and maintains equal chances. "
                f"Always calculate forcing opponent responses before committing your pieces forward into enemy lines. "
                f"Taking time to secure your position prevents sudden tactics, protects your camp, and preserves your winning prospects."
            )
        else:
            explanation = (
                f"Playing {played_san} is a {verdict.lower()} that overlooks an opponent threat. "
                f"Although this move appears tempting to challenge territory, the {refutation}. "
                f"Instead of {played_san}, you should play {best_san} to resolve tactical tension safely and keep your position coordinated."
            )
    else:
        # Good/Best/Only
        trace = (
            f"verdict_or_eval: {verdict} ({played_san} coordinates pieces cleanly)\n"
            f"main_fact: {hint1}\n"
            f"best_move: {best_san}\n"
            f"coach_plan: Confirm good move, explain positional benefits around key squares, recommend solid plan"
        )
        extra_clause = f" It also accounts for defense {ref2} cleanly." if ref2 else " It keeps your central pawns well protected."
        if detail == "long":
            explanation = (
                f"Playing {played_san} is a {verdict.lower()} move that improves your piece activity and preserves the initiative. "
                f"This principled decision solidifies your control {ref1} and supports harmonious piece development.{extra_clause} "
                f"By keeping your camp safe, you limit opponent counterplay and maintain steady pressure across the board.\n\n"
                f"Continuing with energetic play such as {best_san} remains the most logical follow-up. "
                f"Maintain watchful defense against sudden threats while building your positional advantages methodically. "
                f"Coordinated pieces will help you dictate terms as the position develops. "
                f"Stay focused on gradual expansion and preserve your harmonious piece coordination throughout the game."
            )
        else:
            explanation = (
                f"Playing {played_san} is a {verdict.lower()} move that improves your piece coordination. "
                f"This sensible choice controls critical squares {ref1} and keeps your pieces harmonious.{extra_clause} "
                f"Continuing with {best_san} remains a solid plan to maintain your advantage and restrict opponent counterplay effectively."
            )

    return f"<think>\n{trace}\n</think>\n\n{explanation}"


def run_teacher_synthesis(
    prompts_path: Path | str = DEFAULT_PROMPTS_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_TEACHER_MODEL,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    samples_per_prompt: int = 1,
    mock: bool = False,
    limit: int | None = None,
    load_in_4bit: bool = False,
) -> Path:
    """Run teacher completion generation over input prompt records.

    If mock is True, generate valid mock completions without loading neural network weights.
    If mock is False, load the teacher model with HuggingFace transformers.
    If load_in_4bit is True, configure 4-bit BitsAndBytes quantization.
    Save outputs to target JSONL file with incremental checkpointing.
    Return path to output completions file.
    """
    p_path = Path(prompts_path)
    if not p_path.is_file() and (PROJECT_ROOT / p_path).is_file():
        p_path = PROJECT_ROOT / p_path

    out_file = Path(output_path)
    if not out_file.is_absolute() and not out_file.parent.exists():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    prompts: list[dict[str, Any]] = []
    with open(p_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
                if limit and len(prompts) >= limit:
                    break

    LOGGER.info("Loaded %d prompts for teacher synthesis (mock=%s)", len(prompts), mock)

    # Check already processed prompts to enable resuming
    existing_counts: Counter[str] = Counter()
    if out_file.is_file():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    pid = rec.get("id") or rec.get("prompt_id")
                    if pid:
                        existing_counts[pid] += 1
        LOGGER.info("Resuming synthesis: %d prompt completions already recorded", sum(existing_counts.values()))

    remaining_prompts = [
        p for p in prompts
        if existing_counts[p.get("id", "")] < samples_per_prompt
    ]

    if mock:
        with open(out_file, "a", encoding="utf-8") as f:
            for item in remaining_prompts:
                p_id = item.get("id", "")
                already_done = existing_counts[p_id]
                for s_idx in range(already_done, samples_per_prompt):
                    comp_text = generate_mock_teacher_completion(item)
                    out_entry = {
                        "id": p_id,
                        "sample_index": s_idx,
                        "model": "mock-teacher",
                        "completion": comp_text,
                    }
                    f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                    f.flush()
                    existing_counts[p_id] += 1
        LOGGER.info("Mock synthesis complete for %d prompts.", len(remaining_prompts))
        return out_file

    # Real model execution on GPU
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("Loading model %s (4-bit=%s)...", model_name, load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
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
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with open(out_file, "a", encoding="utf-8") as f:
        for s_idx in range(samples_per_prompt):
            prompts_for_sample = [
                p for p in remaining_prompts
                if existing_counts[p.get("id", "")] <= s_idx
            ]
            if not prompts_for_sample:
                continue

            LOGGER.info(
                "Generating sample %d/%d for %d prompts...",
                s_idx + 1,
                samples_per_prompt,
                len(prompts_for_sample),
            )

            for start_idx in range(0, len(prompts_for_sample), batch_size):
                batch = prompts_for_sample[start_idx : start_idx + batch_size]
                batch_messages = [
                    build_teacher_messages(item.get("user_message", ""))
                    for item in batch
                ]
                batch_prompts = [
                    tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    for msgs in batch_messages
                ]

                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=pad_id,
                    )

                for item, out_ids, in_ids in zip(batch, outputs, inputs["input_ids"]):
                    gen_ids = out_ids[len(in_ids):]
                    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    out_entry = {
                        "id": item.get("id"),
                        "sample_index": s_idx,
                        "model": model_name,
                        "completion": completion,
                    }
                    f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                    f.flush()
                    existing_counts[item.get("id", "")] += 1

    LOGGER.info("Teacher synthesis complete. Saved completions to %s", out_file)
    return out_file


def main() -> None:
    """CLI entry point for teacher synthesis."""
    parser = argparse.ArgumentParser(
        description="Run teacher completion generation for Stage T2 knowledge distillation."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=str(DEFAULT_PROMPTS_PATH),
        help="Path to input teacher prompts JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output completions JSONL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TEACHER_MODEL,
        help="HuggingFace model identifier for teacher model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for model inference (default 4 for 24GB L4 GPU)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per completion",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Number of candidate completions to sample per prompt",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit bitsandbytes quantization for model weights",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock generator for offline dry runs and testing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of prompts to process",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_teacher_synthesis(
        prompts_path=args.prompts,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        samples_per_prompt=args.samples_per_prompt,
        mock=args.mock,
        limit=args.limit,
        load_in_4bit=args.load_in_4bit,
    )


if __name__ == "__main__":
    main()
