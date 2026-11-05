"""Baseline explanation generator for the Second evaluation suite.

This module generates and stores baseline explanations for the frozen test set.
It supports multiple backends: transformers, openai, gemini, and rule-baseline.
It resumes interrupted generation runs without duplicate work.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import requests

from core.python.prompt import SYSTEM_MESSAGE
from eval.judge import get_gemini_api_key
from eval.metrics import clean_completion_text, ensure_prompt_text
from rewards.api import parse_prompt

LOGGER = logging.getLogger(__name__)

DEFAULT_BASELINES_DIR = Path("eval/results/baselines")
DEFAULT_TESTSET_PATH = Path("eval/testset-1.0.jsonl")
SYSTEM_PROMPT = SYSTEM_MESSAGE


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a newline-delimited JSONL file.

    If file does not exist, return an empty list.
    Return list of parsed dictionaries.
    """
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


class RuleBaselineGenerator:
    """Offline rule-based generator for smoke tests and simple baselines."""

    def __init__(self, name: str = "rule-baseline") -> None:
        """Initialize generator name."""
        self.name = name

    def generate(self, prompt_text: str, item: dict[str, Any]) -> str:
        """Generate a factual explanation directly from prompt metadata.

        Use prompt details to assemble concise, legal, and grounded sentences.
        Return the explanation string.
        """
        try:
            info = parse_prompt(prompt_text)
            task = info.task
            level = info.level
            detail = info.detail
            verdict = info.verdict
            move_san = info.move_played_san
            best_move = info.best_move_san
            facts = info.facts
        except Exception:
            task = str(item.get("task", "explain_move"))
            level = str(item.get("level", "club"))
            detail = str(item.get("detail", "short"))
            verdict = str(item.get("verdict")) if item.get("verdict") else None
            move_san = str(item.get("move_played", ""))
            best_move = ""
            facts = [str(f) for f in item.get("facts", [])]

        sentences: list[str] = []
        if task == "explain_move" and move_san:
            if verdict:
                v_low = verdict.lower()
                if v_low in ("good", "best", "only move"):
                    article = "an" if v_low.startswith("o") else "a"
                    label = v_low if v_low.endswith("move") else f"{v_low} move"
                    sentences.append(f"The move {move_san} is {article} {label}.")
                else:
                    article = "an" if v_low.startswith("i") else "a"
                    sentences.append(f"The move {move_san} is {article} {v_low}.")
            else:
                sentences.append(f"The played move is {move_san}.")

            if facts:
                sentences.append(facts[0])

            if best_move and verdict and verdict.lower() in ("inaccuracy", "mistake", "blunder"):
                sentences.append(f"A better move was {best_move} to improve the position.")
            elif len(facts) > 1:
                sentences.append(facts[1])
            sentences.append("Active pieces and king safety decide the next phase.")
        else:
            sentences.append("The position requires careful planning and piece coordination.")
            if facts:
                sentences.append(facts[0])
            if best_move:
                sentences.append(f"The recommended move is {best_move} to fight for the initiative.")
            sentences.append("Both sides must watch for tactical threats before advancing.")

        if detail == "long":
            for f in facts[2:4]:
                sentences.append(f)
            sentences.append(
                "Developing remaining pieces and controlling key central squares "
                "creates lasting pressure across the board. "
                "Before committing to an exchange, check the opponent's best reply "
                "and verify that every key square remains defended."
            )

        if level == "beginner":
            # Keep sentences short and simple while meeting target length
            if detail == "short":
                return " ".join(sentences[:4])
            return " ".join(sentences)

        return " ".join(sentences)


class GeminiBaselineGenerator:
    """Gemini API baseline generator with greedy decoding."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        api_key: str | None = None,
        max_retries: int = 5,
    ) -> None:
        """Initialize Gemini baseline parameters."""
        self.model = model
        self.api_key = api_key or get_gemini_api_key(required=True)
        self.max_retries = max_retries
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def generate(self, prompt_text: str, item: dict[str, Any]) -> str:
        """Generate baseline completion using Gemini REST API.

        Send system instruction and user message with greedy decoding.
        Return the generated text string.
        """
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key or "",
        }
        body = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
            },
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.url, headers=headers, json=body, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return str(parts[0].get("text", "")).strip()
                    return ""

                if resp.status_code in (429, 500, 502, 503, 504):
                    delay = 1.5 * (2 ** attempt)
                    LOGGER.warning("Gemini rate limit or error (%s). Waiting %.1f s.", resp.status_code, delay)
                    time.sleep(delay)
                    continue

                raise RuntimeError(f"Gemini call failed with code {resp.status_code}: {resp.text}")
            except Exception as exc:
                last_error = exc
                delay = 1.5 * (2 ** attempt)
                time.sleep(delay)

        raise RuntimeError(f"Gemini baseline generation failed: {last_error}")


class OpenAIBaselineGenerator:
    """OpenAI-compatible HTTP endpoint baseline generator."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:1234/v1",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        """Initialize OpenAI API client endpoint and model."""
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.max_tokens = max_tokens

    def generate(self, prompt_text: str, item: dict[str, Any]) -> str:
        """Generate baseline completion via OpenAI chat completions endpoint.

        Send system message and user message with greedy decoding.
        Return completion string.
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI endpoint error ({resp.status_code}): {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()


class TransformersBaselineGenerator:
    """Hugging Face transformers baseline generator."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        """Initialize transformers model and tokenizer.

        If model directory contains adapter_config.json, load PEFT adapter weights.
        Otherwise load full model weights directly.
        Import transformers and torch lazily.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Packages transformers and torch are required for the transformers backend. "
                "Install them or select another backend."
            ) from exc

        self.model_path = model_path
        self.max_tokens = max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.is_file():
            try:
                import peft
            except ImportError as exc:
                raise ImportError(
                    "Package peft is required to load adapter checkpoints. Install peft."
                ) from exc

            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_model_name_or_path = adapter_cfg.get("base_model_name_or_path")
            if not base_model_name_or_path:
                raise ValueError(
                    f"adapter_config.json in {model_path} does not specify base_model_name_or_path"
                )

            LOGGER.info("Loading tokenizer from %s (fallback to %s)", model_path, base_model_name_or_path)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)

            LOGGER.info("Loading base model from %s for adapter %s on %s", base_model_name_or_path, model_path, self.device)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            self.model = peft.PeftModel.from_pretrained(base_model, model_path)
        else:
            LOGGER.info("Loading tokenizer from %s", model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            LOGGER.info("Loading model from %s on %s", model_path, self.device)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt_text: str, item: dict[str, Any]) -> str:
        """Generate greedy completion with Hugging Face model.

        Format prompt into chat messages and decode tokens.
        Disable thinking mode when the tokenizer supports it.
        Return the generated completion text.
        """
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            input_text = f"System: {SYSTEM_PROMPT}\nUser: {prompt_text}\nAssistant:"

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return clean_completion_text(decoded)


def create_baseline_generator(
    backend: str,
    model_id: str | None = None,
    api_base: str = "http://localhost:1234/v1",
) -> Any:
    """Create a baseline generator for the specified backend.

    Allowed backends are rule-baseline, gemini, openai, and transformers.
    Return generator instance.
    """
    b = backend.strip().lower()
    if b == "rule-baseline":
        return RuleBaselineGenerator(name=model_id or "rule-baseline")
    if b == "gemini":
        return GeminiBaselineGenerator(model=model_id or "gemini-3.5-flash-lite")
    if b == "openai":
        return OpenAIBaselineGenerator(
            model=model_id or "base-qwen-2b",
            api_base=api_base,
        )
    if b == "transformers":
        if not model_id:
            raise ValueError("model_id or model_path is required for transformers backend.")
        return TransformersBaselineGenerator(model_path=model_id)

    raise ValueError(f"Unsupported baseline backend: {backend}")


def generate_baseline(
    baseline_name: str,
    backend: str,
    testset_path: Path | str = DEFAULT_TESTSET_PATH,
    output_dir: Path | str = DEFAULT_BASELINES_DIR,
    model_id: str | None = None,
    api_base: str = "http://localhost:1234/v1",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Generate baseline outputs for the test set and save to JSONL.

    Resume execution if output file contains existing item identifiers.
    Return all generated records.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{baseline_name}.jsonl"

    testset_items = load_jsonl(testset_path)
    if not testset_items:
        LOGGER.warning("Test set file is empty or not found: %s", testset_path)
        return []

    if limit is not None and limit > 0:
        testset_items = testset_items[:limit]

    # Load existing items to resume safely
    existing_records = load_jsonl(out_file)
    existing_ids = {str(r.get("id")) for r in existing_records if r.get("id")}
    LOGGER.info(
        "Baseline %s: %d existing items loaded from %s",
        baseline_name,
        len(existing_ids),
        out_file,
    )

    all_records = list(existing_records)
    pending_items = [it for it in testset_items if str(it.get("id")) not in existing_ids]

    if not pending_items:
        LOGGER.info("All %d baseline items already generated.", len(testset_items))
        return all_records

    generator = create_baseline_generator(backend=backend, model_id=model_id, api_base=api_base)

    LOGGER.info("Generating %d pending baseline items for %s...", len(pending_items), baseline_name)
    with open(out_file, "a", encoding="utf-8") as f:
        for idx, item in enumerate(pending_items, start=1):
            item_id = str(item.get("id", f"item_{idx}"))
            prompt_str = ensure_prompt_text(item)

            try:
                completion = generator.generate(prompt_str, item)
            except Exception as exc:
                LOGGER.error("Failed to generate baseline for %s: %s", item_id, exc)
                completion = ""

            record = {
                "id": item_id,
                "prompt": prompt_str,
                "user_message": prompt_str,
                "completion": completion,
                "output": completion,
                "model": baseline_name,
                "task": item.get("task"),
                "level": item.get("level"),
                "detail": item.get("detail"),
                "fen": item.get("fen"),
                "verdict": item.get("verdict"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            all_records.append(record)

    return all_records


def load_baseline(
    baseline_name_or_path: str | Path,
    baselines_dir: Path | str = DEFAULT_BASELINES_DIR,
) -> list[dict[str, Any]]:
    """Load baseline records by name or file path.

    Return list of records or empty list if not found.
    """
    p = Path(baseline_name_or_path)
    if p.is_file():
        return load_jsonl(p)

    target = Path(baselines_dir) / f"{baseline_name_or_path}.jsonl"
    if target.is_file():
        return load_jsonl(target)

    return []


def main() -> None:
    """CLI entry point for baseline generation."""
    parser = argparse.ArgumentParser(description="Generate baseline explanations for evaluation.")
    parser.add_argument("--baseline-name", type=str, required=True, help="Baseline name identifier")
    parser.add_argument(
        "--backend",
        type=str,
        default="rule-baseline",
        choices=["rule-baseline", "gemini", "openai", "transformers"],
        help="Inference backend",
    )
    parser.add_argument("--testset", type=str, default=str(DEFAULT_TESTSET_PATH), help="Test set JSONL path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_BASELINES_DIR), help="Output directory")
    parser.add_argument("--model-id", type=str, default=None, help="Model ID or checkpoint path")
    parser.add_argument("--api-base", type=str, default="http://localhost:1234/v1", help="API base URL")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    args = parser.parse_args()

    print(f"Generating baseline: {args.baseline_name} (backend={args.backend})")
    records = generate_baseline(
        baseline_name=args.baseline_name,
        backend=args.backend,
        testset_path=args.testset,
        output_dir=args.output_dir,
        model_id=args.model_id,
        api_base=args.api_base,
        limit=args.limit,
    )
    print(f"Done. Total baseline items: {len(records)}")


if __name__ == "__main__":
    main()
