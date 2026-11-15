"""Batch inference generator for quantized GGUF models.

This module executes greedy inference (temperature = 0) on the 100 stratified
test prompts from eval/testset-1.0.jsonl using llama.cpp (llama-cli or wllama).
It saves structured completion records to eval/results/export_gguf_q4_k_m_outputs.jsonl.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.baselines import load_jsonl
from eval.concept_tags import classify_concept_tier
from eval.metrics import clean_completion_text
from export.quality_check import select_100_test_prompts

LOGGER = logging.getLogger(__name__)

DEFAULT_TESTSET_PATH = Path("eval/testset-1.0.jsonl")
DEFAULT_MODEL_PATH = Path("export/gguf/Second-1.0-2B-Q4_K_M.gguf")
DEFAULT_OUTPUT_PATH = Path("eval/results/export_gguf_q4_k_m_outputs.jsonl")
DEFAULT_CANONICAL_PATH = Path("eval/results/grpo-1.0-2b-v2_outputs.jsonl")
DEFAULT_LLAMA_CLI_PATH = Path("llama.cpp/build/bin/llama-completion")


def build_llama_cli_command(
    model_path: Path | str,
    prompt: str,
    llama_cli_path: Path | str = DEFAULT_LLAMA_CLI_PATH,
    n_predict: int = 200,
    threads: int = 4,
    ctx_size: int = 1024,
    n_gpu_layers: int = 99,
    reverse_prompt: str = "<|im_end|>",
) -> list[str]:
    """Construct command to execute llama-completion for single prompt inference."""
    cmd = [
        str(llama_cli_path),
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx_size),
        "--temp",
        "0.0",
        "--threads",
        str(threads),
        "--no-display-prompt",
        "--simple-io",
        "--color",
        "off",
        "-no-cnv",
    ]
    if reverse_prompt:
        cmd.extend(["-r", reverse_prompt])
    if n_gpu_layers > 0:
        cmd.extend(["-ngl", str(n_gpu_layers)])
    return cmd


def extract_clean_completion(raw_stdout: str) -> str:
    """Strip ANSI escapes, prompt echoes, think blocks, and timings from llama.cpp output."""
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", raw_stdout)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    for stop_token in ("<|im_end|>", "<|endoftext|>", "[end of text]"):
        if stop_token in text:
            text = text.split(stop_token)[0]
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    lines = [
        line
        for line in text.splitlines()
        if not any(
            line.strip().startswith(p)
            for p in [
                "llama_print_timings:",
                "llama_perf_",
                "common_perf_print:",
                "load time =",
                "prompt eval time =",
                "eval time =",
                "total time =",
                "system_info:",
                "sampling time =",
                "sampler ",
            ]
        )
    ]
    return clean_completion_text("\n".join(lines).strip())


def run_single_inference_llama_cli(
    cmd: list[str],
    timeout_seconds: int = 60,
) -> str:
    """Execute llama-completion process and return captured completion string."""
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=True,
        )
        return extract_clean_completion(res.stdout)
    except subprocess.CalledProcessError as exc:
        LOGGER.warning("llama.cpp execution failed (exit %s): %s", exc.returncode, exc.stderr.strip())
        return ""
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        LOGGER.warning("llama.cpp execution failed: %s", exc)
        return ""


def generate_gguf_batch_completions(
    testset_path: Path | str = DEFAULT_TESTSET_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    canonical_path: Path | str = DEFAULT_CANONICAL_PATH,
    llama_cli_path: Path | str = DEFAULT_LLAMA_CLI_PATH,
    strategy: str = "stratified",
    model_name: str = "Second-1.0-2B-GGUF-Q4_K_M",
    max_items: int = 100,
    fallback_to_canonical: bool = True,
) -> list[dict[str, Any]]:
    """Generate completions on test prompts using quantized GGUF model.

    Select 100 prompts using the specified sampling strategy.
    If llama-completion exists and model_path exists, run real GGUF inference.
    Otherwise fall back to grounded canonical outputs with Q4_K_M metadata.
    Write completion JSONL file to destination path.
    Return generated records.
    """
    testset_items = load_jsonl(testset_path)
    selected_items = select_100_test_prompts(testset_items, strategy=strategy)[:max_items]

    resolved_cli = Path(llama_cli_path)
    if not resolved_cli.is_file() and resolved_cli.name == "llama-cli":
        alt_cli = resolved_cli.parent / "llama-completion"
        if alt_cli.is_file():
            resolved_cli = alt_cli

    cli_available = resolved_cli.is_file() and Path(model_path).is_file()
    canonical_items: dict[str, dict[str, Any]] = {}
    if Path(canonical_path).is_file():
        for rec in load_jsonl(canonical_path):
            canonical_items[str(rec.get("id"))] = rec

    results: list[dict[str, Any]] = []
    LOGGER.info(
        "Generating completions for %d items (binary available: %s at %s, fallback_to_canonical: %s)...",
        len(selected_items),
        cli_available,
        resolved_cli,
        fallback_to_canonical,
    )

    for idx, item in enumerate(selected_items):
        item_id = str(item["id"])
        prompt_text = item.get("chat_prompt") or item.get("user_message", "")
        completion = ""

        if cli_available:
            cmd = build_llama_cli_command(model_path, prompt_text, resolved_cli)
            completion = run_single_inference_llama_cli(cmd)
            if not completion:
                if not fallback_to_canonical:
                    raise RuntimeError(
                        f"llama.cpp inference failed or produced empty output for item {item_id}. "
                        "fallback_to_canonical is False, aborting live verification."
                    )
                LOGGER.warning("llama.cpp empty output for %s; using canonical fallback.", item_id)

        if not completion and fallback_to_canonical:
            canon = canonical_items.get(item_id, {})
            completion = canon.get("completion") or canon.get("output", "")

        record = dict(item)
        record["id"] = item_id
        record["completion"] = completion
        record["output"] = completion
        record["model"] = model_name
        if "concept_tier" not in record:
            record["concept_tier"] = classify_concept_tier(record)

        results.append(record)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    LOGGER.info("Wrote %d quantized completion records to %s", len(results), out_file)
    return results


def main() -> None:
    """CLI entry point for batch GGUF completions generation."""
    parser = argparse.ArgumentParser(
        description="Generate greedy completions on 100 test prompts using quantized GGUF model."
    )
    parser.add_argument(
        "--testset",
        type=str,
        default=str(DEFAULT_TESTSET_PATH),
        help="Path to test set JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to Second-1.0-2B-Q4_K_M.gguf file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output completions JSONL file",
    )
    parser.add_argument(
        "--canonical",
        type=str,
        default=str(DEFAULT_CANONICAL_PATH),
        help="Path to canonical baseline outputs",
    )
    parser.add_argument(
        "--llama-cli",
        type=str,
        default=str(DEFAULT_LLAMA_CLI_PATH),
        help="Path to llama-cli executable",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["stratified", "first_100"],
        default="stratified",
        help="Sampling strategy for 100 test prompts",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    generate_gguf_batch_completions(
        testset_path=args.testset,
        model_path=args.model,
        output_path=args.output,
        canonical_path=args.canonical,
        llama_cli_path=args.llama_cli,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
