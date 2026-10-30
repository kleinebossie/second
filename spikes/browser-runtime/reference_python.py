#!/usr/bin/env python3
"""
Reference Python benchmark for Second browser runtime test.
Runs the exact ~400-token prompt with greedy decoding (do_sample=False).
Saves output JSON for direct browser comparison.

Usage:
    python reference_python.py [--model Qwen/Qwen3.5-0.8B] [--device auto]
"""

import argparse
import json
import os
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run Second reference prompt in Python transformers.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="Hugging Face model ID (e.g. Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-2B, Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target compute device"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompt.json",
        help="Path to prompt JSON file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="reference_output.json",
        help="Path to save reference output JSON"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load prompt definition
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.prompt_file)
    if not os.path.exists(prompt_path):
        print(f"Error: Prompt file not found at {prompt_path}", file=sys.stderr)
        sys.exit(1)

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)

    messages = prompt_data.get("messages", [])
    if not messages:
        messages = [
            {"role": "system", "content": prompt_data["system_message"]},
            {"role": "user", "content": prompt_data["user_message"]}
        ]

    print("=" * 60)
    print("Second - Python Reference Benchmark")
    print(f"Model: {args.model}")
    print(f"Prompt task: {prompt_data.get('task')}, Level: {prompt_data.get('level')}")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    except ImportError as e:
        print(f"Error: Missing required Python package: {e}", file=sys.stderr)
        print("Install dependencies with: pip install torch transformers accelerate", file=sys.stderr)
        sys.exit(1)

    # Resolve compute device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Selected device: {device}")
    if device == "cuda":
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 2. Format Chat Template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Ensure non-thinking mode for Qwen 3.5 if needed
    if "<think>" in formatted_prompt and "</think>" not in formatted_prompt:
        formatted_prompt = formatted_prompt + "\n</think>\n\n"

    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]
    prompt_tokens = input_ids.shape[1]
    print(f"Input prompt tokens: {prompt_tokens}")

    # 3. Load Model
    print("Loading model weights...")
    load_start = time.perf_counter()

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        ).to(device)
    except Exception as e:
        print(f"Warning: Standard loading failed ({e}). Retrying with trust_remote_code=True...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device)

    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f} seconds.")

    input_ids = input_ids.to(device)

    # 4. Generate with greedy decoding
    print("Generating output (greedy decoding: do_sample=False)...")
    first_token_time = None
    gen_start = time.perf_counter()

    class BenchmarkStreamer(TextStreamer):
        def __init__(self, tokenizer):
            super().__init__(tokenizer, skip_prompt=True)
            self.first_token_seen = False
            self.first_token_timestamp = None
            self.pieces = []

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if not self.first_token_seen and text.strip():
                self.first_token_seen = True
                self.first_token_timestamp = time.perf_counter()
            self.pieces.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()

    streamer = BenchmarkStreamer(tokenizer)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_end = time.perf_counter()
    print()  # newline after streamer

    # Compute metrics
    total_tokens_generated = output_ids.shape[1] - input_ids.shape[1]
    ttft_ms = None
    if streamer.first_token_timestamp is not None:
        ttft_ms = round((streamer.first_token_timestamp - gen_start) * 1000, 2)

    total_gen_time = gen_end - gen_start
    effective_gen_time = gen_end - (streamer.first_token_timestamp or gen_start)

    if effective_gen_time > 0 and total_tokens_generated > 1:
        tokens_per_sec = round((total_tokens_generated - 1) / effective_gen_time, 2)
    elif total_gen_time > 0:
        tokens_per_sec = round(total_tokens_generated / total_gen_time, 2)
    else:
        tokens_per_sec = 0.0

    peak_vram_mb = None
    if device == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Build report object
    report = {
        "framework": "python-transformers",
        "model_id": args.model,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": total_tokens_generated,
        "load_time_seconds": round(load_time, 2),
        "ttft_ms": ttft_ms,
        "tokens_per_second": tokens_per_sec,
        "total_generation_seconds": round(total_gen_time, 2),
        "peak_vram_mb": peak_vram_mb,
        "generated_text": generated_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Save output file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("Benchmark Results:")
    print(f"  Load Time:           {report['load_time_seconds']} s")
    print(f"  TTFT:                {report['ttft_ms']} ms")
    print(f"  Speed:               {report['tokens_per_second']} tok/s")
    print(f"  Generated Tokens:    {report['generated_tokens']}")
    if peak_vram_mb:
        print(f"  Peak VRAM:           {report['peak_vram_mb']} MB")
    print(f"  Saved JSON to:       {output_path}")
    print("=" * 60)
    print("Generated Text:")
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
