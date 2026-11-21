"""Verified open-weight RFT data pipeline for Second v1.1.

This module rebuilds prompts with v1.1 facts and filters open-weight completions.
It rejects proprietary model names so training never uses Gemini or closed APIs.
It scores candidates with Reward v3 and outputs Fast-mode training records.
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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess

from core.python.prompt import build_user_message, format_qwen_chat_prompt, rebuild_v1_1_record
from data.pipeline.synthesize_teacher import parse_teacher_completion
from rewards.checks import is_bad_verdict
from rewards.scorer import compute_reward

LOGGER = logging.getLogger(__name__)

DEFAULT_OPEN_WEIGHT_MODEL: str = "Qwen/Qwen3.5-27B"

APPROVED_OPEN_WEIGHT_PREFIXES: tuple[str, ...] = (
    "qwen/",
    "unsloth/",
    "joebos/second-",
    "mistralai/",
    "deepseek-ai/",
    "huggingfacetb/",
)

PROPRIETARY_MODEL_RE: re.Pattern[str] = re.compile(
    r"(?:gemini|openai|anthropic|claude|chatgpt|\bgpt(?:[-_]?[0-9]|\b)|\bo[134](?:[-_]|$)|google/gemini|cohere|palm)",
    re.IGNORECASE,
)

MIN_RFT_REWARD: float = 0.82
MIN_RFT_KEY_POINT: float = 0.75
MIN_RFT_BETTER_MOVE: float = 1.00
MIN_RFT_NOT_COPIED: float = 0.85

SHORT_MIN_WORDS: int = 35
SHORT_MAX_WORDS: int = 75
LONG_MIN_WORDS: int = 90
LONG_MAX_WORDS: int = 180

RFT_SYSTEM_PROMPT: str = (
    "You are Second, an expert chess coach. "
    "Explain the move or position clearly in Fast mode with no thinking block. "
    "Follow these rules strictly:\n"
    "1. Ground every move, square, piece, and opening claim in the Facts block.\n"
    "2. If no tactic exists in the Facts, explain the positional facts such as "
    "central space, pawn tension, piece centralization, king safety, or open files.\n"
    "3. Never use vague filler phrases such as blocks coordination, maintains harmony, "
    "or unnecessary weaknesses without a concrete square and reason.\n"
    "4. Never use first-person words such as I, me, or my.\n"
    "5. Match the requested length: 35 to 75 words for detail short, "
    "or 90 to 180 words for detail long."
)


def validate_open_weight_model(model_name: str) -> str:
    """Validate that candidate synthesis uses an approved open-weight model.

    Reject proprietary models such as Gemini, GPT, or Claude with ValueError.
    Allow approved open-weight prefixes and local checkpoint paths.
    """
    cleaned = model_name.strip()
    if not cleaned:
        raise ValueError("Model name must not be empty.")
    if PROPRIETARY_MODEL_RE.search(cleaned):
        raise ValueError(
            f"Proprietary API model rejected for v1.1 training data: {cleaned!r}. "
            "Use an open-weight Qwen model instead."
        )
    lowered = cleaned.lower()
    is_local_path = cleaned.startswith(("/", "./", "../")) or Path(cleaned).exists()
    if not is_local_path and not any(lowered.startswith(prefix) for prefix in APPROVED_OPEN_WEIGHT_PREFIXES):
        raise ValueError(
            f"Model {cleaned!r} is not in the approved open-weight allowlist: "
            f"{', '.join(APPROVED_OPEN_WEIGHT_PREFIXES)}"
        )
    return cleaned


def _resolve_best_move_san(prompt_item: dict[str, Any], board: chess.Board | None = None) -> str:
    """Resolve the engine best move SAN from item metadata, eval_before PVs, or prompt text."""
    best_move_san = str(prompt_item.get("best_move_san") or "").strip()
    if best_move_san:
        return best_move_san

    eval_before = prompt_item.get("eval_before") or {}
    pvs = eval_before.get("pvs") or []
    if pvs and pvs[0].get("moves"):
        try:
            active_board = board or chess.Board(str(prompt_item.get("fen", chess.STARTING_FEN)))
            best_m = chess.Move.from_uci(str(pvs[0]["moves"][0]))
            if best_m in active_board.legal_moves:
                return active_board.san(best_m)
        except ValueError:
            pass

    m_best = re.search(
        r"^Best move:\s*([^\s.]+)\.",
        str(prompt_item.get("user_message", "")),
        re.MULTILINE,
    )
    return m_best.group(1) if m_best else ""


def _resolve_user_message(prompt_item: dict[str, Any]) -> str:
    """Return user_message or rebuild it from eval_before when missing."""
    user_msg = str(prompt_item.get("user_message", "")).strip()
    if user_msg:
        return user_msg
    if prompt_item.get("eval_before") is not None:
        return build_user_message(
            task=str(prompt_item.get("task", "explain_move")),
            level=str(prompt_item.get("level", "club")),
            detail=str(prompt_item.get("detail", "short")),
            fen=str(prompt_item.get("fen", chess.STARTING_FEN)),
            eval_before=prompt_item["eval_before"],
            eval_after=prompt_item.get("eval_after"),
            move_uci=prompt_item.get("move_uci"),
            facts=prompt_item.get("facts"),
        )
    return ""


def _apply_chat_template_no_thinking(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> str:
    """Format chat messages with enable_thinking=False, falling back on TypeError."""
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


def build_rft_messages(user_message: str) -> list[dict[str, str]]:
    """Build Fast-mode chat messages for open-weight RFT candidate synthesis."""
    return [
        {"role": "system", "content": RFT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def evaluate_rft_candidate(
    prompt_item: dict[str, Any],
    raw_completion: str,
) -> dict[str, Any]:
    """Evaluate one RFT candidate completion against Reward v3 and length gates."""
    parsed = parse_teacher_completion(raw_completion)
    explanation = parsed.get("explanation", "").strip()
    words = explanation.split()
    word_count = len(words)

    fen = str(prompt_item.get("fen", chess.STARTING_FEN))
    level = str(prompt_item.get("level", "club"))
    detail = str(prompt_item.get("detail", "short")).lower()
    verdict = prompt_item.get("verdict")
    move_uci = prompt_item.get("move_uci")
    facts = list(prompt_item.get("facts", []))
    user_msg = str(prompt_item.get("user_message", ""))

    best_move_san = _resolve_best_move_san(prompt_item)

    reward_res = compute_reward(
        text=explanation,
        fen=fen,
        level=level,
        detail=detail,
        verdict=verdict,
        move_uci=move_uci,
        best_move_san=best_move_san,
        facts=facts,
        prompt_text=user_msg,
        reward_version="v3",
    )

    failed_gates: list[str] = []
    if not reward_res.passed_all_hard:
        for check_name, chk in reward_res.hard_checks.items():
            if not chk.passed:
                failed_gates.append(f"hard_{check_name}")

    if detail == "long":
        if not (LONG_MIN_WORDS <= word_count <= LONG_MAX_WORDS):
            failed_gates.append("word_count_long")
    else:
        if not (SHORT_MIN_WORDS <= word_count <= SHORT_MAX_WORDS):
            failed_gates.append("word_count_short")

    if reward_res.reward < MIN_RFT_REWARD:
        failed_gates.append("min_reward")
    if reward_res.soft_scores.get("key_point", 0.0) < MIN_RFT_KEY_POINT:
        failed_gates.append("min_key_point")
    if is_bad_verdict(verdict):
        if reward_res.soft_scores.get("better_move", 0.0) < MIN_RFT_BETTER_MOVE:
            failed_gates.append("min_better_move")
    if reward_res.soft_scores.get("not_copied", 0.0) < MIN_RFT_NOT_COPIED:
        failed_gates.append("min_not_copied")

    return {
        "passed": len(failed_gates) == 0,
        "failed_gates": failed_gates,
        "reward": reward_res.reward,
        "explanation": explanation,
        "word_count": word_count,
        "hard_checks": {k: v.passed for k, v in reward_res.hard_checks.items()},
        "soft_scores": dict(reward_res.soft_scores),
    }


def select_best_rft_candidate(
    prompt_item: dict[str, Any],
    candidates: list[str],
    model_name: str = DEFAULT_OPEN_WEIGHT_MODEL,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select the highest-scoring passing candidate for one prompt item."""
    validated_model = validate_open_weight_model(model_name)

    user_msg = _resolve_user_message(prompt_item)
    working_item = dict(prompt_item)
    working_item["user_message"] = user_msg

    gate_counts: Counter[str] = Counter()
    passing: list[dict[str, Any]] = []

    for raw in candidates:
        ev = evaluate_rft_candidate(working_item, raw)
        if ev["passed"]:
            passing.append(ev)
        else:
            for gate in ev["failed_gates"]:
                gate_counts[gate] += 1

    telemetry = {
        "candidates_total": len(candidates),
        "candidates_passed": len(passing),
        "gate_rejections": dict(gate_counts),
    }
    if not passing:
        return None, telemetry

    best = max(
        passing,
        key=lambda ev: (
            ev["reward"],
            ev["soft_scores"].get("key_point", 0.0),
            ev["soft_scores"].get("not_copied", 0.0),
        ),
    )

    explanation = best["explanation"]
    record = dict(working_item)
    record.update({
        "user_message": user_msg,
        "comment": explanation,
        "explanation": explanation,
        "assistant_response": f"<think>\n\n</think>\n\n{explanation}",
        "chat_prompt": format_qwen_chat_prompt(user_msg),
        "thinking_mode": "off",
        "reward": best["reward"],
        "reward_version": "v3",
        "rft_model": validated_model,
        "word_count": best["word_count"],
    })
    return record, telemetry


def filter_rft_dataset(
    input_path: Path | str,
    output_path: Path | str,
    model_name: str = DEFAULT_OPEN_WEIGHT_MODEL,
) -> dict[str, Any]:
    """Filter Best-of-K open-weight candidate records using Reward v3 gates."""
    validated_model = validate_open_weight_model(model_name)
    in_p = Path(input_path)
    out_p = Path(output_path)

    total_prompts = 0
    total_candidates = 0
    accepted = 0
    gate_totals: Counter[str] = Counter()
    selected_records: list[dict[str, Any]] = []

    with open(in_p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            item = json.loads(s)
            total_prompts += 1

            raw_cands = item.get("candidates")
            if isinstance(raw_cands, list):
                candidates = [str(c) for c in raw_cands]
            elif item.get("completion") or item.get("output") or item.get("raw_completion"):
                candidates = [
                    str(item.get("raw_completion") or item.get("completion") or item.get("output"))
                ]
            else:
                candidates = []

            total_candidates += len(candidates)
            item_model = str(item.get("model") or validated_model)
            rec, tel = select_best_rft_candidate(item, candidates, model_name=item_model)
            for gate, count in tel["gate_rejections"].items():
                gate_totals[gate] += count
            if rec is not None:
                rec.pop("candidates", None)
                selected_records.append(rec)
                accepted += 1

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for rec in selected_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "model": validated_model,
        "total_prompts": total_prompts,
        "total_candidates": total_candidates,
        "accepted_prompts": accepted,
        "gate_rejections": dict(gate_totals),
    }


def _build_mock_rft_candidate(prompt_item: dict[str, Any], variant_idx: int) -> str:
    """Create a deterministic Fast-mode candidate completion for offline pipeline checks."""
    fen = str(prompt_item.get("fen", chess.STARTING_FEN))
    board = chess.Board(fen)
    move_uci = prompt_item.get("move_uci")
    move_san = str(prompt_item.get("move_san") or "").strip()
    if not move_san and move_uci:
        try:
            m = chess.Move.from_uci(str(move_uci))
            if m in board.legal_moves:
                move_san = board.san(m)
        except ValueError:
            move_san = ""

    best_san = _resolve_best_move_san(prompt_item, board=board)
    facts_list = [str(f) for f in prompt_item.get("facts", [])]
    first_reply = ""
    extra_fact_cues: list[str] = []
    for fact in facts_list:
        m_rep = re.search(r"best reply is\s+([^\s.]+)\.", fact)
        if m_rep and not first_reply:
            first_reply = m_rep.group(1)
        m_chk = re.search(r"has a check:\s+([^\s.]+)\.", fact)
        if m_chk:
            extra_fact_cues.append(f"the check {m_chk.group(1)}")
        m_cap = re.search(r"can capture .*?\swith\s+([^\s.]+)\.", fact)
        if m_cap:
            extra_fact_cues.append(f"the capture {m_cap.group(1)}")
        m_thr = re.search(r"threatens\s+([^\s.]+)\.", fact)
        if m_thr:
            extra_fact_cues.append(f"the threat {m_thr.group(1)}")
        m_kchk = re.search(r"king is in check on ([a-h][1-8])\.", fact)
        if m_kchk:
            extra_fact_cues.append(f"the check on {m_kchk.group(1)}")
        m_of = re.search(r"The ([a-h])-file is open\.", fact)
        if m_of:
            extra_fact_cues.append(f"the open {m_of.group(1)}-file")
        m_pt = re.search(r"Central pawn tension exists between ([a-h][1-8]) and ([a-h][1-8])\.", fact)
        if m_pt:
            extra_fact_cues.append(f"pawn tension near {m_pt.group(1)} and {m_pt.group(2)}")
        m_dev = re.search(r"(White|Black) leads in piece development\.", fact)
        if m_dev:
            extra_fact_cues.append(f"{m_dev.group(1)} holds a development lead")
        m_spc = re.search(r"(White|Black) has more central space\.", fact)
        if m_spc:
            extra_fact_cues.append(f"{m_spc.group(1)} holds more central space")
        m_hang = re.search(r"The (pawn|knight|bishop|rook|queen) on ([a-h][1-8]) is not defended\.", fact)
        if m_hang:
            extra_fact_cues.append(f"the {m_hang.group(1)} resting on {m_hang.group(2)} lacks protection")
        m_cas = re.search(r"(White|Black) has not castled\.", fact)
        if m_cas:
            extra_fact_cues.append(f"{m_cas.group(1)} has uncastled safety risks")
        m_mat = re.search(r"(White|Black) is up (?:a|two|the) ([a-z ]+?)(?:\s+in material)?\.", fact)
        if m_mat:
            extra_fact_cues.append(f"{m_mat.group(1)} holds an extra {m_mat.group(2)}")

    verdict = str(prompt_item.get("verdict") or "Good")
    facts_text = " ".join(facts_list).lower()
    detail = str(prompt_item.get("detail", "short")).lower()

    concept_phrases: list[str] = []
    if "centraliz" in facts_text or "recapture" in facts_text:
        concept_phrases.append("guides pieces toward the center")
    if any(k in facts_text for k in ("central space", "central tension", "tension", "instead of trading", "central pawn", "challenges the center", "gains space")):
        concept_phrases.append("holds central space and pawn tension")
    if any(k in facts_text for k in ("activity", "active pieces", "develops", "piece development")):
        concept_phrases.append("brings pieces out into active play")
    if any(k in facts_text for k in ("castles to", "king safety", "prepares castling")):
        concept_phrases.append("supports castling and king safety")
    if "-file" in facts_text or "open file" in facts_text or "half-open file" in facts_text:
        concept_phrases.append("contests open files cleanly")
    if not concept_phrases:
        concept_phrases.append("keeps central control and active pieces")

    concept_clause = " and ".join(concept_phrases[:2])
    side_name = "White" if board.turn == chess.WHITE else "Black"
    opp_name = "Black" if board.turn == chess.WHITE else "White"
    opp_concept = (
        "piece moves toward the center"
        if "centraliz" in facts_text or "recapture" in facts_text
        else "central space and active play"
    )
    if first_reply:
        reply_clause = f" After {first_reply}, {opp_name} gains pressure."
        cue_clause = f" Also note {extra_fact_cues[0]}." if extra_fact_cues else ""
    elif extra_fact_cues:
        reply_clause = f" Also note {extra_fact_cues[0]}."
        cue_clause = f" Notice {extra_fact_cues[1]}." if len(extra_fact_cues) >= 2 else ""
    else:
        reply_clause = ""
        cue_clause = ""

    if move_san:
        verdict_phrase = verdict.lower() if is_bad_verdict(verdict) else "sound choice"
        article = "an" if verdict_phrase.startswith(("a", "e", "i", "o", "u")) else "a"
        lead_openers = (
            f"Playing {move_san} is {article} {verdict_phrase}. It {concept_clause}.{reply_clause}",
            f"Choosing {move_san} is {article} {verdict_phrase}. It {concept_clause}.{reply_clause}",
            f"The move {move_san} is {article} {verdict_phrase}. It {concept_clause}.{reply_clause}",
            f"Here {move_san} is {article} {verdict_phrase}. It {concept_clause}.{reply_clause}",
        )
        lead = lead_openers[variant_idx % len(lead_openers)]
    else:
        pos_openers = (
            f"{side_name} faces a key moment. The plan {concept_clause}.{reply_clause}",
            f"In this position, {side_name} {concept_clause}.{reply_clause}",
            f"{side_name} can guide the game cleanly. This setup {concept_clause}.{reply_clause}",
            f"Here {side_name}'s main plan {concept_clause}.{reply_clause}",
        )
        lead = pos_openers[variant_idx % len(pos_openers)]

    if best_san and is_bad_verdict(verdict):
        alt = (
            f"Instead, {side_name} should play {best_san}. "
            f"That move guards targets, supports {opp_concept}, and limits {opp_name}."
        )
    elif best_san:
        alt = (
            f"Continuing with {best_san} keeps balanced pieces. "
            f"It protects the king safely and limits {opp_name}."
        )
    else:
        alt = (
            f"{side_name} should focus on active pieces and king safety. "
            f"Steady central pressure keeps {opp_name} contained."
        )

    if detail == "long":
        long_tails = (
            (
                f"Over the next few moves, careful piece placement matters most. "
                f"{side_name} should avoid giving {opp_name} free targets or open files. "
                f"Check forcing captures and checks before picking the next move. "
                f"Safe king shelter lets {side_name} build pressure step by step."
            ),
            (
                f"Pay close attention to open files and safe squares for every piece. "
                f"Keep {side_name}'s pieces active without letting {opp_name} find easy replies. "
                f"Always check forcing moves first before choosing a quiet plan. "
                f"Patient play turns small gains into lasting pressure."
            ),
        )
        long_mid = long_tails[variant_idx % len(long_tails)]
        combined = f"{lead}{cue_clause} {alt}\n\n{long_mid}".strip()
        pad_sentences = [
            f"Steady development and king safety remain main goals for {side_name}.",
            f"Guard key central squares before starting a direct attack against {opp_name}.",
            f"Clear calculation prevents missed replies and keeps every piece useful.",
            f"Each move should protect weak spots while preparing an active follow-up.",
        ]
        p_idx = 0
        while len(combined.split()) < 105 and p_idx < len(pad_sentences):
            combined = f"{combined} {pad_sentences[p_idx]}"
            p_idx += 1
        explanation = combined
    else:
        short_base = f"{lead}{cue_clause} {alt}".strip()
        short_pads = (
            "Careful piece placement and clear calculation guide the next moves.",
            "Keep checking forcing captures and checks before picking the follow-up.",
        )
        sp_idx = 0
        while len(short_base.split()) < 42 and sp_idx < len(short_pads):
            short_base = f"{short_base} {short_pads[sp_idx]}"
            sp_idx += 1
        words_now = short_base.split()
        if len(words_now) > 68:
            short_base = " ".join(words_now[:66]).rstrip(",;:") + "."
        explanation = short_base

    return f"<think>\n\n</think>\n\n{explanation}"


def generate_rft_candidates(
    prompts_path: Path | str,
    output_path: Path | str,
    model_name: str = DEFAULT_OPEN_WEIGHT_MODEL,
    num_samples: int = 4,
    backend: str = "vllm",
    mock: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate Fast-mode candidate completions using an approved open-weight model."""
    validated_model = validate_open_weight_model(model_name)
    in_p = Path(prompts_path)
    out_p = Path(output_path)

    items: list[dict[str, Any]] = []
    with open(in_p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
            if limit is not None and len(items) >= limit:
                break

    if not mock and backend not in ("vllm", "transformers"):
        raise ValueError(f"Unsupported backend: {backend!r}. Expected 'vllm' or 'transformers'.")

    results: list[dict[str, Any]] = []
    if mock:
        for item in items:
            cands = [_build_mock_rft_candidate(item, idx) for idx in range(max(1, num_samples))]
            out_rec = dict(item)
            out_rec["model"] = validated_model
            out_rec["candidates"] = cands
            results.append(out_rec)
    elif backend == "vllm":
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(validated_model, trust_remote_code=True)
        llm = LLM(model=validated_model, trust_remote_code=True)
        sampling_params = SamplingParams(
            n=max(1, num_samples),
            temperature=0.7,
            top_p=0.9,
            max_tokens=320,
        )
        prompts_text = [
            _apply_chat_template_no_thinking(
                tokenizer,
                build_rft_messages(_resolve_user_message(item)),
            )
            for item in items
        ]
        outputs = llm.generate(prompts_text, sampling_params)
        for item, req_out in zip(items, outputs):
            out_rec = dict(item)
            out_rec["model"] = validated_model
            out_rec["candidates"] = [cand.text for cand in req_out.outputs]
            results.append(out_rec)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(validated_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            validated_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        for item in items:
            formatted = _apply_chat_template_no_thinking(
                tokenizer,
                build_rft_messages(_resolve_user_message(item)),
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=320,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=max(1, num_samples),
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            cands = [
                tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                for seq in gen_out
            ]
            out_rec = dict(item)
            out_rec["model"] = validated_model
            out_rec["candidates"] = cands
            results.append(out_rec)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "model": validated_model,
        "backend": "mock" if mock else backend,
        "prompts_processed": len(results),
        "candidates_per_prompt": max(1, num_samples),
    }


def rebuild_v1_1_prompts(
    input_path: Path | str,
    output_path: Path | str,
) -> int:
    """Re-extract v1.1 facts and prompts from analyzed JSONL records."""
    in_p = Path(input_path)
    out_p = Path(output_path)
    rebuilt: list[dict[str, Any]] = []

    with open(in_p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            rebuilt.append(rebuild_v1_1_record(rec))

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for rec in rebuilt:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(rebuilt)


def main() -> None:
    """Parse CLI arguments and run v1.1 prompt rebuilding, candidate generation, or RFT filtering."""
    parser = argparse.ArgumentParser(description="Second v1.1 open-weight RFT pipeline.")
    parser.add_argument(
        "--mode",
        choices=("rebuild-prompts", "generate", "filter"),
        required=True,
        help="Pipeline mode: rebuild-prompts, generate, or filter.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--model",
        default=DEFAULT_OPEN_WEIGHT_MODEL,
        help="Open-weight generator model name.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of candidate completions per prompt for generate mode.",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "transformers"),
        default="vllm",
        help="Inference backend for generate mode.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic offline mock generation in generate mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional prompt count limit in generate mode.",
    )
    args = parser.parse_args()

    if args.mode == "rebuild-prompts":
        count = rebuild_v1_1_prompts(args.input, args.output)
        print(f"Rebuilt {count} v1.1 prompts to {args.output}.")
    elif args.mode == "generate":
        stats = generate_rft_candidates(
            args.input,
            args.output,
            model_name=args.model,
            num_samples=args.num_samples,
            backend=args.backend,
            mock=args.mock,
            limit=args.limit,
        )
        print(json.dumps(stats, indent=2))
    else:
        stats = filter_rft_dataset(args.input, args.output, model_name=args.model)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
