"""Prepare Stage T4 RL v2 prompts with Stage T1 motif facts and difficulty filtering.

This module builds data/processed/rl_prompts_v2.jsonl for GRPO v2 training.
It prioritizes medium-to-hard positions where teacher candidates did not pass
and supplements them with motif-refreshed RL positions.
It enforces zero game_id and FEN overlap with devset-1.0 and testset-1.0.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.python import (
    build_user_message,
    extract_facts,
    format_qwen_chat_prompt,
)
from data.pipeline.build_teacher_prompts import load_exclusion_sets
from data.pipeline.splits import normalize_fen

LOGGER = logging.getLogger(__name__)

DEFAULT_TEACHER_PROMPTS = Path("data/processed/teacher_prompts.jsonl")
DEFAULT_VERIFIED_TEACHER = Path("data/processed/teacher_verified_non_thinking.jsonl")
DEFAULT_RL_PROMPTS_V1 = Path("data/processed/rl_prompts.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/rl_prompts_v2.jsonl")
DEFAULT_TARGET_COUNT = 5000


def refresh_record_with_v2_facts(rec: dict[str, Any]) -> dict[str, Any]:
    """Re-extract Stage T1 tactical motif facts if eval_before is present.

    Update facts, user_message, and chat_prompt with the shared Python core.
    Remove heavy eval_before and eval_after fields to keep bundle size compact.
    Return the refreshed prompt dictionary.
    """
    out = dict(rec)
    eval_before = out.pop("eval_before", None)
    eval_after = out.pop("eval_after", None)

    fen = str(out.get("fen", ""))
    task = str(out.get("task", "explain_move"))
    level = str(out.get("level", "club"))
    detail = str(out.get("detail", "short"))
    move_uci = out.get("move_uci")

    if eval_before and fen:
        try:
            facts = extract_facts(
                task=task,
                fen=fen,
                eval_before=eval_before,
                move_uci=move_uci if task == "explain_move" else None,
                eval_after=eval_after if task == "explain_move" else None,
            )
            user_msg = build_user_message(
                task=task,
                level=level,
                detail=detail,
                fen=fen,
                eval_before=eval_before,
                move_uci=move_uci if task == "explain_move" else None,
                eval_after=eval_after if task == "explain_move" else None,
            )
            chat_prompt = format_qwen_chat_prompt(user_msg)
            out["facts"] = facts
            out["user_message"] = user_msg
            out["chat_prompt"] = chat_prompt
        except Exception as exc:
            LOGGER.warning("Could not refresh facts for %s: %s", out.get("id"), exc)

    return out


def build_rl_prompts_v2(
    teacher_prompts_path: Path | str = DEFAULT_TEACHER_PROMPTS,
    verified_teacher_path: Path | str = DEFAULT_VERIFIED_TEACHER,
    rl_prompts_v1_path: Path | str = DEFAULT_RL_PROMPTS_V1,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build difficulty-filtered RL v2 prompt dataset with motif facts.

    Exclude all game IDs and FENs from devset-1.0, testset-1.0, and sft_val.
    Prioritize medium-to-hard prompts from teacher_prompts that were rejected
    during SFT teacher filtering (focusing RL compute on unsolved positions).
    Supplement with motif-refreshed prompts from rl_prompts.jsonl and remaining
    teacher prompts up to target_count.
    Save records to output_path and return (records, summary_stats).
    """
    excluded_games, excluded_fens = load_exclusion_sets()

    # Load IDs of prompts that already succeeded in SFT teacher verification
    easy_solved_ids: set[str] = set()
    v_path = Path(verified_teacher_path)
    if not v_path.is_file() and (PROJECT_ROOT / v_path).is_file():
        v_path = PROJECT_ROOT / v_path
    if v_path.is_file():
        with open(v_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    pid = str(item.get("prompt_id") or item.get("id") or "")
                    if pid:
                        easy_solved_ids.add(pid)
                    # Also strip _non_thinking suffix if present
                    if pid.endswith("_non_thinking"):
                        easy_solved_ids.add(pid[: -len("_non_thinking")])

    hard_teacher_pool: list[dict[str, Any]] = []
    solved_teacher_pool: list[dict[str, Any]] = []
    seen_fens: set[str] = set(excluded_fens)

    t_path = Path(teacher_prompts_path)
    if not t_path.is_file() and (PROJECT_ROOT / t_path).is_file():
        t_path = PROJECT_ROOT / t_path
    if t_path.is_file():
        with open(t_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                gid = str(rec.get("game_id", ""))
                fen_norm = normalize_fen(str(rec.get("fen", "")))
                if (gid and gid in excluded_games) or (fen_norm in seen_fens):
                    continue
                seen_fens.add(fen_norm)
                cleaned = refresh_record_with_v2_facts(rec)
                if str(rec.get("id", "")) in easy_solved_ids:
                    solved_teacher_pool.append(cleaned)
                else:
                    hard_teacher_pool.append(cleaned)

    rng = random.Random(seed)
    rng.shuffle(hard_teacher_pool)
    rng.shuffle(solved_teacher_pool)

    selected: list[dict[str, Any]] = []
    for rec in hard_teacher_pool:
        if len(selected) >= target_count:
            break
        rec["difficulty_bucket"] = "hard_unsolved"
        selected.append(rec)

    # Supplement from rl_prompts.jsonl (refreshed with Stage T1 motif facts)
    r1_path = Path(rl_prompts_v1_path)
    if not r1_path.is_file() and (PROJECT_ROOT / r1_path).is_file():
        r1_path = PROJECT_ROOT / r1_path
    refreshed_v1_count = 0
    if len(selected) < target_count and r1_path.is_file():
        v1_candidates: list[dict[str, Any]] = []
        with open(r1_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                gid = str(rec.get("game_id", ""))
                fen_norm = normalize_fen(str(rec.get("fen", "")))
                if (gid and gid in excluded_games) or (fen_norm in seen_fens):
                    continue
                seen_fens.add(fen_norm)
                v1_candidates.append(rec)
                if len(v1_candidates) >= (target_count - len(selected)) * 2:
                    break

        rng.shuffle(v1_candidates)
        for rec in v1_candidates:
            if len(selected) >= target_count:
                break
            cleaned = refresh_record_with_v2_facts(rec)
            cleaned["difficulty_bucket"] = "refreshed_rl_v1"
            selected.append(cleaned)
            refreshed_v1_count += 1

    # If still below target_count, fill from solved_teacher_pool
    solved_fill_count = 0
    for rec in solved_teacher_pool:
        if len(selected) >= target_count:
            break
        rec["difficulty_bucket"] = "medium_solved"
        selected.append(rec)
        solved_fill_count += 1

    rng.shuffle(selected)

    out_file = Path(output_path)
    if not out_file.is_absolute() and not out_file.parent.exists():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for idx, r in enumerate(selected):
            r["rl_v2_index"] = idx
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total_selected": len(selected),
        "hard_unsolved_count": len(selected) - refreshed_v1_count - solved_fill_count,
        "refreshed_rl_v1_count": refreshed_v1_count,
        "medium_solved_count": solved_fill_count,
        "output_path": str(out_file),
    }
    LOGGER.info("Built RL prompts v2: %s", json.dumps(stats, indent=2))
    return selected, stats


def main() -> None:
    """CLI entry point for building rl_prompts_v2.jsonl."""
    parser = argparse.ArgumentParser(
        description="Build difficulty-filtered RL v2 prompts with Stage T1 motif facts."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for rl_prompts_v2.jsonl",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Target number of RL prompts",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _, stats = build_rl_prompts_v2(
        output_path=args.output,
        target_count=args.count,
    )
    print(f"Created {stats['total_selected']} RL v2 prompts at {stats['output_path']}")


if __name__ == "__main__":
    main()
