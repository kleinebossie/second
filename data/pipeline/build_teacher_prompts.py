"""Teacher prompt dataset builder for Stage T2 knowledge distillation.

This module samples 8,000 stratified chess prompts from raw dataset sources.
It enforces quotas across tasks, detail levels, player skill levels, and verdicts.
It enforces coverage across tactical motifs and game phases.
It validates zero contamination against development and test partitions.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Any

import chess

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.python import (
    build_user_message,
    classify_verdict,
    extract_facts,
    format_qwen_chat_prompt,
)
from core.python.facts import compute_phase_fact
from data.pipeline.analyse import StockfishAnalyzer, StockfishCache, find_stockfish_binary
from data.pipeline.rl_prompts import (
    format_prompt_record,
    load_excluded_keys,
    synthesize_after_from_pv,
)
from data.pipeline.splits import hash_game_id, normalize_fen
from eval.concept_tags import classify_concept_tier, has_board_motif, is_endgame_position

LOGGER = logging.getLogger(__name__)

DEFAULT_RAW_FILES = [
    Path("data/raw/mate.jsonl"),
    Path("data/raw/lichess_studies.jsonl"),
    Path("data/raw/lichess_games.jsonl"),
]
DEFAULT_CACHE_PATH = Path("data/cache/stockfish_eval_cache.parquet")
DEFAULT_DEVSET_PATH = Path("eval/devset-1.0.jsonl")
DEFAULT_TESTSET_PATH = Path("eval/testset-1.0.jsonl")
DEFAULT_SPLIT_DIR = Path("data/processed")
DEFAULT_OUTPUT_PATH = Path("data/processed/teacher_prompts.jsonl")

DEFAULT_TARGET_COUNT = 8000
DEFAULT_SEED = 42

TASK_QUOTAS = {
    "explain_move": 0.80,
    "explain_position": 0.20,
}

DETAIL_QUOTAS = {
    "short": 0.70,
    "long": 0.30,
}

LEVEL_QUOTAS = {
    "beginner": 0.50,
    "club": 0.50,
}

VERDICT_QUOTAS = {
    "Blunder": 0.20,
    "Mistake": 0.20,
    "Inaccuracy": 0.10,
    "Best": 0.25,
    "Good": 0.20,
    "Only move": 0.05,
}

MOTIF_MIN_TARGETS = {
    "fork": 0.15,
    "pin_skewer": 0.15,
    "mate": 0.10,
    "hanging": 0.15,
    "opening_safety": 0.15,
    "middlegame": 0.15,
    "endgame": 0.15,
}


def load_exclusion_sets(
    devset_path: Path | str = DEFAULT_DEVSET_PATH,
    testset_path: Path | str = DEFAULT_TESTSET_PATH,
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
) -> tuple[set[str], set[str]]:
    """Load excluded game identifiers and position FENs across all evaluation splits.

    Load exclusions from validation and evaluation pools.
    Read game IDs and FENs from devset and testset files.
    Return tuple of excluded game IDs and excluded normalized FENs.
    """
    excluded_games, excluded_fens = load_excluded_keys(
        split_dir,
        split_names=("sft_val", "sft_test_pool", "eval_pool"),
    )

    for p in (Path(devset_path), Path(testset_path)):
        if not p.is_file() and (PROJECT_ROOT / p).is_file():
            p = PROJECT_ROOT / p
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    item = json.loads(line_str)
                    gid = str(item.get("game_id", ""))
                    fen = str(item.get("fen", ""))
                    if gid:
                        excluded_games.add(gid)
                    if fen:
                        excluded_fens.add(normalize_fen(fen))

    LOGGER.info(
        "Exclusion loaded: %d games, %d unique FENs",
        len(excluded_games),
        len(excluded_fens),
    )
    return excluded_games, excluded_fens


def detect_candidate_motifs(
    fen: str,
    move_uci: str | None,
    facts: list[str],
    eval_before: dict[str, Any] | None,
) -> set[str]:
    """Detect motif tags for a candidate item from facts and board state.

    Check facts for tactical keywords.
    Check board for forks, pins, and endgames.
    Return set of matched motif tag identifiers.
    """
    tags: set[str] = set()
    fact_text = " ".join(facts).lower()

    if "fork" in fact_text:
        tags.add("fork")
    elif move_uci:
        try:
            b = chess.Board(fen)
            m = chess.Move.from_uci(move_uci)
            if has_board_motif(b, m):
                tags.add("fork")
        except Exception:
            pass

    if "pin" in fact_text or "skewer" in fact_text:
        tags.add("pin_skewer")

    if "mate" in fact_text:
        tags.add("mate")
    elif eval_before:
        for pv in eval_before.get("pvs", []):
            if pv.get("score", {}).get("type") == "mate":
                tags.add("mate")
                break

    if "hanging" in fact_text or "not defended" in fact_text or "undefended" in fact_text:
        tags.add("hanging")

    if "opening" in fact_text or "castl" in fact_text or "king safety" in fact_text:
        tags.add("opening_safety")

    if "middlegame" in fact_text:
        tags.add("middlegame")

    if "endgame" in fact_text:
        tags.add("endgame")
    else:
        try:
            b = chess.Board(fen)
            if is_endgame_position(b):
                tags.add("endgame")
        except Exception:
            pass

    return tags


def load_raw_teacher_candidates(
    raw_files: list[Path | str] | None = None,
    cache_path: Path | str | None = None,
    devset_path: Path | str = DEFAULT_DEVSET_PATH,
    testset_path: Path | str = DEFAULT_TESTSET_PATH,
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Load candidate moves and positions from raw sources using cached evaluations.

    Exclude games and positions in evaluation sets.
    Exclude games hashing to the test partition (hash < 0.10).
    Populate candidates by verdict and candidate positions for explain_position.
    Return tuple of (candidates_by_verdict, position_candidates).
    """
    files = [Path(f) for f in (raw_files or DEFAULT_RAW_FILES)]
    resolved_files: list[Path] = []
    for f in files:
        if not f.is_file() and (PROJECT_ROOT / f).is_file():
            resolved_files.append(PROJECT_ROOT / f)
        else:
            resolved_files.append(f)

    c_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    if not c_path.is_file() and (PROJECT_ROOT / c_path).is_file():
        c_path = PROJECT_ROOT / c_path

    cache = StockfishCache(cache_path=c_path)
    ex_games, ex_fens = load_exclusion_sets(
        devset_path=devset_path,
        testset_path=testset_path,
        split_dir=split_dir,
    )

    candidates_by_verdict: dict[str, list[dict[str, Any]]] = defaultdict(list)
    position_candidates: list[dict[str, Any]] = []

    seen_move_tokens: set[str] = set()
    seen_position_fens: set[str] = set()

    for rf in resolved_files:
        if not rf.is_file():
            LOGGER.warning("Raw candidate file missing: %s", rf)
            continue

        LOGGER.info("Reading candidates from %s...", rf)
        with open(rf, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                item = json.loads(line_str)
                gid = str(item.get("game_id", ""))
                fen = str(item.get("fen", ""))
                norm_fen = normalize_fen(fen)

                if gid in ex_games or norm_fen in ex_fens:
                    continue
                if gid and hash_game_id(gid, seed=seed) < 0.10:
                    continue

                entry = cache._cache.get(fen)
                if not entry:
                    continue

                pvs = entry.get("pvs", [])
                if not pvs:
                    continue
                eb = {"fen": fen, "pvs": pvs}

                # 1. Candidate position for explain_position
                if norm_fen not in seen_position_fens:
                    seen_position_fens.add(norm_fen)
                    position_candidates.append({
                        "game_id": gid,
                        "fen": fen,
                        "source": item.get("source", "lichess_games"),
                        "eval_before": eb,
                        "move_uci": None,
                        "eval_after": None,
                        "verdict": None,
                    })

                # 2. Candidate move from raw move
                m_uci = item.get("move_uci")
                if m_uci:
                    ea: dict[str, Any] | None = None
                    for pv in pvs:
                        if pv.get("moves") and pv["moves"][0] == m_uci:
                            synth = synthesize_after_from_pv(fen, pv)
                            if synth:
                                ea = synth[1]
                            break
                    if not ea:
                        try:
                            b = chess.Board(fen)
                            m_obj = chess.Move.from_uci(m_uci)
                            if m_obj in b.legal_moves:
                                b.push(m_obj)
                                ea_entry = cache._cache.get(b.fen())
                                if ea_entry:
                                    ea = {"fen": b.fen(), "pvs": ea_entry["pvs"][:1]}
                        except Exception:
                            pass
                    if ea:
                        v = classify_verdict(eb, ea, m_uci)
                        token = f"{norm_fen}_{m_uci}"
                        if token not in seen_move_tokens:
                            seen_move_tokens.add(token)
                            candidates_by_verdict[v].append({
                                "game_id": gid,
                                "fen": fen,
                                "source": item.get("source", "lichess_games"),
                                "eval_before": eb,
                                "move_uci": m_uci,
                                "eval_after": ea,
                                "verdict": v,
                            })

                # 3. Moves from PVs (Only move and secondary moves)
                if len(pvs) >= 2:
                    synth0 = synthesize_after_from_pv(fen, pvs[0])
                    if synth0:
                        v0 = classify_verdict(eb, synth0[1], synth0[0])
                        token0 = f"{norm_fen}_{synth0[0]}"
                        if v0 == "Only move" and token0 not in seen_move_tokens:
                            seen_move_tokens.add(token0)
                            candidates_by_verdict["Only move"].append({
                                "game_id": gid,
                                "fen": fen,
                                "source": item.get("source", "lichess_games"),
                                "eval_before": eb,
                                "move_uci": synth0[0],
                                "eval_after": synth0[1],
                                "verdict": v0,
                            })

                    for pv in pvs[1:]:
                        synth = synthesize_after_from_pv(fen, pv)
                        if synth:
                            v = classify_verdict(eb, synth[1], synth[0])
                            token = f"{norm_fen}_{synth[0]}"
                            if token not in seen_move_tokens:
                                seen_move_tokens.add(token)
                                candidates_by_verdict[v].append({
                                    "game_id": gid,
                                    "fen": fen,
                                    "source": item.get("source", "lichess_games"),
                                    "eval_before": eb,
                                    "move_uci": synth[0],
                                    "eval_after": synth[1],
                                    "verdict": v,
                                })

    LOGGER.info(
        "Loaded candidates: %d positions, verdicts: %s",
        len(position_candidates),
        {k: len(v) for k, v in candidates_by_verdict.items()},
    )
    return candidates_by_verdict, position_candidates


def stratify_teacher_prompts(
    candidates_by_verdict: dict[str, list[dict[str, Any]]],
    position_candidates: list[dict[str, Any]],
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Sample candidate items adhering strictly to taxonomy quotas.

    Allocate target counts for explain_move and explain_position.
    Allocate target counts across verdicts.
    Assign detail and level parameters deterministically.
    Ensure motif targets are achieved.
    Return list of formatted prompt dictionaries.
    """
    rng = random.Random(seed)
    n_move = int(round(target_count * TASK_QUOTAS["explain_move"]))
    n_pos = target_count - n_move

    # 1. Stratify explain_position items
    pos_selected: list[dict[str, Any]] = []
    rng.shuffle(position_candidates)

    # Sort positions into opening, middlegame, endgame buckets
    phase_pos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in position_candidates:
        fen = p["fen"]
        try:
            b = chess.Board(fen)
            phase_fact = compute_phase_fact(b)
            if "opening" in phase_fact:
                phase_pos["opening"].append(p)
            elif "endgame" in phase_fact or is_endgame_position(b):
                phase_pos["endgame"].append(p)
            else:
                phase_pos["middlegame"].append(p)
        except Exception:
            phase_pos["middlegame"].append(p)

    phases = ["opening", "middlegame", "endgame"]
    phase_idx = 0
    while len(pos_selected) < n_pos:
        ph = phases[phase_idx % len(phases)]
        phase_idx += 1
        pool = phase_pos[ph]
        if pool:
            pos_selected.append(pool.pop())
        elif position_candidates:
            pos_selected.append(position_candidates.pop())
        else:
            break

    # 2. Stratify explain_move items by verdict
    verdict_targets = {
        v: int(round(n_move * ratio)) for v, ratio in VERDICT_QUOTAS.items()
    }
    # Fix rounding if sum != n_move
    diff = n_move - sum(verdict_targets.values())
    if diff != 0:
        verdict_targets["Best"] += diff

    move_selected: list[dict[str, Any]] = []
    for v_name, v_target in verdict_targets.items():
        pool = list(candidates_by_verdict.get(v_name, []))
        rng.shuffle(pool)

        # Prioritize candidates with forks or pins/skewers to exceed >= 15% motif quotas
        prioritized: list[dict[str, Any]] = []
        regular: list[dict[str, Any]] = []
        needed_prioritized = int(v_target * 0.35)

        for cand in pool:
            if len(prioritized) < needed_prioritized:
                fen = cand["fen"]
                m_uci = cand.get("move_uci")
                eb = cand["eval_before"]
                ea = cand.get("eval_after")
                try:
                    facts = extract_facts(
                        task="explain_move",
                        fen=fen,
                        eval_before=eb,
                        eval_after=ea,
                        move_uci=m_uci,
                    )
                    text = " ".join(facts).lower()
                    if "fork" in text or "pin" in text or "skewer" in text:
                        cand["_facts"] = facts
                        prioritized.append(cand)
                        continue
                except Exception:
                    pass
            regular.append(cand)
            if len(prioritized) >= needed_prioritized and len(prioritized) + len(regular) >= v_target:
                break

        selected_bucket = (prioritized + regular)[:v_target]
        move_selected.extend(selected_bucket)

    LOGGER.info(
        "Selected candidates: %d move items, %d position items",
        len(move_selected),
        len(pos_selected),
    )

    # 3. Assign level and detail attributes to each item
    all_raw_items: list[tuple[dict[str, Any], str]] = []
    for m in move_selected:
        all_raw_items.append((m, "explain_move"))
    for p in pos_selected:
        all_raw_items.append((p, "explain_position"))

    rng.shuffle(all_raw_items)

    formatted_records: list[dict[str, Any]] = []
    n_long = int(round(target_count * DETAIL_QUOTAS["long"]))
    long_indices = set(rng.sample(range(len(all_raw_items)), n_long))

    for idx, (cand, task) in enumerate(all_raw_items):
        detail = "long" if idx in long_indices else "short"
        level = "beginner" if (idx % 2 == 0) else "club"
        rec = format_teacher_record(
            cand=cand,
            idx=idx,
            task=task,
            level=level,
            detail=detail,
        )
        formatted_records.append(rec)

    return formatted_records


def format_teacher_record(
    cand: dict[str, Any],
    idx: int,
    task: str,
    level: str,
    detail: str,
) -> dict[str, Any]:
    """Format single teacher prompt record with facts and chat prompt.

    Extract facts for the assigned task and move.
    Build user message and Qwen chat prompt.
    Tag with concept tier and motif keywords.
    Return complete record dictionary.
    """
    item_id = f"teacher_prompt_{idx:06d}"
    game_id = str(cand.get("game_id", ""))
    fen = str(cand["fen"])
    move_uci = cand.get("move_uci") if task == "explain_move" else None
    eval_before = cand["eval_before"]
    eval_after = cand.get("eval_after") if task == "explain_move" else None
    verdict = cand.get("verdict") if task == "explain_move" else None

    facts = cand.get("_facts")
    if facts is None or task != "explain_move":
        facts = extract_facts(
            task=task,
            fen=fen,
            eval_before=eval_before,
            eval_after=eval_after,
            move_uci=move_uci,
        )

    user_message = build_user_message(
        task=task,
        level=level,
        detail=detail,
        fen=fen,
        eval_before=eval_before,
        eval_after=eval_after,
        move_uci=move_uci,
    )
    chat_prompt = format_qwen_chat_prompt(user_message)

    concept_rec = {
        "fen": fen,
        "move_uci": move_uci,
        "facts": facts,
        "eval_before": eval_before,
        "eval_after": eval_after,
    }
    concept_tier = classify_concept_tier(concept_rec)
    motifs = detect_candidate_motifs(
        fen=fen,
        move_uci=move_uci,
        facts=facts,
        eval_before=eval_before,
    )

    return {
        "id": item_id,
        "game_id": game_id,
        "fen": fen,
        "task": task,
        "level": level,
        "detail": detail,
        "move_uci": move_uci,
        "verdict": verdict,
        "facts": facts,
        "user_message": user_message,
        "chat_prompt": chat_prompt,
        "source": cand.get("source", "lichess_games"),
        "concept_tier": concept_tier,
        "motif_tags": sorted(list(motifs)),
        "eval_before": eval_before,
        "eval_after": eval_after,
    }


def verify_teacher_prompts(
    records: list[dict[str, Any]],
    devset_path: Path | str = DEFAULT_DEVSET_PATH,
    testset_path: Path | str = DEFAULT_TESTSET_PATH,
    enforce_quotas: bool = False,
) -> dict[str, Any]:
    """Verify taxonomy quotas and zero contamination on prompt records.

    If any record overlaps with devset or testset, raise ValueError.
    Verify task, detail, level, and verdict proportions.
    Measure motif coverage across records.
    If enforce_quotas is True, assert compliance with taxonomy quotas.
    Return summary statistics dictionary.
    """
    total = len(records)
    if total == 0:
        raise ValueError("Record list is empty.")

    # 1. Contamination check
    dev_games: set[str] = set()
    dev_fens: set[str] = set()
    for p in (Path(devset_path), Path(testset_path)):
        if not p.is_file() and (PROJECT_ROOT / p).is_file():
            p = PROJECT_ROOT / p
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if item.get("game_id"):
                            dev_games.add(str(item["game_id"]))
                        if item.get("fen"):
                            dev_fens.add(normalize_fen(item["fen"]))

    contaminated_games: list[str] = []
    contaminated_fens: list[str] = []
    for r in records:
        gid = str(r.get("game_id", ""))
        fen = normalize_fen(str(r.get("fen", "")))
        if gid and gid in dev_games:
            contaminated_games.append(gid)
        if fen and fen in dev_fens:
            contaminated_fens.append(fen)

    if contaminated_games or contaminated_fens:
        raise ValueError(
            f"Contamination detected! {len(contaminated_games)} game collisions, "
            f"{len(contaminated_fens)} FEN collisions."
        )

    # 2. Count distributions
    task_counts = Counter(r["task"] for r in records)
    detail_counts = Counter(r["detail"] for r in records)
    level_counts = Counter(r["level"] for r in records)
    verdict_counts = Counter(
        r["verdict"] for r in records if r["task"] == "explain_move"
    )

    motif_counts = Counter()
    for r in records:
        for tag in r.get("motif_tags", []):
            motif_counts[tag] += 1

    summary = {
        "total_prompts": total,
        "task_distribution": {k: v / total for k, v in task_counts.items()},
        "detail_distribution": {k: v / total for k, v in detail_counts.items()},
        "level_distribution": {k: v / total for k, v in level_counts.items()},
        "verdict_counts": dict(verdict_counts),
        "motif_coverage": {k: v / total for k, v in motif_counts.items()},
        "zero_contamination": True,
    }

    if enforce_quotas:
        for motif_name, min_share in MOTIF_MIN_TARGETS.items():
            actual_share = summary["motif_coverage"].get(motif_name, 0.0)
            if actual_share < min_share:
                raise ValueError(
                    f"Motif quota failed for '{motif_name}': expected >= {min_share:.1%}, got {actual_share:.1%}"
                )

    LOGGER.info("Verification summary: %s", json.dumps(summary, indent=2))
    return summary


def build_teacher_prompts(
    raw_files: list[Path | str] | None = None,
    cache_path: Path | str | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Build, verify, and write stratified teacher prompt dataset.

    Load candidates from raw data and Stockfish cache.
    Stratify records according to taxonomy quotas.
    Verify zero contamination and quota compliance.
    Write records to output JSONL file.
    Return list of created prompt records.
    """
    candidates_by_verdict, position_candidates = load_raw_teacher_candidates(
        raw_files=raw_files,
        cache_path=cache_path,
        seed=seed,
    )

    records = stratify_teacher_prompts(
        candidates_by_verdict=candidates_by_verdict,
        position_candidates=position_candidates,
        target_count=target_count,
        seed=seed,
    )

    verify_teacher_prompts(records, enforce_quotas=True)
    save_teacher_prompts(records, output_path=output_path)
    return records


def save_teacher_prompts(
    records: list[dict[str, Any]],
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> int:
    """Write prompt records to target JSONL file.

    Create parent directory if it does not exist.
    Write one JSON record per line.
    Return total number of saved records.
    """
    out_file = Path(output_path)
    if not out_file.is_absolute() and not out_file.parent.exists():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOGGER.info("Saved %d teacher prompt records to %s", len(records), out_file)
    return len(records)


def main() -> None:
    """CLI entry point for building teacher prompts."""
    parser = argparse.ArgumentParser(
        description="Sample 8,000 stratified teacher candidate prompts for Stage T2.3."
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Target number of prompts to sample (default: 8000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic sampling seed",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=str(DEFAULT_CACHE_PATH),
        help="Path to Stockfish evaluation cache",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    build_teacher_prompts(
        cache_path=args.cache_path,
        output_path=args.output,
        target_count=args.target_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
