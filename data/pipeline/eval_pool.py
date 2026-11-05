"""Evaluation pool generator module.

This module builds an evaluation pool of held-out positions from CC0 Lichess games.
Positions in this pool provide candidates for the frozen Phase 4 test set.
Games in this pool are reserved strictly.
No game or position in the evaluation pool may appear in SFT splits or RL prompts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import chess
import pandas as pd

from core.python import (
    classify_verdict,
)
from core.python.facts import (
    compute_phase_fact,
)

try:
    from .analyse import (
        StockfishAnalyzer,
    )
    from .rl_prompts import (
        evaluate_candidates_with_stockfish,
        format_prompt_record,
        load_excluded_keys,
        synthesize_after_from_pv,
    )
    from .splits import (
        hash_game_id,
        normalize_fen,
    )
except ImportError:
    from data.pipeline.analyse import (  # type: ignore
        StockfishAnalyzer,
    )
    from data.pipeline.rl_prompts import (  # type: ignore
        evaluate_candidates_with_stockfish,
        format_prompt_record,
        load_excluded_keys,
        synthesize_after_from_pv,
    )
    from data.pipeline.splits import (  # type: ignore
        hash_game_id,
        normalize_fen,
    )

LOGGER = logging.getLogger(__name__)
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_PUBLIC_DIR = Path("data/processed/public")
DEFAULT_TARGET_COUNT = 600
DEFAULT_TARGET_BAD = 240
DEFAULT_TARGET_GOOD = 180
DEFAULT_TARGET_POS = 90
DEFAULT_MIN_LONG = 90
DEFAULT_SEED = 42

BAD_VERDICTS = {"Inaccuracy", "Mistake", "Blunder"}
GOOD_VERDICTS = {"Good", "Best", "Only move"}


def get_avg_elo(cand: dict[str, Any]) -> float | None:
    """Calculate the average Elo rating from white and black player ratings.

    If either rating is missing or invalid, return None.
    """
    w = cand.get("white_elo")
    b = cand.get("black_elo")
    if w is not None and b is not None:
        try:
            return (float(w) + float(b)) / 2.0
        except (ValueError, TypeError):
            return None
    return None


def get_candidate_phase(fen: str) -> str:
    """Identify the game phase fact for a chess position FEN.

    Extract the phase from compute_phase_fact.
    Return 'opening', 'middlegame', or 'endgame'.
    """
    try:
        board = chess.Board(fen)
        phase_fact = compute_phase_fact(board)
        if "opening" in phase_fact:
            return "opening"
        if "endgame" in phase_fact:
            return "endgame"
        return "middlegame"
    except Exception:
        return "middlegame"


def select_bucket_items(
    candidates: list[dict[str, Any]],
    target_count: int,
    seed: int,
    reserved_games: set[str],
    reserved_fens: set[str],
) -> list[dict[str, Any]]:
    """Select candidate records for one bucket across phases and Elo bands.

    Prefer players with average Elo ratings between 800 and 2000.
    Distribute selection evenly across opening, middlegame, and endgame phases.
    Prefer unselected games to keep distinct games in the evaluation pool.
    Return the selected candidate records list.
    """
    if target_count <= 0 or not candidates:
        return []

    def sort_key(item: dict[str, Any]) -> tuple[int, float]:
        focus_tier = 0 if item.get("in_focus_elo", False) else 1
        gid = str(item.get("game_id", ""))
        fen = str(item.get("fen", ""))
        m_uci = str(item.get("move_uci", ""))
        token = f"{gid}_{fen}_{m_uci}"
        h = hash_game_id(token, seed=seed)
        return (focus_tier, h)

    sorted_candidates = sorted(candidates, key=sort_key)
    phases = ["opening", "middlegame", "endgame"]
    by_phase: dict[str, list[dict[str, Any]]] = {p: [] for p in phases}
    for c in sorted_candidates:
        p = c.get("phase", "middlegame")
        if p not in by_phase:
            p = "middlegame"
        by_phase[p].append(c)

    selected: list[dict[str, Any]] = []

    # Pass 1: Select candidates with unseen game_id and unseen normalized FEN
    phase_idx = 0
    stagnant_steps = 0
    while len(selected) < target_count and stagnant_steps < len(phases):
        p = phases[phase_idx % len(phases)]
        phase_idx += 1
        pool = by_phase[p]

        found_idx = -1
        for idx, cand in enumerate(pool):
            gid = str(cand.get("game_id", ""))
            norm_fen = normalize_fen(str(cand.get("fen", "")))
            if gid not in reserved_games and norm_fen not in reserved_fens:
                found_idx = idx
                break

        if found_idx >= 0:
            picked = pool.pop(found_idx)
            selected.append(picked)
            gid = str(picked.get("game_id", ""))
            norm_fen = normalize_fen(str(picked.get("fen", "")))
            if gid:
                reserved_games.add(gid)
            if norm_fen:
                reserved_fens.add(norm_fen)
            stagnant_steps = 0
        else:
            stagnant_steps += 1

    # Pass 2: If target is not reached, allow already reserved games with unseen FEN
    if len(selected) < target_count:
        stagnant_steps = 0
        while len(selected) < target_count and stagnant_steps < len(phases):
            p = phases[phase_idx % len(phases)]
            phase_idx += 1
            pool = by_phase[p]

            found_idx = -1
            for idx, cand in enumerate(pool):
                norm_fen = normalize_fen(str(cand.get("fen", "")))
                if norm_fen not in reserved_fens:
                    found_idx = idx
                    break

            if found_idx >= 0:
                picked = pool.pop(found_idx)
                selected.append(picked)
                gid = str(picked.get("game_id", ""))
                norm_fen = normalize_fen(str(picked.get("fen", "")))
                if gid:
                    reserved_games.add(gid)
                if norm_fen:
                    reserved_fens.add(norm_fen)
                stagnant_steps = 0
            else:
                stagnant_steps += 1

    # Pass 3: If target is still not reached, allow remaining candidates with unique (fen, move)
    if len(selected) < target_count:
        seen_keys = {(normalize_fen(str(s.get("fen", ""))), str(s.get("move_uci", ""))) for s in selected}
        stagnant_steps = 0
        while len(selected) < target_count and stagnant_steps < len(phases):
            p = phases[phase_idx % len(phases)]
            phase_idx += 1
            pool = by_phase[p]

            found_idx = -1
            for idx, cand in enumerate(pool):
                key = (normalize_fen(str(cand.get("fen", ""))), str(cand.get("move_uci", "")))
                if key not in seen_keys:
                    found_idx = idx
                    break

            if found_idx >= 0:
                picked = pool.pop(found_idx)
                selected.append(picked)
                seen_keys.add((normalize_fen(str(picked.get("fen", ""))), str(picked.get("move_uci", ""))))
                gid = str(picked.get("game_id", ""))
                norm_fen = normalize_fen(str(picked.get("fen", "")))
                if gid:
                    reserved_games.add(gid)
                if norm_fen:
                    reserved_fens.add(norm_fen)
                stagnant_steps = 0
            else:
                stagnant_steps += 1

    return selected


def assign_levels_and_details(
    items: list[dict[str, Any]],
    target_long: int,
) -> list[dict[str, Any]]:
    """Assign player level and explanation detail to a candidate list.

    Set exactly half the items to beginner and half to club.
    Set target_long items to long detail, distributed evenly.
    Ensure that long detail items also alternate beginner and club.
    Return candidate records with assigned level and detail fields.
    """
    total = len(items)
    if total == 0:
        return []

    target_long = min(target_long, total)
    long_even_count = target_long // 2
    long_odd_count = target_long - long_even_count

    even_indices = [i for i in range(total) if i % 2 == 0]
    odd_indices = [i for i in range(total) if i % 2 == 1]

    if long_even_count > len(even_indices):
        long_even_count = len(even_indices)
        long_odd_count = min(target_long - long_even_count, len(odd_indices))
    elif long_odd_count > len(odd_indices):
        long_odd_count = len(odd_indices)
        long_even_count = min(target_long - long_odd_count, len(even_indices))

    long_indices: set[int] = set()

    if long_even_count > 0 and even_indices:
        step = len(even_indices) / long_even_count
        for k in range(long_even_count):
            pick_k = min(int((k + 0.5) * step), len(even_indices) - 1)
            long_indices.add(even_indices[pick_k])

    if long_odd_count > 0 and odd_indices:
        step = len(odd_indices) / long_odd_count
        for k in range(long_odd_count):
            pick_k = min(int((k + 0.5) * step), len(odd_indices) - 1)
            long_indices.add(odd_indices[pick_k])

    # If long picks are still below target_long, fill from unused indices
    if len(long_indices) < target_long:
        for idx in range(total):
            if idx not in long_indices:
                long_indices.add(idx)
                if len(long_indices) >= target_long:
                    break

    result: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        updated = dict(it)
        updated["level"] = "beginner" if (i % 2 == 0) else "club"
        updated["detail"] = "long" if (i in long_indices) else "short"
        result.append(updated)

    return result


def build_eval_pool_from_evaluated(
    evaluated_candidates: list[dict[str, Any]],
    target_count: int = DEFAULT_TARGET_COUNT,
    target_bad: int = DEFAULT_TARGET_BAD,
    target_good: int = DEFAULT_TARGET_GOOD,
    target_position: int = DEFAULT_TARGET_POS,
    min_long: int = DEFAULT_MIN_LONG,
    seed: int = DEFAULT_SEED,
    excluded_games: set[str] | None = None,
    excluded_fens: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble balanced evaluation pool records from evaluated candidates.

    Exclude positions or games that appear in SFT splits or RL prompts.
    Balance tasks and verdicts according to parameters:
    - Inaccuracy, Mistake, or Blunder moves.
    - Good, Best, or Only move moves.
    - explain_position tasks.
    Balance player levels to 50 percent beginner and 50 percent club.
    Ensure at least min_long items receive long detail.
    Cover opening, middlegame, and endgame positions.
    Return formatted prompt records with id prefix 'eval_'.
    """
    ex_games = set(excluded_games) if excluded_games else set()
    ex_fens = set(excluded_fens) if excluded_fens else set()

    bad_candidates: list[dict[str, Any]] = []
    good_candidates: list[dict[str, Any]] = []
    pos_candidates: list[dict[str, Any]] = []

    for item in evaluated_candidates:
        gid = str(item.get("game_id", ""))
        fen = str(item.get("fen", ""))

        if gid and gid in ex_games:
            continue
        if fen and normalize_fen(fen) in ex_fens:
            continue

        eval_before = item.get("eval_before")
        if not eval_before:
            continue

        avg_elo = get_avg_elo(item)
        in_focus_elo = (800.0 <= avg_elo <= 2000.0) if avg_elo is not None else True
        phase = get_candidate_phase(fen)

        move_uci = item.get("move_uci")
        eval_after = item.get("eval_after")
        verdict = item.get("verdict")

        if move_uci and eval_after and not verdict:
            verdict = classify_verdict(eval_before, eval_after, move_uci)

        cand_data = {
            "game_id": gid,
            "fen": fen,
            "task": "explain_move",
            "move_uci": move_uci,
            "verdict": verdict,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "phase": phase,
            "in_focus_elo": in_focus_elo,
        }

        if verdict in BAD_VERDICTS and move_uci and eval_after:
            bad_candidates.append(cand_data)
        elif verdict in GOOD_VERDICTS and move_uci and eval_after:
            good_candidates.append(cand_data)

        pos_data = {
            "game_id": gid,
            "fen": fen,
            "task": "explain_position",
            "move_uci": None,
            "verdict": None,
            "eval_before": eval_before,
            "eval_after": None,
            "phase": phase,
            "in_focus_elo": in_focus_elo,
        }
        pos_candidates.append(pos_data)

    # Determine effective target counts
    sum_targets = target_bad + target_good + target_position
    if sum_targets > 0 and target_count != sum_targets:
        scale = target_count / sum_targets
        eff_bad = max(0, round(target_bad * scale))
        eff_good = max(0, round(target_good * scale))
        eff_pos = max(0, target_count - eff_bad - eff_good)
        # Adjust if rounding causes total to diverge from target_count
        cur_total = eff_bad + eff_good + eff_pos
        if cur_total < target_count:
            eff_bad += target_count - cur_total
        elif cur_total > target_count:
            diff = cur_total - target_count
            if eff_pos >= diff:
                eff_pos -= diff
            elif eff_good >= diff:
                eff_good -= diff
            else:
                eff_bad = max(0, eff_bad - diff)
    else:
        eff_bad = target_bad
        eff_good = target_good
        eff_pos = target_position

    # Synthesize good moves from eval_before PVs if bucket is short
    if len(good_candidates) < eff_good:
        seen_good = {(r["fen"], r["move_uci"]) for r in good_candidates}
        for item in evaluated_candidates:
            if len(good_candidates) >= eff_good:
                break
            gid = str(item.get("game_id", ""))
            fen = str(item.get("fen", ""))
            if (gid and gid in ex_games) or (fen and normalize_fen(fen) in ex_fens):
                continue
            eval_before = item.get("eval_before")
            if not eval_before or not eval_before.get("pvs"):
                continue
            pvs = eval_before["pvs"]
            res = synthesize_after_from_pv(fen, pvs[0])
            if not res:
                continue
            best_uci, synth_after = res
            if (fen, best_uci) in seen_good:
                continue
            v = classify_verdict(eval_before, synth_after, best_uci)
            if v in GOOD_VERDICTS:
                seen_good.add((fen, best_uci))
                avg_elo = get_avg_elo(item)
                in_focus = (800.0 <= avg_elo <= 2000.0) if avg_elo is not None else True
                good_candidates.append({
                    "game_id": gid,
                    "fen": fen,
                    "task": "explain_move",
                    "move_uci": best_uci,
                    "verdict": v,
                    "eval_before": eval_before,
                    "eval_after": synth_after,
                    "phase": get_candidate_phase(fen),
                    "in_focus_elo": in_focus,
                })

    # Synthesize bad moves from eval_before PVs if bucket is short
    if len(bad_candidates) < eff_bad:
        seen_bad = {(r["fen"], r["move_uci"]) for r in bad_candidates}
        for item in evaluated_candidates:
            if len(bad_candidates) >= eff_bad:
                break
            gid = str(item.get("game_id", ""))
            fen = str(item.get("fen", ""))
            if (gid and gid in ex_games) or (fen and normalize_fen(fen) in ex_fens):
                continue
            eval_before = item.get("eval_before")
            if not eval_before or len(eval_before.get("pvs", [])) < 2:
                continue
            pvs = eval_before["pvs"]
            for alt_pv in pvs[1:]:
                res = synthesize_after_from_pv(fen, alt_pv)
                if not res:
                    continue
                alt_uci, synth_after = res
                if (fen, alt_uci) in seen_bad:
                    continue
                v = classify_verdict(eval_before, synth_after, alt_uci)
                if v in BAD_VERDICTS:
                    seen_bad.add((fen, alt_uci))
                    avg_elo = get_avg_elo(item)
                    in_focus = (800.0 <= avg_elo <= 2000.0) if avg_elo is not None else True
                    bad_candidates.append({
                        "game_id": gid,
                        "fen": fen,
                        "task": "explain_move",
                        "move_uci": alt_uci,
                        "verdict": v,
                        "eval_before": eval_before,
                        "eval_after": synth_after,
                        "phase": get_candidate_phase(fen),
                        "in_focus_elo": in_focus,
                    })
                    break

    # Select candidates for each bucket while reserving games
    reserved_games: set[str] = set()
    reserved_fens: set[str] = set()

    selected_bad = select_bucket_items(bad_candidates, eff_bad, seed, reserved_games, reserved_fens)
    selected_good = select_bucket_items(good_candidates, eff_good, seed, reserved_games, reserved_fens)
    selected_pos = select_bucket_items(pos_candidates, eff_pos, seed, reserved_games, reserved_fens)

    # Backfill if target count is not reached
    total_selected = len(selected_bad) + len(selected_good) + len(selected_pos)
    if total_selected < target_count:
        seen_keys = {(r["fen"], r.get("move_uci")) for r in selected_bad + selected_good + selected_pos}
        for pool in (pos_candidates, good_candidates, bad_candidates):
            for cand in pool:
                key = (cand["fen"], cand.get("move_uci"))
                norm_f = normalize_fen(cand["fen"])
                if key not in seen_keys and norm_f not in ex_fens:
                    seen_keys.add(key)
                    gid = str(cand.get("game_id", ""))
                    if gid:
                        reserved_games.add(gid)
                    reserved_fens.add(norm_f)
                    selected_pos.append(cand)
                    total_selected += 1
                    if total_selected >= target_count:
                        break
            if total_selected >= target_count:
                break

    total_selected = len(selected_bad) + len(selected_good) + len(selected_pos)
    if total_selected == 0:
        return []

    # Calculate long detail requirements across tasks
    total_long = max(min_long, round(total_selected * 0.20))
    total_long = min(total_long, total_selected)
    long_bad = round(total_long * len(selected_bad) / total_selected) if total_selected > 0 else 0
    long_good = round(total_long * len(selected_good) / total_selected) if total_selected > 0 else 0
    long_pos = total_long - long_bad - long_good

    bad_items = assign_levels_and_details(selected_bad, long_bad)
    good_items = assign_levels_and_details(selected_good, long_good)
    pos_items = assign_levels_and_details(selected_pos, long_pos)

    # Interleave items across buckets for even distribution
    merged_items: list[dict[str, Any]] = []
    bucket_lists = [bad_items, good_items, pos_items]
    max_len = max(len(b) for b in bucket_lists)
    for i in range(max_len):
        for b in bucket_lists:
            if i < len(b):
                merged_items.append(b[i])

    # Build prompt records with prefix 'eval_'
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(merged_items):
        rec = format_prompt_record(
            item_id=f"eval_{idx:06d}",
            game_id=item["game_id"],
            fen=item["fen"],
            task=item["task"],
            level=item["level"],
            detail=item["detail"],
            eval_before=item["eval_before"],
            eval_after=item["eval_after"],
            move_uci=item["move_uci"],
            verdict=item["verdict"],
            source="lichess_games",
            redistributable=True,
        )
        records.append(rec)

    return records


def generate_eval_pool(
    raw_candidates: list[dict[str, Any]],
    target_count: int = DEFAULT_TARGET_COUNT,
    target_bad: int = DEFAULT_TARGET_BAD,
    target_good: int = DEFAULT_TARGET_GOOD,
    target_position: int = DEFAULT_TARGET_POS,
    min_long: int = DEFAULT_MIN_LONG,
    analyzer: StockfishAnalyzer | None = None,
    stockfish_path: str | Path | None = None,
    cache_path: Path | str | None = None,
    depth: int = 12,
    num_workers: int = 4,
    split_dir: Path | str = DEFAULT_PROCESSED_DIR,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Generate the full evaluation pool dataset.

    Filter candidates against all SFT split positions and games.
    Evaluate missing positions with Stockfish.
    Balance tasks, verdicts, player levels, and detail lengths.
    Return formatted prompt records with id prefix 'eval_'.
    """
    excluded_games, excluded_fens = load_excluded_keys(
        split_dir,
        split_names=("sft_train", "sft_val", "sft_test_pool"),
    )

    valid_candidates: list[dict[str, Any]] = []
    for cand in raw_candidates:
        gid = str(cand.get("game_id", ""))
        fen = str(cand.get("fen", ""))
        if (gid and gid in excluded_games) or (fen and normalize_fen(fen) in excluded_fens):
            continue
        valid_candidates.append(cand)

    if not valid_candidates:
        LOGGER.warning("No valid candidate positions remain for evaluation pool after exclusions.")
        return []

    needs_eval = any("eval_before" not in c for c in valid_candidates)
    if needs_eval:
        if analyzer is None:
            analyzer = StockfishAnalyzer(
                stockfish_path=stockfish_path,
                cache_path=cache_path,
                depth=depth,
                num_workers=num_workers,
            )

        max_candidates_to_eval = min(
            len(valid_candidates),
            max(int(target_count * 1.5), target_count + 200),
        )
        valid_candidates = evaluate_candidates_with_stockfish(
            candidates=valid_candidates,
            analyzer=analyzer,
            depth=depth,
            max_count=max_candidates_to_eval,
        )

    eval_items = build_eval_pool_from_evaluated(
        valid_candidates,
        target_count=target_count,
        target_bad=target_bad,
        target_good=target_good,
        target_position=target_position,
        min_long=min_long,
        seed=seed,
        excluded_games=excluded_games,
        excluded_fens=excluded_fens,
    )
    return eval_items


def save_eval_pool(
    eval_pool: list[dict[str, Any]],
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    public_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Save evaluation pool records to JSONL and Parquet files.

    Save full evaluation pool to the output directory.
    Save public evaluation pool to the public directory.
    All evaluation pool items originate from CC0 Lichess games and are redistributable.
    Return a summary statistics dictionary.
    """
    out_path = Path(output_dir)
    pub_path = Path(public_dir) if public_dir else out_path / "public"

    out_path.mkdir(parents=True, exist_ok=True)
    pub_path.mkdir(parents=True, exist_ok=True)

    jsonl_file = out_path / "eval_pool.jsonl"
    parquet_file = out_path / "eval_pool.parquet"
    pub_jsonl = pub_path / "eval_pool.jsonl"
    pub_parquet = pub_path / "eval_pool.parquet"

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for p in eval_pool:
            f.write(json.dumps(p) + "\n")

    shutil.copy2(jsonl_file, pub_jsonl)

    if eval_pool:
        df = pd.DataFrame(eval_pool)
        df.to_parquet(parquet_file, index=False)
        df.to_parquet(pub_parquet, index=False)
    else:
        empty_df = pd.DataFrame()
        empty_df.to_parquet(parquet_file, index=False)
        empty_df.to_parquet(pub_parquet, index=False)

    stats = {
        "count": len(eval_pool),
        "tasks": {},
        "levels": {},
        "details": {},
        "verdicts": {},
        "phases": {},
    }
    for p in eval_pool:
        stats["tasks"][p["task"]] = stats["tasks"].get(p["task"], 0) + 1
        stats["levels"][p["level"]] = stats["levels"].get(p["level"], 0) + 1
        stats["details"][p["detail"]] = stats["details"].get(p["detail"], 0) + 1
        if p.get("verdict"):
            stats["verdicts"][p["verdict"]] = stats["verdicts"].get(p["verdict"], 0) + 1
        ph = get_candidate_phase(p["fen"])
        stats["phases"][ph] = stats["phases"].get(ph, 0) + 1

    LOGGER.info("Saved %d evaluation pool items to %s and %s", len(eval_pool), jsonl_file, pub_jsonl)
    return stats


def main() -> None:
    """Command line entry point for generating the evaluation pool."""
    parser = argparse.ArgumentParser(description="Generate evaluation pool from Lichess games.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file of candidate games or positions.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Output directory.")
    parser.add_argument("--public-dir", type=str, default=str(DEFAULT_PUBLIC_DIR), help="Public export directory.")
    parser.add_argument("--split-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Directory with SFT splits to exclude.")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT, help="Target item count.")
    parser.add_argument("--target-bad", type=int, default=DEFAULT_TARGET_BAD, help="Target inaccuracy or worse count.")
    parser.add_argument("--target-good", type=int, default=DEFAULT_TARGET_GOOD, help="Target good or best count.")
    parser.add_argument("--target-pos", type=int, default=DEFAULT_TARGET_POS, help="Target explain_position count.")
    parser.add_argument("--min-long", type=int, default=DEFAULT_MIN_LONG, help="Minimum detail long count.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic hashing.")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish executable.")
    parser.add_argument("--cache-path", type=str, default=None, help="Stockfish cache Parquet path.")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth.")
    parser.add_argument("--workers", type=int, default=4, help="Stockfish worker count.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    candidates: list[dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    eval_pool = generate_eval_pool(
        candidates,
        target_count=args.target_count,
        target_bad=args.target_bad,
        target_good=args.target_good,
        target_position=args.target_pos,
        min_long=args.min_long,
        stockfish_path=args.stockfish_path,
        cache_path=args.cache_path,
        depth=args.depth,
        num_workers=args.workers,
        split_dir=args.split_dir,
        seed=args.seed,
    )
    save_eval_pool(eval_pool, output_dir=args.output_dir, public_dir=args.public_dir)


if __name__ == "__main__":
    main()
