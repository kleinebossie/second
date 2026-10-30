"""Stockfish analysis and persistent FEN evaluation caching module.

This module provides tools to analyze chess positions with Stockfish.
It caches engine evaluations in a Parquet file by FEN.
It enriches records with engine evaluations, facts, and model prompts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import stat
import sys
import tempfile
import threading
from typing import Any

import chess
import chess.engine
import pandas as pd

from core.python import (
    build_user_message,
    classify_verdict,
    extract_facts,
    format_qwen_chat_prompt,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CACHE_PATH = Path("data/cache/stockfish_eval_cache.parquet")
DEFAULT_DEPTH = 12
DEFAULT_WORKERS = 4
DEFAULT_FLUSH_INTERVAL = 100


def find_stockfish_binary(custom_path: str | Path | None = None) -> str:
    """Find the Stockfish executable binary on the system.

    If custom_path exists and is executable, return custom_path.
    If STOCKFISH_PATH environment variable is set, check that path.
    Search the project spikes directory for pre-compiled binaries.
    Search the system PATH for stockfish.
    If no binary is found, raise FileNotFoundError.
    """
    if custom_path:
        cand = Path(custom_path).resolve()
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
        raise FileNotFoundError(f"Specified Stockfish binary not executable: {custom_path}")

    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        cand = Path(env_path).resolve()
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)

    # Search project spikes directory
    project_root = Path(__file__).resolve().parent.parent.parent
    spikes_bin_dir = project_root / "spikes" / "colab-stockfish" / "stockfish_bin"
    if spikes_bin_dir.exists():
        for item in spikes_bin_dir.glob("**/stockfish*"):
            if item.is_file() and not item.name.endswith((".tar.gz", ".txt", ".md", ".py")):
                try:
                    item.chmod(item.stat().st_mode | stat.S_IEXEC)
                except OSError:
                    pass
                if os.access(item, os.X_OK):
                    return str(item.resolve())

    # Search PATH
    path_bin = shutil.which("stockfish")
    if path_bin:
        return path_bin

    # Search standard linux installation locations
    standard_paths = [
        Path("/usr/games/stockfish"),
        Path("/usr/bin/stockfish"),
        Path("/usr/local/bin/stockfish"),
    ]
    for sp in standard_paths:
        if sp.is_file() and os.access(sp, os.X_OK):
            return str(sp.resolve())

    raise FileNotFoundError("Stockfish executable binary not found.")


def analyze_board(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int = DEFAULT_DEPTH,
    multipv: int = 3,
) -> dict[str, Any]:
    """Run Stockfish analysis on a chess position.

    Convert scores to White perspective.
    Return evaluation dictionary with FEN and variation lines.
    """
    if board.is_game_over():
        if board.is_checkmate():
            val = -1 if board.turn == chess.WHITE else 1
            return {
                "fen": board.fen(),
                "pvs": [{"score": {"type": "mate", "value": val}, "moves": []}],
            }
        return {
            "fen": board.fen(),
            "pvs": [{"score": {"type": "cp", "value": 0}, "moves": []}],
        }

    actual_multipv = min(multipv, len(list(board.legal_moves)))
    if actual_multipv <= 0:
        actual_multipv = 1

    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=actual_multipv)
    if not isinstance(info, list):
        info = [info]

    pvs: list[dict[str, Any]] = []
    for entry in info:
        score = entry["score"].white()
        if score.is_mate():
            s_dict = {"type": "mate", "value": score.mate()}
        else:
            s_dict = {"type": "cp", "value": score.score() or 0}

        moves = [m.uci() for m in entry.get("pv", [])]
        pvs.append({"score": s_dict, "moves": moves})

    return {"fen": board.fen(), "pvs": pvs}


class StockfishCache:
    """Thread-safe persistent Parquet cache for Stockfish evaluations.

    This class stores evaluations keyed by FEN string.
    It writes atomic updates to disk in Parquet format.
    It optionally synchronizes cache files with Google Drive.
    """

    def __init__(
        self,
        cache_path: Path | str | None = None,
        drive_dir: Path | str | None = None,
    ) -> None:
        """Initialize the evaluation cache instance."""
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self.drive_dir = Path(drive_dir) if drive_dir else self._find_drive_dir()
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self.load()

    @staticmethod
    def _find_drive_dir() -> Path | None:
        """Detect Google Drive mount directory if available."""
        colab_drive = Path("/content/drive/MyDrive/second")
        if colab_drive.exists():
            return colab_drive
        user_drive = Path.home() / "Google Drive" / "My Drive" / "second"
        if user_drive.exists():
            return user_drive
        return None

    def load(self) -> None:
        """Load cached evaluations from local file or drive.

        If local file is missing, try loading from drive directory.
        If both exist, merge records into memory.
        """
        with self._lock:
            # Check drive copy if local file is missing
            if not self.cache_path.exists() and self.drive_dir:
                drive_cache = self.drive_dir / self.cache_path.name
                if drive_cache.exists():
                    try:
                        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(drive_cache, self.cache_path)
                    except OSError as err:
                        LOGGER.warning("Failed to copy cache from drive: %s", err)

            if not self.cache_path.exists():
                return

            try:
                df = pd.read_parquet(self.cache_path)
                for _, row in df.iterrows():
                    fen = str(row["fen"])
                    depth = int(row["depth"])
                    multipv = int(row["multipv"])
                    pvs_raw = row["pvs_json"]
                    pvs = json.loads(pvs_raw) if isinstance(pvs_raw, str) else pvs_raw
                    updated_at = str(row.get("updated_at", ""))
                    self._cache[fen] = {
                        "fen": fen,
                        "depth": depth,
                        "multipv": multipv,
                        "pvs": pvs,
                        "updated_at": updated_at,
                    }
                LOGGER.info("Loaded %d evaluations from cache %s", len(self._cache), self.cache_path)
            except Exception as err:
                LOGGER.error("Failed to read cache file %s: %s", self.cache_path, err)

    def get(
        self,
        fen: str,
        min_depth: int | None = None,
        min_multipv: int | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve evaluation for a position if depth and multipv suffice.

        If position is cached and meets depth and multipv, return evaluation.
        Otherwise, return None.
        """
        with self._lock:
            entry = self._cache.get(fen)
            if entry is None:
                return None

            if min_depth is not None and entry["depth"] < min_depth:
                return None

            if min_multipv is not None and entry["multipv"] < min_multipv:
                return None

            pvs = entry["pvs"]
            if min_multipv is not None and len(pvs) > min_multipv:
                pvs = pvs[:min_multipv]

            return {"fen": fen, "pvs": pvs}

    def put(
        self,
        fen: str,
        depth: int,
        multipv: int,
        pvs: list[dict[str, Any]],
    ) -> None:
        """Store evaluation into cache.

        Update existing entry if new depth is greater or equal.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            existing = self._cache.get(fen)
            if existing:
                # Keep entry with greater depth, or greater multipv if depth equal
                if depth < existing["depth"]:
                    return
                if depth == existing["depth"] and multipv <= existing["multipv"]:
                    return

            self._cache[fen] = {
                "fen": fen,
                "depth": depth,
                "multipv": multipv,
                "pvs": pvs,
                "updated_at": now_iso,
            }
            self._dirty = True

    def save(self, force: bool = False) -> None:
        """Save cached evaluations to disk in Parquet format.

        If cache is not dirty and force is False, do nothing.
        Perform atomic file write using temporary file.
        If drive directory is configured, copy file to drive.
        """
        with self._lock:
            if not self._dirty and not force:
                return

            if not self._cache:
                return

            rows = []
            for fen, entry in self._cache.items():
                rows.append({
                    "fen": fen,
                    "depth": entry["depth"],
                    "multipv": entry["multipv"],
                    "pvs_json": json.dumps(entry["pvs"]),
                    "updated_at": entry["updated_at"],
                })

            df = pd.DataFrame(rows)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write to temporary file
            with tempfile.NamedTemporaryFile(
                dir=self.cache_path.parent,
                prefix="cache_",
                suffix=".parquet",
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                df.to_parquet(tmp_path, index=False)
                tmp_path.replace(self.cache_path)
                self._dirty = False
                LOGGER.info("Saved %d evaluations to %s", len(rows), self.cache_path)
            except Exception as err:
                if tmp_path.exists():
                    tmp_path.unlink()
                LOGGER.error("Failed to save cache file: %s", err)
                raise

            # Sync to drive if available
            if self.drive_dir and self.drive_dir.exists():
                try:
                    drive_target = self.drive_dir / self.cache_path.name
                    drive_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.cache_path, drive_target)
                except OSError as err:
                    LOGGER.warning("Failed to sync cache to drive: %s", err)

    def __len__(self) -> int:
        """Return total count of cached positions."""
        with self._lock:
            return len(self._cache)


class StockfishPool:
    """Pool of persistent Stockfish engine workers.

    Each worker runs a persistent Stockfish process.
    Workers reuse processes to avoid process start overhead.
    """

    def _spawn_engine(self) -> chess.engine.SimpleEngine:
        """Spawn and configure a single Stockfish engine instance."""
        engine = chess.engine.SimpleEngine.popen_uci(self.binary_path)
        try:
            engine.configure({"Threads": 1, "Hash": 16})
        except Exception:
            pass
        self._all_engines.append(engine)
        return engine

    def __init__(
        self,
        binary_path: str,
        num_workers: int = DEFAULT_WORKERS,
    ) -> None:
        """Initialize worker engines in the pool."""
        self.binary_path = binary_path
        self.num_workers = max(1, num_workers)
        self._engines: queue.Queue[chess.engine.SimpleEngine] = queue.Queue()
        self._all_engines: list[chess.engine.SimpleEngine] = []

        for _ in range(self.num_workers):
            self._engines.put(self._spawn_engine())

    def close(self) -> None:
        """Close all engine processes in the pool."""
        for engine in self._all_engines:
            try:
                engine.quit()
            except Exception:
                pass
        self._all_engines.clear()

    def __enter__(self) -> StockfishPool:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close engine processes."""
        self.close()

    def analyze_fen(
        self,
        fen: str,
        depth: int = DEFAULT_DEPTH,
        multipv: int = 3,
    ) -> dict[str, Any]:
        """Analyze one position using an available engine worker.

        Borrow engine worker from queue.
        If the engine process stops or the board position is invalid, recover and return a fallback evaluation.
        Return healthy engine worker to queue after analysis completes.
        """
        try:
            board = chess.Board(fen)
        except ValueError:
            return {"fen": fen, "pvs": [{"score": {"type": "cp", "value": 0}, "moves": []}]}

        if not board.is_valid():
            return {"fen": fen, "pvs": [{"score": {"type": "cp", "value": 0}, "moves": []}]}

        engine = self._engines.get()
        try:
            return analyze_board(engine, board, depth=depth, multipv=multipv)
        except Exception as exc:
            LOGGER.warning("Engine failed on FEN %s (%s); respawning worker.", fen, exc)
            try:
                engine.quit()
            except Exception:
                pass
            engine = self._spawn_engine()
            try:
                return analyze_board(engine, board, depth=depth, multipv=multipv)
            except Exception:
                try:
                    engine.quit()
                except Exception:
                    pass
                engine = self._spawn_engine()
                return {"fen": fen, "pvs": [{"score": {"type": "cp", "value": 0}, "moves": []}]}
        finally:
            self._engines.put(engine)


class StockfishAnalyzer:
    """High-level analyzer supporting parallel analysis and Parquet caching.

    This class coordinates Stockfish pool workers and the Parquet cache.
    """

    def __init__(
        self,
        stockfish_path: str | Path | None = None,
        cache_path: Path | str | None = None,
        drive_dir: Path | str | None = None,
        depth: int = DEFAULT_DEPTH,
        num_workers: int = DEFAULT_WORKERS,
    ) -> None:
        """Initialize the analyzer with binary and cache settings."""
        self.binary_path = find_stockfish_binary(stockfish_path)
        self.cache = StockfishCache(cache_path=cache_path, drive_dir=drive_dir)
        self.depth = depth
        self.num_workers = num_workers

    def analyze_fens(
        self,
        fens_with_multipv: list[tuple[str, int]],
        flush_interval: int = DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        """Analyze missing positions and store results in cache.

        If a position already exists at required depth and multipv, skip it.
        Run missing evaluations in parallel using engine pool.
        Flush new results to disk at regular intervals.
        """
        missing: list[tuple[str, int]] = []
        for fen, mp in fens_with_multipv:
            if self.cache.get(fen, min_depth=self.depth, min_multipv=mp) is None:
                missing.append((fen, mp))

        if not missing:
            return

        # Deduplicate missing positions by keeping maximum requested multipv
        max_mp_map: dict[str, int] = {}
        for fen, mp in missing:
            if fen not in max_mp_map or mp > max_mp_map[fen]:
                max_mp_map[fen] = mp

        unique_missing = list(max_mp_map.items())
        LOGGER.info(
            "Analyzing %d missing positions at depth %d with %d workers",
            len(unique_missing),
            self.depth,
            self.num_workers,
        )

        with StockfishPool(self.binary_path, num_workers=self.num_workers) as pool:
            processed_since_flush = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_item = {
                    executor.submit(pool.analyze_fen, fen, self.depth, mp): (fen, mp)
                    for fen, mp in unique_missing
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    fen, mp = future_to_item[future]
                    eval_result = future.result()
                    self.cache.put(fen, self.depth, mp, eval_result["pvs"])
                    processed_since_flush += 1

                    if processed_since_flush >= flush_interval:
                        self.cache.save()
                        processed_since_flush = 0

            self.cache.save(force=True)


def enrich_record(
    record: dict[str, Any],
    cache: StockfishCache,
    depth: int = DEFAULT_DEPTH,
) -> dict[str, Any]:
    """Populate factual annotations and prompt text for a record.

    Retrieve evaluations from cache.
    Extract facts and classify verdict using Phase 1 core logic.
    Build structured user message and Qwen chat prompt.
    """
    task = record.get("task") or "explain_move"
    level = record.get("level") or "club"
    detail = record.get("detail") or "short"
    record["task"] = task
    record["level"] = level
    record["detail"] = detail
    fen = record["fen"]
    move_uci = record.get("move_uci")

    eval_before = cache.get(fen, min_depth=depth, min_multipv=3)
    if eval_before is None:
        raise ValueError(f"Missing eval_before in cache for FEN: {fen}")

    eval_after: dict[str, Any] | None = None
    verdict: str | None = None

    if task == "explain_move":
        if not move_uci:
            raise ValueError("move_uci is required for task explain_move")

        board = chess.Board(fen)
        move_obj = chess.Move.from_uci(move_uci)
        board.push(move_obj)
        fen_after = board.fen()

        eval_after = cache.get(fen_after, min_depth=depth, min_multipv=1)
        if eval_after is None:
            raise ValueError(f"Missing eval_after in cache for FEN: {fen_after}")

        verdict = classify_verdict(eval_before, eval_after, move_uci)
        record["verdict"] = verdict
        record["eval_after"] = eval_after
    else:
        record["eval_after"] = None

    record["eval_before"] = eval_before

    facts = extract_facts(
        task=task,
        fen=fen,
        eval_before=eval_before,
        eval_after=eval_after,
        move_uci=move_uci,
    )
    record["facts"] = facts

    user_message = build_user_message(
        task=task,
        level=level,
        detail=detail,
        fen=fen,
        eval_before=eval_before,
        eval_after=eval_after,
        move_uci=move_uci,
    )
    record["user_message"] = user_message
    record["chat_prompt"] = format_qwen_chat_prompt(user_message)

    return record


def analyse_records(
    records: list[dict[str, Any]],
    cache: StockfishCache | None = None,
    stockfish_path: str | Path | None = None,
    depth: int = DEFAULT_DEPTH,
    num_workers: int = DEFAULT_WORKERS,
    flush_interval: int = DEFAULT_FLUSH_INTERVAL,
    drive_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Run end-to-end evaluation and enrichment on a list of records.

    Collect required FEN positions from all records.
    Check cache and evaluate missing positions in parallel.
    Enrich each record with evaluations, facts, and prompt text.
    """
    if cache is None:
        cache = StockfishCache(drive_dir=drive_dir)

    analyzer = StockfishAnalyzer(
        stockfish_path=stockfish_path,
        cache_path=cache.cache_path,
        drive_dir=drive_dir,
        depth=depth,
        num_workers=num_workers,
    )
    # Share loaded cache
    analyzer.cache = cache

    # Step 1: Analyze all before-move FENs at MultiPV=3
    before_fens: list[tuple[str, int]] = [(r["fen"], 3) for r in records]
    analyzer.analyze_fens(before_fens, flush_interval=flush_interval)

    # Step 2: Check which explain_move records need an additional after-move evaluation
    after_fens: list[tuple[str, int]] = []
    for r in records:
        task = r.get("task", "explain_move")
        if task == "explain_move":
            move_uci = r.get("move_uci")
            if move_uci:
                fen = r["fen"]
                board = chess.Board(fen)
                board.push(chess.Move.from_uci(move_uci))
                fen_after = board.fen()
                if cache.get(fen_after, min_depth=depth, min_multipv=1) is None:
                    # Check if move_uci is already the first move of a PV in eval_before with a reply
                    eb = cache.get(fen, min_depth=depth, min_multipv=3)
                    matched_pv = None
                    if eb:
                        for pv in eb.get("pvs", []):
                            if pv.get("moves") and pv["moves"][0] == move_uci and len(pv["moves"]) >= 2:
                                matched_pv = pv
                                break
                    if matched_pv is not None:
                        cache.put(
                            fen_after,
                            depth,
                            1,
                            [{"score": matched_pv["score"], "moves": matched_pv["moves"][1:]}],
                        )
                    else:
                        after_fens.append((fen_after, 1))

    if after_fens:
        analyzer.analyze_fens(after_fens, flush_interval=flush_interval)
    else:
        cache.save()

    enriched: list[dict[str, Any]] = []
    for r in records:
        enriched.append(enrich_record(r, cache, depth=depth))

    return enriched


def main() -> None:
    """Command line entry point for Stockfish analysis."""
    parser = argparse.ArgumentParser(description="Analyze chess positions with Stockfish and Parquet cache.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with chess records.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for enriched records.")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish executable.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH), help="Path to Parquet cache file.")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive sync directory.")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Stockfish search depth.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of concurrent workers.")
    parser.add_argument("--flush-interval", type=int, default=DEFAULT_FLUSH_INTERVAL, help="Cache flush interval.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_path = Path(args.input)
    if not input_path.exists():
        LOGGER.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    records: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    cache = StockfishCache(cache_path=args.cache_path, drive_dir=args.drive_dir)
    enriched = analyse_records(
        records,
        cache=cache,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        num_workers=args.workers,
        flush_interval=args.flush_interval,
        drive_dir=args.drive_dir,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in enriched:
            f.write(json.dumps(rec) + "\n")

    LOGGER.info("Enriched %d records and wrote to %s", len(enriched), output_path)


if __name__ == "__main__":
    main()
