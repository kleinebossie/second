#!/usr/bin/env python3
"""
Stockfish Colab Benchmark (Phase 0.4)
Analyzes 100 diverse chess positions at fixed depth 18 with single-PV and MultiPV=3.
Measures total time, time per position, and projected time for 10k and 50k positions.
"""

import argparse
import os
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    from positions import BENCHMARK_POSITIONS
except ImportError:
    from spikes.colab_stockfish.positions import BENCHMARK_POSITIONS

def _find_binary_in_dir(directory: Path) -> str:
    """Find and return the first executable stockfish binary in directory."""
    ignored_exts = (".tar.gz", ".txt", ".md")
    for b in directory.glob("**/stockfish*"):
        if b.is_file() and not b.name.endswith(ignored_exts):
            b.chmod(b.stat().st_mode | stat.S_IEXEC)
            return str(b)
    return ""


def find_or_download_stockfish(custom_path: str = None) -> str:
    """Find local Stockfish binary or download latest official universal release."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Check system PATH
    sys_path = subprocess.run(["which", "stockfish"], capture_output=True, text=True).stdout.strip()
    if sys_path and os.path.exists(sys_path):
        return sys_path

    # Check local spikes/colab-stockfish/stockfish_bin/ directory for existing binary
    bin_dir = Path(__file__).resolve().parent / "stockfish_bin"
    if bin_dir.exists():
        found = _find_binary_in_dir(bin_dir)
        if found:
            return found

    print("Stockfish not found on PATH. Attempting automatic download from GitHub releases...")
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-linux-x86-64-universal.tar.gz"
    tar_path = bin_dir / "sf.tar.gz"

    try:
        subprocess.run(["curl", "-L", "-s", url, "-o", str(tar_path)], check=True)
        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(bin_dir)], check=True)
        
        found = _find_binary_in_dir(bin_dir)
        if found:
            return found
    except Exception as exc:
        print(f"Direct download failed: {exc}")

    # Fallback to apt-get on Debian/Ubuntu
    try:
        print("Trying apt-get install stockfish...")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "stockfish"], check=True)
        sys_path = subprocess.run(["which", "stockfish"], capture_output=True, text=True).stdout.strip()
        if sys_path:
            return sys_path
    except Exception as exc:
        print(f"apt-get fallback failed: {exc}")

    raise FileNotFoundError("Could not find or install Stockfish. Please specify --stockfish-path.")


def run_benchmark_pass(
    stockfish_path: str,
    positions: List[str],
    depth: int,
    multipv: int,
    threads: int
) -> Dict[str, Any]:
    """Execute analysis over positions with given engine settings."""
    try:
        import chess
        import chess.engine
    except ImportError:
        print("python-chess not installed. Attempting installation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess"], check=True)
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "python-chess"], check=True)
        import chess
        import chess.engine

    print(f"\n--- Running Benchmark (Depth={depth}, MultiPV={multipv}, Threads={threads}, Positions={len(positions)}) ---")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": threads, "Hash": 128})

    times = []
    total_nodes = 0
    t_start = time.time()

    for idx, fen in enumerate(positions):
        board = chess.Board(fen)
        p_start = time.time()
        
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        p_duration = time.time() - p_start
        times.append(p_duration)

        # Capture true search nodes from engine analysis
        if isinstance(info, list):
            nodes = max((item.get("nodes", 0) for item in info), default=0)
        else:
            nodes = info.get("nodes", 0)
        total_nodes += nodes

        if (idx + 1) % 25 == 0 or (idx + 1) == len(positions):
            print(f"Processed {idx + 1}/{len(positions)} positions... (Current avg: {sum(times)/len(times):.3f}s/pos)")

    total_time = time.time() - t_start
    engine.quit()

    avg_time = sum(times) / len(times)
    median_time = statistics.median(times)
    nps = total_nodes / total_time if total_time > 0 else 0

    # Project time for large datasets (Phase 2.2 needs)
    time_10k_hrs = (avg_time * 10_000) / 3600.0
    time_50k_hrs = (avg_time * 50_000) / 3600.0

    return {
        "depth": depth,
        "multipv": multipv,
        "threads": threads,
        "count": len(positions),
        "total_time_s": total_time,
        "avg_time_s": avg_time,
        "median_time_s": median_time,
        "total_nodes": total_nodes,
        "nps": nps,
        "est_10k_hrs": time_10k_hrs,
        "est_50k_hrs": time_50k_hrs,
    }


def main():
    parser = argparse.ArgumentParser(description="Stockfish Colab Benchmark (Phase 0.4)")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish binary")
    parser.add_argument("--depth", type=int, default=18, help="Search depth (default: 18)")
    parser.add_argument("--positions", type=int, default=100, help="Number of positions (max: 100, default: 100)")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for engine (default: 1)")
    parser.add_argument("--multipv", choices=["1", "3", "both"], default="both", help="MultiPV setting")
    args = parser.parse_args()

    stockfish_bin = find_or_download_stockfish(args.stockfish_path)
    print(f"Using Stockfish binary: {stockfish_bin}")

    # Query engine version
    ver_out = subprocess.run([stockfish_bin, "uci"], capture_output=True, text=True, timeout=5).stdout
    for line in ver_out.splitlines():
        if "id name" in line:
            print(f"Engine identification: {line.strip()}")
            break

    num_pos = min(args.positions, len(BENCHMARK_POSITIONS))
    positions = BENCHMARK_POSITIONS[:num_pos]

    results = []
    if args.multipv in ["1", "both"]:
        res_single = run_benchmark_pass(
            stockfish_path=stockfish_bin,
            positions=positions,
            depth=args.depth,
            multipv=1,
            threads=args.threads
        )
        results.append(res_single)

    if args.multipv in ["3", "both"]:
        res_multi = run_benchmark_pass(
            stockfish_path=stockfish_bin,
            positions=positions,
            depth=args.depth,
            multipv=3,
            threads=args.threads
        )
        results.append(res_multi)

    print("\n" + "=" * 60)
    print("PHASE 0.4 STOCKFISH BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Platform: Linux | Stockfish Depth: {args.depth} | Threads: {args.threads}")
    print(f"Test Positions: {num_pos} (30% opening, 40% middlegame, 30% endgame)\n")

    for r in results:
        pv_name = "Single-PV (1 line)" if r["multipv"] == 1 else "MultiPV=3 (3 lines)"
        print(f"Mode: {pv_name}")
        print(f"  Total time ({r['count']} pos):  {r['total_time_s']:.2f} s")
        print(f"  Avg time per pos:     {r['avg_time_s']:.3f} s")
        print(f"  Median time per pos:  {r['median_time_s']:.3f} s")
        print(f"  Nodes per second:     {r['nps']:,.0f}")
        print(f"  Projected 10k pos:    {r['est_10k_hrs']:.2f} hours")
        print(f"  Projected 50k pos:    {r['est_50k_hrs']:.2f} hours\n")

    print("\nMarkdown Table for docs/decisions/compute.md:")
    print("-" * 50)
    print("| Metric | Single-PV (Depth 18) | MultiPV=3 (Depth 18) |")
    print("|---|---|---|")
    if len(results) == 2:
        single_res, multi_res = results[0], results[1]
        print(f"| Avg time / position | {single_res['avg_time_s']:.3f} s | {multi_res['avg_time_s']:.3f} s |")
        print(f"| {single_res['count']} positions total | {single_res['total_time_s']:.2f} s | {multi_res['total_time_s']:.2f} s |")
        print(f"| Projected 10k positions | {single_res['est_10k_hrs']:.2f} hours | {multi_res['est_10k_hrs']:.2f} hours |")
        print(f"| Projected 50k positions | {single_res['est_50k_hrs']:.2f} hours | {multi_res['est_50k_hrs']:.2f} hours |")
    elif len(results) == 1:
        res = results[0]
        mode = "Single-PV" if res["multipv"] == 1 else "MultiPV=3"
        print(f"| Mode | {mode} |")
        print(f"| Avg time / position | {res['avg_time_s']:.3f} s |")
        print(f"| Projected 10k positions | {res['est_10k_hrs']:.2f} hours |")
        print(f"| Projected 50k positions | {res['est_50k_hrs']:.2f} hours |")
    print("-" * 50)

if __name__ == "__main__":
    main()
