"""Build archive bundle for Google Colab SFT training.

This script packages dataset splits, frozen test sets, baselines, and code.
It creates a compressed zip archive at data/processed/phase5_colab_bundle.zip.
Google Colab loads this bundle to execute Phase 5 training and evaluation.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import zipfile

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_ZIP = Path("data/processed/phase5_colab_bundle.zip")

REQUIRED_DATA_FILES: list[str] = [
    "data/processed/sft_train.jsonl",
    "data/processed/sft_val.jsonl",
    "eval/testset-1.0.jsonl",
    "eval/results/baselines/base-qwen-2b.jsonl",
]

OPTIONAL_CONFIG_FILES: list[str] = [
    "training/configs/second-1.0-2b.yaml",
]

CODE_DIRECTORIES: list[str] = [
    "core/python",
    "rewards",
    "eval",
    "training",
]


def collect_bundle_files(repo_root: Path) -> list[tuple[Path, str]]:
    """Find all files needed for the Colab training bundle.

    Collect data splits, baseline records, configuration files, and Python source code.
    Exclude cache files and bytecode.
    Return list of tuples with absolute source path and relative archive path.
    """
    entries: list[tuple[Path, str]] = []

    # 1. Collect required data files
    for rel_path in REQUIRED_DATA_FILES:
        full_path = repo_root / rel_path
        if full_path.is_file():
            entries.append((full_path, rel_path))
        else:
            LOGGER.warning("Required data file not found: %s", full_path)

    # 2. Collect optional configs
    for rel_path in OPTIONAL_CONFIG_FILES:
        full_path = repo_root / rel_path
        if full_path.is_file():
            entries.append((full_path, rel_path))

    # 3. Collect Python source files
    for dir_rel in CODE_DIRECTORIES:
        source_dir = repo_root / dir_rel
        if not source_dir.is_dir():
            continue
        for file_path in source_dir.rglob("*.py"):
            if "__pycache__" in file_path.parts:
                continue
            archive_path = str(file_path.relative_to(repo_root))
            entries.append((file_path, archive_path))

    return entries


def build_colab_bundle(
    output_path: str | Path = DEFAULT_OUTPUT_ZIP,
    repo_root: str | Path | None = None,
) -> Path:
    """Create the Colab zip archive.

    If parent directory does not exist, create the directory.
    Compress all selected files using zipfile.ZIP_DEFLATED.
    Return path to created zip archive.
    """
    out_file = Path(output_path).resolve()
    root = Path(repo_root).resolve() if repo_root else Path.cwd()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    file_entries = collect_bundle_files(root)

    LOGGER.info("Packaging %d files into %s...", len(file_entries), out_file)
    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path, arc_path in file_entries:
            zf.write(src_path, arcname=arc_path)

    size_mb = out_file.stat().st_size / (1024 * 1024)
    LOGGER.info("Bundle created successfully. Total size: %.2f MB", size_mb)
    return out_file


def main() -> None:
    """CLI entry point for building the Colab bundle."""
    parser = argparse.ArgumentParser(description="Build Phase 5 Colab bundle zip.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_ZIP),
        help="Target zip file path",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    bundle_path = build_colab_bundle(output_path=args.output)
    print(f"Colab bundle built: {bundle_path} ({bundle_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
