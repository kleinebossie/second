"""Dataset publication and Google Drive synchronization module.

This module prepares clean dataset export packages for Hugging Face Hub.
It includes only redistributable open-license partitions.
It generates local rebuild instructions for proprietary sources.
It uploads packages to Hugging Face Hub when authenticated.
It synchronizes processed dataset files and evaluation caches to Google Drive.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any

LOGGER = logging.getLogger(__name__)
DEFAULT_PUBLIC_DIR = Path("data/processed/public")
DEFAULT_EXPORT_DIR = Path("data/processed/hf_export")
DEFAULT_CACHE_FILE = Path("data/cache/stockfish_eval_cache.parquet")
DEFAULT_DATASET_CARD = Path("data/cards/README.md")
DEFAULT_REPO_ID = "second-chess/second-data-1.0"


def find_default_drive_dir() -> Path | None:
    """Find the default Google Drive synchronization directory if mounted.

    Check standard Google Colab mount path first.
    Check standard local Google Drive desktop path next.
    If no directory exists, return None.
    """
    colab_path = Path("/content/drive/MyDrive/second/data")
    if colab_path.parent.exists():
        return colab_path

    local_path = Path.home() / "Google Drive" / "My Drive" / "second" / "data"
    if local_path.parent.exists():
        return local_path

    return None


def prepare_hf_export(
    public_dir: Path | str = DEFAULT_PUBLIC_DIR,
    export_dir: Path | str = DEFAULT_EXPORT_DIR,
    card_path: Path | str | None = None,
) -> Path:
    """Prepare the export directory containing only redistributable files.

    Copy redistributable dataset partitions from the public directory.
    Copy or write the dataset card README file.
    Create rebuild instructions for proprietary data sources.
    Create git-lfs tracking attributes for parquet files.
    """
    pub_path = Path(public_dir)
    exp_path = Path(export_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    # Copy public dataset files
    expected_files = [
        "sft_train.parquet",
        "sft_train.jsonl",
        "sft_val.parquet",
        "sft_val.jsonl",
        "sft_test_pool.parquet",
        "sft_test_pool.jsonl",
        "rl_prompts.parquet",
        "rl_prompts.jsonl",
    ]

    copied_count = 0
    for name in expected_files:
        src = pub_path / name
        if src.exists():
            dst = exp_path / name
            shutil.copy2(src, dst)
            copied_count += 1

    LOGGER.info("Copied %d public data files to export directory %s", copied_count, exp_path)

    # Copy dataset card README
    card_source = Path(card_path) if card_path else DEFAULT_DATASET_CARD
    readme_dest = exp_path / "README.md"
    if card_source.exists():
        shutil.copy2(card_source, readme_dest)
    else:
        # Create minimal fallback README if card file is missing
        readme_dest.write_text(
            "# Second Chess Dataset (second-data-1.0)\n\n"
            "This repository contains public CC0 and MIT partitions for the Second project.\n",
            encoding="utf-8",
        )

    # Write REBUILD.md instructions
    rebuild_dest = exp_path / "REBUILD.md"
    rebuild_text = (
        "# Rebuild Instructions for Proprietary Data Sources\n\n"
        "This public repository excludes proprietary human commentary.\n"
        "Excluded sources include GameKnot commentary and standard Lichess studies.\n\n"
        "## Rebuild Steps\n\n"
        "1. Clone repository: `git clone https://github.com/second-chess/second.git`\n"
        "2. Install packages: `pip install -e .`\n"
        "3. Run GameKnot scraper: `python -m data.sources.gameknot`\n"
        "4. Run Lichess study parser: `python -m data.sources.lichess_studies`\n"
        "5. Run full pipeline: `python -m data.pipeline.run_pipeline`\n"
    )
    rebuild_dest.write_text(rebuild_text, encoding="utf-8")

    # Write .gitattributes for Git LFS
    gitattributes_dest = exp_path / ".gitattributes"
    gitattributes_text = "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
    gitattributes_dest.write_text(gitattributes_text, encoding="utf-8")

    return exp_path


def upload_to_hf(
    export_dir: Path | str = DEFAULT_EXPORT_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
    private: bool = False,
) -> dict[str, Any]:
    """Upload prepared export files to Hugging Face Hub.

    If no authentication token exists, report status and stop.
    Create dataset repository if it does not exist.
    Upload all files from the export directory.
    """
    auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        LOGGER.error("huggingface_hub package is not installed.")
        return {"uploaded": False, "reason": "huggingface_hub package is missing."}

    exp_path = Path(export_dir)
    if not exp_path.exists():
        return {"uploaded": False, "reason": f"Export directory not found: {exp_path}"}

    try:
        api = HfApi(token=auth_token)
        # Check login status
        user_info = api.whoami()
        LOGGER.info("Authenticated to Hugging Face as user: %s", user_info.get("name", "unknown"))
    except Exception as auth_err:
        LOGGER.warning("Hugging Face authentication not configured: %s", auth_err)
        return {
            "uploaded": False,
            "reason": "Not authenticated. Set HF_TOKEN environment variable to upload.",
        }

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(exp_path),
            repo_id=repo_id,
            repo_type="dataset",
        )
        LOGGER.info("Uploaded export files to %s on Hugging Face Hub", repo_id)
        return {"uploaded": True, "repo_id": repo_id}
    except Exception as err:
        LOGGER.error("Failed to upload dataset to Hugging Face: %s", err)
        return {"uploaded": False, "reason": str(err)}


def sync_to_drive(
    source_dir: Path | str = DEFAULT_PUBLIC_DIR.parent,
    drive_dir: Path | str | None = None,
    cache_path: Path | str | None = DEFAULT_CACHE_FILE,
) -> dict[str, Any]:
    """Synchronize processed dataset files and evaluation cache to Google Drive.

    If target drive directory does not exist, return failure status.
    Copy dataset files and cache to drive directory.
    Skip copying files when destination is newer or identical.
    """
    target_drive = Path(drive_dir) if drive_dir else find_default_drive_dir()
    if not target_drive:
        return {"synced": False, "reason": "Google Drive destination directory not found."}

    src_path = Path(source_dir)
    if not src_path.exists():
        return {"synced": False, "reason": f"Source directory not found: {src_path}"}

    target_drive.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []

    # Sync processed split files
    for src_file in src_path.glob("**/*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(src_path)
            dest_file = target_drive / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy if destination missing or source newer
            if not dest_file.exists() or src_file.stat().st_mtime > dest_file.stat().st_mtime:
                shutil.copy2(src_file, dest_file)
                copied_files.append(str(rel_path))

    # Sync cache file
    if cache_path:
        c_path = Path(cache_path)
        if c_path.exists():
            dest_cache = target_drive / "cache" / c_path.name
            dest_cache.parent.mkdir(parents=True, exist_ok=True)
            if not dest_cache.exists() or c_path.stat().st_mtime > dest_cache.stat().st_mtime:
                shutil.copy2(c_path, dest_cache)
                copied_files.append(f"cache/{c_path.name}")

    LOGGER.info("Synced %d files to Google Drive at %s", len(copied_files), target_drive)
    return {
        "synced": True,
        "files_copied": len(copied_files),
        "destination": str(target_drive),
    }


def main() -> None:
    """Command line entry point for publishing and syncing datasets."""
    parser = argparse.ArgumentParser(description="Export, publish, and sync dataset files.")
    parser.add_argument("--public-dir", type=str, default=str(DEFAULT_PUBLIC_DIR), help="Public processed directory.")
    parser.add_argument("--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR), help="Hugging Face export folder.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face repository ID.")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive sync directory.")
    parser.add_argument("--upload", action="store_true", help="Upload export folder to Hugging Face Hub.")
    parser.add_argument("--sync-drive", action="store_true", help="Synchronize files to Google Drive.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    exp_dir = prepare_hf_export(public_dir=args.public_dir, export_dir=args.export_dir)

    if args.upload:
        status = upload_to_hf(export_dir=exp_dir, repo_id=args.repo_id)
        if not status["uploaded"]:
            LOGGER.warning("Hugging Face upload skipped: %s", status.get("reason"))

    if args.sync_drive:
        sync_status = sync_to_drive(drive_dir=args.drive_dir)
        if not sync_status["synced"]:
            LOGGER.warning("Google Drive sync skipped: %s", sync_status.get("reason"))


if __name__ == "__main__":
    main()
