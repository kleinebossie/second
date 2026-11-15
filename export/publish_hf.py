"""Hugging Face Hub publication module for Second-1.0-2B models and GGUFs.

This module uploads merged safetensors models to second-chess/Second-1.0-2B.
It uploads quantized GGUF models and shards to second-chess/Second-1.0-2B-GGUF.
It creates destination model repositories on Hugging Face Hub if they do not exist.
It attaches model cards and metadata tags.
All docstrings follow ASD-STE100 Simplified Technical English.
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

DEFAULT_BASE_REPO_ID = "joebos/Second-1.0-2B"
DEFAULT_GGUF_REPO_ID = "joebos/Second-1.0-2B-GGUF"

DEFAULT_MERGED_DIR = Path("export/merged_safetensors")
DEFAULT_GGUF_DIR = Path("export/gguf")
DEFAULT_MODEL_CARD_BASE = Path("export/model_cards/Second-1.0-2B/README.md")
DEFAULT_MODEL_CARD_GGUF = Path("export/model_cards/Second-1.0-2B-GGUF/README.md")


def get_hf_token() -> str | None:
    """Retrieve Hugging Face API token from environment variables.

    Check HF_TOKEN first.
    Check HUGGING_FACE_HUB_TOKEN next.
    If no token is set, return None.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token.strip():
        return token.strip()
    return None


def prepare_repo_upload_folder(
    source_dir: Path | str,
    model_card_path: Path | str,
    staging_dir: Path | str,
    include_extensions: tuple[str, ...] | None = None,
) -> Path:
    """Prepare a clean staging directory with model assets and model card.

    Create staging directory.
    Copy matching files from source directory.
    Copy model card to README.md in staging directory.
    Return path to staging directory.
    """
    src = Path(source_dir)
    card = Path(model_card_path)
    stage = Path(staging_dir)
    stage.mkdir(parents=True, exist_ok=True)

    if not card.is_file():
        raise FileNotFoundError(f"Model card not found at {card}")

    shutil.copy2(card, stage / "README.md")

    if src.is_dir():
        for file_path in src.iterdir():
            if file_path.is_file():
                if include_extensions and file_path.suffix not in include_extensions:
                    continue
                shutil.copy2(file_path, stage / file_path.name)

    return stage


def publish_model_to_hf(
    repo_id: str,
    folder_path: Path | str,
    commit_message: str,
    token: str | None = None,
    dry_run: bool = False,
    model_card_path: Path | str | None = None,
    subfolders: list[tuple[Path | str, str]] | None = None,
) -> dict[str, Any]:
    """Upload folder contents, optional model card, and subfolders to Hugging Face Hub.

    Check for API token.
    Create repository if missing.
    Upload folder with provided commit message.
    If model_card_path is provided, upload it as README.md.
    If subfolders are provided, upload each into its designated path_in_repo.
    Return summary metadata dictionary.
    """
    f_path = Path(folder_path)
    auth_token = token or get_hf_token()

    if dry_run:
        LOGGER.info("Dry-run mode active. Simulating upload of '%s' to '%s'...", f_path, repo_id)
        files = [p.name for p in f_path.iterdir()] if f_path.is_dir() else []
        if model_card_path and Path(model_card_path).is_file():
            if "README.md" not in files:
                files.append("README.md")
        subfolder_info = {}
        if subfolders:
            for s_path, repo_sub in subfolders:
                sp = Path(s_path)
                sub_files = [p.name for p in sp.iterdir()] if sp.is_dir() else []
                subfolder_info[repo_sub] = sub_files
        return {
            "repo_id": repo_id,
            "folder": str(f_path),
            "files_count": len(files),
            "files": files,
            "subfolders": subfolder_info,
            "commit_message": commit_message,
            "status": "simulated",
            "url": f"https://huggingface.co/{repo_id}",
        }

    if not auth_token:
        raise ValueError(
            "Missing Hugging Face token. Set the HF_TOKEN environment variable "
            "or pass --token to authenticate."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub package is required for publication.") from exc

    api = HfApi(token=auth_token)

    LOGGER.info("Creating or verifying model repository '%s'...", repo_id)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
    except Exception as exc:
        if "403" in str(exc) and "/" in repo_id:
            try:
                user_info = api.whoami()
                username = user_info.get("name")
                orgs = [o.get("name") for o in user_info.get("orgs", [])]
                namespace, model_name = repo_id.split("/", 1)
                if username and namespace != username and namespace not in orgs:
                    fallback_repo = f"{username}/{model_name}"
                    LOGGER.warning(
                        "Namespace '%s' not found in your HF organizations %s. Falling back to your personal namespace: '%s'",
                        namespace,
                        orgs,
                        fallback_repo,
                    )
                    repo_id = fallback_repo
                    api.create_repo(
                        repo_id=repo_id,
                        repo_type="model",
                        exist_ok=True,
                        private=False,
                    )
                else:
                    raise
            except Exception:
                raise exc
        else:
            raise

    LOGGER.info("Uploading folder '%s' to repository '%s'...", f_path, repo_id)
    api.upload_folder(
        folder_path=str(f_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    if model_card_path and Path(model_card_path).is_file():
        LOGGER.info("Uploading model card '%s' as README.md...", model_card_path)
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Attach model card README.md",
        )

    if subfolders:
        for s_path, repo_sub in subfolders:
            sp = Path(s_path)
            if sp.is_dir():
                LOGGER.info("Uploading subfolder '%s' to '%s' in repository '%s'...", sp, repo_sub, repo_id)
                api.upload_folder(
                    folder_path=str(sp),
                    path_in_repo=repo_sub,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {repo_sub} files",
                )

    LOGGER.info("Publication completed: https://huggingface.co/%s", repo_id)
    return {
        "repo_id": repo_id,
        "folder": str(f_path),
        "commit_message": commit_message,
        "status": "uploaded",
        "url": f"https://huggingface.co/{repo_id}",
    }


def publish_all_phase8_models(
    merged_dir: Path | str = DEFAULT_MERGED_DIR,
    gguf_dir: Path | str = DEFAULT_GGUF_DIR,
    adapter_dir: Path | str | None = None,
    base_repo_id: str = DEFAULT_BASE_REPO_ID,
    gguf_repo_id: str = DEFAULT_GGUF_REPO_ID,
    model_card_base: Path | str = DEFAULT_MODEL_CARD_BASE,
    model_card_gguf: Path | str = DEFAULT_MODEL_CARD_GGUF,
    token: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Publish both merged safetensors (with adapter) and quantized GGUF models to Hugging Face.

    Publish base model and adapter to second-chess/Second-1.0-2B.
    Publish GGUF and shards to second-chess/Second-1.0-2B-GGUF.
    Return publication summaries.
    """
    base_subfolders: list[tuple[Path | str, str]] = []
    if adapter_dir and Path(adapter_dir).is_dir():
        base_subfolders.append((adapter_dir, "adapter"))

    LOGGER.info("Publishing base merged safetensors model to '%s'...", base_repo_id)
    base_summary = publish_model_to_hf(
        repo_id=base_repo_id,
        folder_path=merged_dir,
        model_card_path=model_card_base,
        subfolders=base_subfolders if base_subfolders else None,
        commit_message="Release Second-1.0-2B (merged safetensors + LoRA adapter)",
        token=token,
        dry_run=dry_run,
    )

    LOGGER.info("Publishing quantized GGUF model and browser shards to '%s'...", gguf_repo_id)
    gguf_summary = publish_model_to_hf(
        repo_id=gguf_repo_id,
        folder_path=gguf_dir,
        model_card_path=model_card_gguf,
        commit_message="Release Second-1.0-2B-GGUF (Q4_K_M quant and browser shards)",
        token=token,
        dry_run=dry_run,
    )

    return {
        "base_model": base_summary,
        "gguf_model": gguf_summary,
    }


def main() -> None:
    """CLI entry point for publishing models to Hugging Face."""
    parser = argparse.ArgumentParser(
        description="Publish Second-1.0-2B safetensors and GGUF models to Hugging Face Hub."
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default=str(DEFAULT_MERGED_DIR),
        help="Path to merged safetensors model directory",
    )
    parser.add_argument(
        "--gguf-dir",
        type=str,
        default=str(DEFAULT_GGUF_DIR),
        help="Path to GGUF model files directory",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=None,
        help="Optional path to winning LoRA adapter directory to upload alongside merged model",
    )
    parser.add_argument(
        "--base-repo",
        type=str,
        default=DEFAULT_BASE_REPO_ID,
        help="Target Hugging Face repository ID for merged safetensors",
    )
    parser.add_argument(
        "--gguf-repo",
        type=str,
        default=DEFAULT_GGUF_REPO_ID,
        help="Target Hugging Face repository ID for GGUF files",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate publication without uploading files to Hugging Face",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    res = publish_all_phase8_models(
        merged_dir=args.merged_dir,
        gguf_dir=args.gguf_dir,
        adapter_dir=args.adapter_dir,
        base_repo_id=args.base_repo,
        gguf_repo_id=args.gguf_repo,
        token=args.token,
        dry_run=args.dry_run,
    )
    print("Publication completed:")
    print(f"Base model URL: {res['base_model'].get('url')}")
    print(f"GGUF model URL: {res['gguf_model'].get('url')}")


if __name__ == "__main__":
    main()
