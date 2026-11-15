"""LoRA adapter merge and text causal language model export module.

This module loads base causal language models and merges trained LoRA adapters.
It removes vision tower weights if present to minimize download size.
It saves merged model weights in safetensors format with tokenizer files.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_ADAPTER_DIR = Path("/content/drive/MyDrive/second/checkpoints/grpo_v2_b2/final_adapter")
DEFAULT_OUTPUT_DIR = Path("export/merged_safetensors")

VISION_PREFIXES: tuple[str, ...] = (
    "visual.",
    "vision_tower.",
    "mm_projector.",
    "transformer.visual.",
    "model.visual.",
)


def apply_torchao_workaround() -> None:
    """Disable torchao and torchaudio detection to avoid Google Colab binary conflicts."""
    import sys

    # Block broken torchaudio C++ extensions from attempting to load libc10_cuda.so
    sys.modules["torchaudio"] = None

    try:
        import transformers.utils.import_utils

        transformers.utils.import_utils.is_torchaudio_available = lambda: False
    except Exception:
        pass

    try:
        import peft.import_utils

        peft.import_utils.is_torchao_available = lambda: False
        LOGGER.info("Applied torchao/torchaudio workaround.")
    except (ImportError, AttributeError):
        pass


def strip_vision_parameters(
    state_dict: dict[str, Any],
    config_dict: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Remove vision tower weights and configuration entries from model dictionary.

    Find all state dictionary keys starting with vision prefixes.
    Remove those keys from the state dictionary.
    Update configuration dictionary to ensure text causal LM architecture.
    Return updated state dictionary and count of removed parameters.
    """
    keys_to_remove = [
        key
        for key in state_dict.keys()
        if any(key.startswith(prefix) for prefix in VISION_PREFIXES)
    ]
    for key in keys_to_remove:
        del state_dict[key]

    if config_dict is not None:
        if "visual" in config_dict:
            del config_dict["visual"]
        if "vision_config" in config_dict:
            del config_dict["vision_config"]
        if "mtp_num_hidden_layers" in config_dict:
            del config_dict["mtp_num_hidden_layers"]
        if isinstance(config_dict.get("text_config"), dict):
            config_dict["text_config"].pop("mtp_num_hidden_layers", None)
        # Ensure architecture reflects text causal LM
        raw_arch = config_dict.get("architectures")
        current_arch = raw_arch if isinstance(raw_arch, list) else []
        if not current_arch or any(isinstance(arch, str) and "ConditionalGeneration" in arch for arch in current_arch):
            config_dict["architectures"] = ["Qwen2ForCausalLM"]

    LOGGER.info("Stripped %d vision parameters from state dictionary.", len(keys_to_remove))
    return state_dict, len(keys_to_remove)


def merge_and_export_lora(
    base_model_path: str = DEFAULT_BASE_MODEL,
    adapter_dir: Path | str = DEFAULT_ADAPTER_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    strip_vision: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "cpu",
    dry_run: bool = False,
) -> Path:
    """Load base model, merge LoRA adapter, strip vision layers, and save safetensors.

    Load the base transformer model in bfloat16.
    Load the PEFT LoRA adapter and merge with base weights.
    Remove vision layers if strip_vision is True.
    Save model weights in safetensors format and save tokenizer assets.
    Return the destination output path.
    """
    adapter_path = Path(adapter_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preparing to merge base model '%s' with adapter '%s'...", base_model_path, adapter_path)

    if dry_run:
        LOGGER.info("Dry-run mode active. Simulating merge and export...")
        dummy_manifest = {
            "base_model": base_model_path,
            "adapter_dir": str(adapter_path),
            "output_dir": str(out_path),
            "torch_dtype": torch_dtype,
            "strip_vision": strip_vision,
            "status": "simulated",
        }
        manifest_path = out_path / "export_manifest.json"
        manifest_path.write_text(json.dumps(dummy_manifest, indent=2), encoding="utf-8")
        return out_path

    # Apply Colab workaround before importing torch/peft
    apply_torchao_workaround()

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing PyTorch, Transformers, or PEFT packages. "
            "Install required packages or run this script in the Colab GPU environment."
        ) from exc

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    LOGGER.info("Loading tokenizer from '%s'...", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    except ModuleNotFoundError as exc:
        if "qwen3_5" in str(exc).lower():
            raise RuntimeError(
                f"Failed to load '{base_model_path}' because the installed transformers package lacks Qwen3.5 architecture support ({exc}).\n"
                "Please run: !pip install --upgrade git+https://github.com/huggingface/transformers.git\n"
                "and restart your Colab runtime session."
            ) from exc
        raise
    except Exception as exc:
        LOGGER.warning("AutoModelForCausalLM failed (%s); attempting AutoModelForConditionalGeneration...", exc)
        try:
            from transformers import AutoModelForConditionalGeneration
            base_model = AutoModelForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=resolved_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception:
            raise exc

    if strip_vision and hasattr(base_model, "visual"):
        LOGGER.info("Removing visual module from base model...")
        delattr(base_model, "visual")

    LOGGER.info("Loading LoRA adapter from '%s'...", adapter_path)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )

    LOGGER.info("Merging LoRA weights with base model...")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    removed_count = 0
    if strip_vision:
        # 1. Delete visual submodules from PyTorch model hierarchy
        for mod in [merged_model, getattr(merged_model, "model", None)]:
            if mod is not None:
                for attr_name in ["visual", "vision_tower", "mm_projector"]:
                    if hasattr(mod, attr_name):
                        LOGGER.info("Deleting '%s' module from model hierarchy...", attr_name)
                        delattr(mod, attr_name)

        # 2. Extract state dict, clean any remaining vision keys, and update config
        raw_state = merged_model.state_dict()
        clean_state, removed_count = strip_vision_parameters(
            raw_state,
            config_dict=merged_model.config.to_dict() if hasattr(merged_model, "config") else {},
        )
        LOGGER.info("strip_vision_parameters verified %d non-text parameters removed.", removed_count)

        if hasattr(merged_model, "config"):
            merged_model.config.architectures = ["Qwen2ForCausalLM"]
            for vision_attr in ("visual", "vision_config", "vision_tower", "mm_projector", "mtp_num_hidden_layers"):
                if hasattr(merged_model.config, vision_attr):
                    delattr(merged_model.config, vision_attr)
            if hasattr(merged_model.config, "text_config") and hasattr(merged_model.config.text_config, "mtp_num_hidden_layers"):
                delattr(merged_model.config.text_config, "mtp_num_hidden_layers")

        # 3. If any vision parameters were stripped from state_dict, save clean state directly
        if removed_count > 0:
            import safetensors.torch

            LOGGER.info("Saving stripped safetensors directly to guarantee zero vision overhead...")
            safetensors.torch.save_file(clean_state, out_path / "model.safetensors")
            merged_model.config.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            if hasattr(merged_model, "generation_config") and merged_model.generation_config is not None:
                merged_model.generation_config.save_pretrained(str(out_path))

            manifest = {
                "base_model": base_model_path,
                "adapter_dir": str(adapter_path),
                "output_dir": str(out_path),
                "torch_dtype": torch_dtype,
                "strip_vision": strip_vision,
                "removed_vision_parameters": removed_count,
                "architectures": getattr(merged_model.config, "architectures", ["Qwen2ForCausalLM"]),
                "status": "completed",
            }
            manifest_path = out_path / "export_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return out_path

    LOGGER.info("Saving merged model to '%s' in safetensors format...", out_path)
    merged_model.save_pretrained(
        str(out_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(out_path))

    manifest = {
        "base_model": base_model_path,
        "adapter_dir": str(adapter_path),
        "output_dir": str(out_path),
        "torch_dtype": torch_dtype,
        "strip_vision": strip_vision,
        "architectures": getattr(merged_model.config, "architectures", ["Qwen2ForCausalLM"]),
        "status": "completed",
    }
    manifest_path = out_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Merge completed successfully. Artifacts saved to %s", out_path)
    return out_path


def main() -> None:
    """CLI entry point for merging LoRA adapter with base model."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and export safetensors."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Hugging Face model ID or path to base causal language model",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=str(DEFAULT_ADAPTER_DIR),
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for merged safetensors model",
    )
    parser.add_argument(
        "--strip-vision",
        action="store_true",
        default=True,
        help="Strip visual encoder and multi-modal layers",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="PyTorch tensor datatype",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cpu",
        help="Device mapping (e.g., 'cpu', 'auto', 'cuda:0')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without loading large model weights",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    merge_and_export_lora(
        base_model_path=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        strip_vision=args.strip_vision,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
