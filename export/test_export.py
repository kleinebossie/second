"""Unit and integration tests for Phase 8 browser export pipeline.

This module tests LoRA merging, vision stripping, GGUF conversion commands,
llama-gguf-split tool execution, post-quantization quality gates, model cards,
Hugging Face upload staging, and Colab notebook structure.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from export.convert_gguf import (
    GGUF_MAGIC,
    build_convert_command,
    build_gguf_split_command,
    build_quantize_command,
    calculate_shard_count,
    convert_and_quantize_to_gguf,
    split_gguf_with_tool,
)
from export.generate_gguf_completions import (
    build_llama_cli_command,
    generate_gguf_batch_completions,
)
from export.merge_lora import (
    merge_and_export_lora,
    strip_vision_parameters,
)
from export.publish_hf import (
    prepare_repo_upload_folder,
    publish_all_phase8_models,
    publish_model_to_hf,
)
from export.quality_check import (
    MAX_ALLOWED_HARD_FAIL_PCT,
    QualityCheckResult,
    evaluate_quantization_quality,
    format_quantization_report,
    merge_gguf_completions_with_testset,
    run_quality_check_and_save,
    select_100_test_prompts,
)
from training.build_colab_bundle import build_phase8_export_bundle


def test_strip_vision_parameters() -> None:
    """Verify stripping of multi-modal and vision parameters from model dictionary."""
    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": [1.0, 2.0],
        "visual.patch_embed.proj.weight": [3.0, 4.0],
        "vision_tower.encoder.layer.weight": [5.0, 6.0],
        "mm_projector.weight": [7.0, 8.0],
        "lm_head.weight": [9.0, 10.0],
    }
    config_dict = {
        "architectures": ["Qwen2VLForConditionalGeneration"],
        "visual": {"embed_dim": 1024},
        "hidden_size": 2048,
    }

    clean_state, removed_count = strip_vision_parameters(state_dict, config_dict)
    assert removed_count == 3
    assert "visual.patch_embed.proj.weight" not in clean_state
    assert "vision_tower.encoder.layer.weight" not in clean_state
    assert "mm_projector.weight" not in clean_state
    assert "model.layers.0.self_attn.q_proj.weight" in clean_state
    assert "lm_head.weight" in clean_state

    assert "visual" not in config_dict
    assert config_dict["architectures"] == ["Qwen2ForCausalLM"]

    # Verify handling of None architectures in config
    config_none = {"architectures": None}
    _, _ = strip_vision_parameters({}, config_none)
    assert config_none["architectures"] == ["Qwen2ForCausalLM"]


def test_merge_and_export_lora_dry_run() -> None:
    """Verify dry-run mode for merge_and_export_lora produces export manifest."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "merged"
        res_dir = merge_and_export_lora(
            base_model_path="Qwen/Qwen3.5-2B",
            adapter_dir="/dummy/adapter",
            output_dir=out_dir,
            strip_vision=True,
            dry_run=True,
        )
        assert res_dir == out_dir
        manifest_file = out_dir / "export_manifest.json"
        assert manifest_file.is_file()
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert data["status"] == "simulated"
        assert data["base_model"] == "Qwen/Qwen3.5-2B"


def test_calculate_shard_count_and_gguf_container_sharding() -> None:
    """Verify GGUF file sharding splits binary data into valid GGUF container shards."""
    assert calculate_shard_count(100, 500) == 1
    assert calculate_shard_count(1000, 500) == 2
    assert calculate_shard_count(1221 * 1024 * 1024, 480 * 1024 * 1024) == 3

    split_cmd = build_gguf_split_command("model.gguf", "prefix", max_shard_bytes=480 * 1024 * 1024)
    assert "--split-max-size" in split_cmd
    assert "480M" in split_cmd

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_file = tmp_path / "model.gguf"
        # Write valid GGUF binary data
        data_bytes = GGUF_MAGIC + b"\x03\x00\x00\x00" + b"\x00" * (1024 * 1024 * 2)  # ~2 MB
        source_file.write_bytes(data_bytes)

        # Split into shards of at most 1 MB
        shards = split_gguf_with_tool(
            source_file,
            output_prefix=tmp_path / "shards" / "model",
            max_shard_bytes=1024 * 1024,
        )
        assert len(shards) == 3

        # Verify each shard starts with valid GGUF magic bytes
        for shard in shards:
            assert shard.read_bytes()[:4] == GGUF_MAGIC


def test_build_convert_and_quantize_commands() -> None:
    """Verify command list structure for llama.cpp tools."""
    cmd_convert = build_convert_command("model_dir", "out.gguf", "convert.py", "f16")
    assert "convert.py" in cmd_convert[1]
    assert "model_dir" in cmd_convert
    assert "--outtype" in cmd_convert
    assert "f16" in cmd_convert

    cmd_quant = build_quantize_command("in.gguf", "out.gguf", "Q4_K_M", "llama-quantize")
    assert cmd_quant[0] == "llama-quantize"
    assert "in.gguf" in cmd_quant
    assert "out.gguf" in cmd_quant
    assert "Q4_K_M" in cmd_quant


def test_convert_and_quantize_dry_run() -> None:
    """Verify dry-run mode for convert_and_quantize_to_gguf produces manifest and shards."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "gguf_out"
        res = convert_and_quantize_to_gguf(
            model_dir="/dummy/model",
            output_dir=out_dir,
            model_name="Second-1.0-2B",
            quant_type="Q4_K_M",
            shard=True,
            dry_run=True,
        )
        assert res["status"] == "simulated"
        assert res["quant_type"] == "Q4_K_M"
        assert len(res["shards"]) == 3
        assert (out_dir / "gguf_manifest.json").is_file()


def test_generate_gguf_completions_batch() -> None:
    """Verify batch GGUF completions generation creates structured records."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_jsonl = Path(tmp_dir) / "test_outputs.jsonl"
        res = generate_gguf_batch_completions(
            testset_path="eval/testset-1.0.jsonl",
            model_path=Path("/dummy/model.gguf"),
            output_path=out_jsonl,
            canonical_path="eval/results/grpo-1.0-2b-v2_outputs.jsonl",
            strategy="stratified",
            max_items=10,
        )
        assert len(res) == 10
        assert out_jsonl.is_file()
        assert res[0]["id"] == "test_001"
        assert len(res[0]["completion"]) > 0

    cmd = build_llama_cli_command("model.gguf", "prompt", "llama-cli")
    assert "-m" in cmd
    assert "--temp" in cmd
    assert "0.0" in cmd


def test_select_100_test_prompts() -> None:
    """Verify prompt selection selects exactly 100 items with all concept tiers."""
    dummy_items = [{"id": f"test_{i:03d}", "concept_tier": f"tier_{(i%5)+1}"} for i in range(300)]
    stratified = select_100_test_prompts(dummy_items, strategy="stratified")
    assert len(stratified) == 100
    tiers = {item["concept_tier"] for item in stratified}
    assert len(tiers) == 5

    first_100 = select_100_test_prompts(dummy_items, strategy="first_100")
    assert len(first_100) == 100


def test_quality_check_evaluation_with_real_outputs() -> None:
    """Verify post-quantization quality check execution against real completions on disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_out = Path(tmp_dir) / "eval.json"
        report_out = Path(tmp_dir) / "report.md"

        res = run_quality_check_and_save(
            testset_path="eval/testset-1.0.jsonl",
            canonical_outputs_path="eval/results/grpo-1.0-2b-v2_outputs.jsonl",
            quant_outputs_path="eval/results/export_gguf_q4_k_m_outputs.jsonl",
            eval_json_path=json_out,
            report_path=report_out,
            strategy="stratified",
            quant_type="Q4_K_M",
        )

        assert res.total_prompts == 100
        assert res.quant_hard_fail_pct <= MAX_ALLOWED_HARD_FAIL_PCT
        assert res.passed_hard_fail_gate is True
        assert json_out.is_file()
        assert report_out.is_file()

        report_content = report_out.read_text(encoding="utf-8")
        assert "PASSED" in report_content
        assert "Q4_K_M" in report_content
        assert "Hard-Fail Rate" in report_content


def test_model_cards_validity() -> None:
    """Verify that both model cards contain required metadata, benchmarks, and licenses."""
    card_base_path = Path("export/model_cards/Second-1.0-2B/README.md")
    card_gguf_path = Path("export/model_cards/Second-1.0-2B-GGUF/README.md")

    assert card_base_path.is_file()
    assert card_gguf_path.is_file()

    content_base = card_base_path.read_text(encoding="utf-8")
    assert "license: apache-2.0" in content_base
    assert "Second-1.0-2B" in content_base
    assert "Prompt Contract" in content_base
    assert "Stage T6 Frozen Evaluation Benchmarks" in content_base
    assert "Fair Play" in content_base

    content_gguf = card_gguf_path.read_text(encoding="utf-8")
    assert "license: apache-2.0" in content_gguf
    assert "wllama" in content_gguf
    assert "Q4_K_M" in content_gguf
    assert "Browser Shards" in content_gguf
    assert "Fair Play" in content_gguf


def test_publish_hf_staging_and_dry_run() -> None:
    """Verify staging directory creation and simulated Hugging Face publication."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_dir = Path(tmp_dir) / "src"
        src_dir.mkdir()
        (src_dir / "model.safetensors").write_text("dummy_weight", encoding="utf-8")

        card_path = Path(tmp_dir) / "CARD.md"
        card_path.write_text("# Test Card\n", encoding="utf-8")

        stage_dir = Path(tmp_dir) / "stage"
        staged = prepare_repo_upload_folder(src_dir, card_path, stage_dir)
        assert (staged / "README.md").is_file()
        assert (staged / "model.safetensors").is_file()

        upload_res = publish_model_to_hf(
            repo_id="second-chess/test-repo",
            folder_path=staged,
            commit_message="Test commit",
            dry_run=True,
        )
        assert upload_res["status"] == "simulated"
        assert upload_res["repo_id"] == "second-chess/test-repo"

        # Verify adapter subfolder publishing
        adapter_dir = Path(tmp_dir) / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_text("adapter_weight", encoding="utf-8")

        all_res = publish_all_phase8_models(
            merged_dir=src_dir,
            gguf_dir=src_dir,
            adapter_dir=adapter_dir,
            base_repo_id="second-chess/Second-1.0-2B",
            gguf_repo_id="second-chess/Second-1.0-2B-GGUF",
            model_card_base=card_path,
            model_card_gguf=card_path,
            dry_run=True,
        )
        assert all_res["base_model"]["status"] == "simulated"
        assert "adapter" in all_res["base_model"]["subfolders"]
        assert "adapter_model.safetensors" in all_res["base_model"]["subfolders"]["adapter"]


def test_export_browser_notebook_structure() -> None:
    """Verify export/export_browser.ipynb exists and contains all required Phase 8 workflow cells."""
    nb_path = Path("export/export_browser.ipynb")
    assert nb_path.is_file()

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(nb_data["cells"]) >= 8

    full_source = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"])
    assert "grpo_v2_b2" in full_source
    assert "convert_hf_to_gguf.py" in full_source
    assert "Q4_K_M" in full_source
    assert "llama-quantize" in full_source
    assert "llama-gguf-split" in full_source
    assert "generate_gguf_completions" in full_source
    assert "publish_all_phase8_models" in full_source
    assert "phase8_export_bundle.zip" in full_source
    assert "sys.path.insert" in full_source
    # Verify bugfix: quoted dictionary keys
    assert "upload_res['base_model'].get('url')" in full_source or 'upload_res["base_model"].get("url")' in full_source


def test_build_phase8_export_bundle() -> None:
    """Verify building the Colab bundle for Phase 8 includes export files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_out = Path(tmp_dir) / "phase8_bundle.zip"
        res_path = build_phase8_export_bundle(output_path=bundle_out)
        assert res_path.is_file()
        assert res_path.stat().st_size > 100000  # > 100 KB
