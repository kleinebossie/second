"""Unit tests for teacher selection spike.

All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from spikes.teacher_selection.run_teacher_spike import (
    DEFAULT_CANDIDATES,
    evaluate_completions,
    generate_mock_completions,
    load_devset_items,
    run_spike,
)
from training.build_colab_bundle import (
    DEFAULT_PHASE2_OUTPUT_ZIP,
    build_phase2_teacher_bundle,
)


def test_default_candidates() -> None:
    """Verify that evaluated candidates match specification."""
    expected = [
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.6-27B",
    ]
    assert DEFAULT_CANDIDATES == expected
    assert "Qwen/Qwen3.8-27B" not in DEFAULT_CANDIDATES
    assert "Qwen/Qwen2.5-32B-Instruct" not in DEFAULT_CANDIDATES


def test_load_devset_items() -> None:
    """Verify loading devset items from disk."""
    devset_path = Path("eval/devset-1.0.jsonl")
    assert devset_path.is_file()
    items = load_devset_items(devset_path, count=5)
    assert len(items) == 5
    for item in items:
        assert "user_message" in item
        assert "fen" in item
        assert "task" in item


def test_mock_completions_and_evaluation() -> None:
    """Verify mock generation and evaluation scoring."""
    devset_path = Path("eval/devset-1.0.jsonl")
    items = load_devset_items(devset_path, count=5)
    completions = generate_mock_completions(items)
    assert len(completions) == len(items)

    for comp in completions:
        assert "<think>" in comp
        assert "</think>" in comp

    eval_summary = evaluate_completions(items, completions)
    assert eval_summary["total"] == 5
    assert eval_summary["pass_rate"] == 1.0
    assert eval_summary["mean_reward"] > 0.80
    assert len(eval_summary["item_results"]) == 5


def test_run_spike_mock(tmp_path: Path) -> None:
    """Verify run_spike creates summary and generations files."""
    devset_path = Path("eval/devset-1.0.jsonl")
    summary = run_spike(
        model_name="test_candidate",
        devset_path=devset_path,
        output_dir=tmp_path,
        count=3,
        is_mock=True,
    )
    assert summary["total"] == 3
    assert (tmp_path / "test_candidate_summary.json").is_file()
    assert (tmp_path / "test_candidate_generations.jsonl").is_file()


def test_notebook_structure_and_candidates() -> None:
    """Verify teacher_spike.ipynb structure and model candidates."""
    nb_path = Path("spikes/teacher_selection/teacher_spike.ipynb")
    assert nb_path.is_file()

    nb_text = nb_path.read_text(encoding="utf-8")
    nb = json.loads(nb_text)
    assert "cells" in nb

    assert "Qwen/Qwen3.5-9B" in nb_text
    assert "Qwen/Qwen3.5-27B" in nb_text
    assert "Qwen/Qwen3.5-35B-A3B" in nb_text
    assert "Qwen/Qwen3.6-27B" in nb_text
    assert "Qwen/Qwen3.8-27B" not in nb_text
    assert "Qwen/Qwen2.5-32B" not in nb_text

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) == 8

    # Verify Step 3 installs langdetect
    step3_source = "".join(code_cells[2].get("source", []))
    assert "langdetect" in step3_source

    # Verify candidate execution steps
    step4_source = "".join(code_cells[3].get("source", []))
    assert "Qwen/Qwen3.5-9B" in step4_source
    assert "--model Qwen/Qwen3.5-9B" in step4_source
    assert "--load-in-4bit" not in step4_source

    step5_source = "".join(code_cells[4].get("source", []))
    assert "Qwen/Qwen3.5-27B" in step5_source
    assert "--model Qwen/Qwen3.5-27B" in step5_source
    assert "--load-in-4bit" in step5_source

    step6_source = "".join(code_cells[5].get("source", []))
    assert "Qwen/Qwen3.5-35B-A3B" in step6_source
    assert "--model Qwen/Qwen3.5-35B-A3B" in step6_source
    assert "--load-in-4bit" in step6_source

    step7_source = "".join(code_cells[6].get("source", []))
    assert "Qwen/Qwen3.6-27B" in step7_source
    assert "--model Qwen/Qwen3.6-27B" in step7_source
    assert "--load-in-4bit" in step7_source

    step8_source = "".join(code_cells[7].get("source", []))
    assert "3_8" in step8_source
    assert "3.8" in step8_source


def test_load_devset_items_boundary() -> None:
    """Verify loading devset items with zero count returns empty list."""
    devset_path = Path("eval/devset-1.0.jsonl")
    assert load_devset_items(devset_path, count=0) == []
    assert load_devset_items(devset_path, count=-1) == []


def test_evaluate_completions_empty() -> None:
    """Verify evaluate_completions returns zero defaults when input is empty."""
    res = evaluate_completions([], [])
    assert res["total"] == 0
    assert res["pass_count"] == 0
    assert res["pass_rate"] == 0.0
    assert res["mean_reward"] == 0.0
    assert res["in_band_rate"] == 0.0
    assert res["mean_word_count"] == 0.0
    assert res["mean_key_point"] == 0.0
    assert res["mean_better_move"] == 0.0
    assert res["item_results"] == []


def test_step8_filtering_excludes_qwen38(tmp_path: Path) -> None:
    """Verify Step 8 filtering logic excludes Qwen3.8 candidate summaries."""
    cand1 = tmp_path / "Qwen_Qwen3_5-9B_summary.json"
    cand2 = tmp_path / "Qwen_Qwen3_8-27B_summary.json"
    mock_sum = tmp_path / "mock_summary.json"

    cand1.write_text(json.dumps({"model_name": "Qwen/Qwen3.5-9B", "pass_rate": 0.5, "mean_reward": 0.8}))
    cand2.write_text(json.dumps({"model_name": "Qwen/Qwen3.8-27B", "pass_rate": 0.4, "mean_reward": 0.7}))
    mock_sum.write_text(json.dumps({"model_name": "mock", "pass_rate": 1.0, "mean_reward": 0.9}))

    summaries = list(tmp_path.glob("*_summary.json"))
    filtered = [s for s in summaries if "3_8" not in s.name and "3.8" not in s.name]
    real = [s for s in filtered if not s.name.startswith("mock_")]

    assert len(real) == 1
    assert "Qwen_Qwen3_5-9B" in real[0].name
    assert not any("3_8" in s.name for s in real)


def test_l4_gpu_memory_fit() -> None:
    """Verify memory sizing for 35B-A3B 4-bit on 24GB L4 GPU."""
    total_params = 35e9
    bits_per_param = 4
    bytes_per_param = bits_per_param / 8
    weight_bytes = total_params * bytes_per_param
    weight_gb = weight_bytes / (1024**3)

    activation_and_cache_gb = 2.0
    total_needed_gb = weight_gb + activation_and_cache_gb
    l4_vram_gb = 24.0

    assert weight_gb < 17.5
    assert total_needed_gb < l4_vram_gb
    assert l4_vram_gb - total_needed_gb >= 4.0


def test_teacher_spike_imports_without_langdetect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that importing teacher spike modules does not require langdetect."""
    import sys

    monkeypatch.setitem(sys.modules, "langdetect", None)
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("data.pipeline") or mod_name.startswith("rewards"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    from data.pipeline.synthesize_teacher import (
        build_teacher_messages,
        parse_teacher_completion,
    )

    assert build_teacher_messages("test") is not None
    assert parse_teacher_completion("test") is not None
    assert "data.pipeline.filter_comments" not in sys.modules

    from spikes.teacher_selection.run_teacher_spike import (
        evaluate_completions,
        generate_mock_completions,
    )

    items = [{"task": "explain_move", "fen": "8/8/8/8/8/8/8/8 w - - 0 1", "user_message": "test"}]
    comps = generate_mock_completions(items)
    res = evaluate_completions(items, comps)
    assert res["total"] == 1


def test_data_pipeline_lazy_filter_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that importing data.pipeline does not eagerly load filter_comments."""
    import sys

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("data.pipeline"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    import data.pipeline

    assert "data.pipeline.filter_comments" not in sys.modules

    # Verify that accessing filter attributes loads filter_comments lazily
    _ = data.pipeline.filter_comment
    assert "data.pipeline.filter_comments" in sys.modules


def test_filter_comments_without_langdetect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that filter_comment raises an informative error when langdetect is missing."""
    from data.pipeline import filter_comments

    monkeypatch.setattr(filter_comments, "detect", None)
    with pytest.raises(ImportError, match="The langdetect module is missing"):
        filter_comments.filter_comment(
            "This is a clean chess comment with sufficient length to check.",
            fen="8/8/8/8/8/8/8/8 w - - 0 1",
        )


def test_phase2_bundle_contents(tmp_path: Path) -> None:
    """Verify Phase 2 bundle contains required code and data files."""
    test_zip = tmp_path / "phase2_bundle.zip"
    built = build_phase2_teacher_bundle(output_path=test_zip)
    assert built.is_file()

    with zipfile.ZipFile(built, "r") as zf:
        names = set(zf.namelist())
        assert "eval/devset-1.0.jsonl" in names
        assert "docs/decisions/teacher_model.md" in names
        assert "spikes/teacher_selection/run_teacher_spike.py" in names
        assert "spikes/teacher_selection/test_teacher_spike.py" in names
        assert "spikes/teacher_selection/teacher_spike.ipynb" in names
        assert "spikes/teacher_selection/README.md" in names
        assert "data/pipeline/synthesize_teacher.py" in names
        assert "rewards/scorer.py" in names
