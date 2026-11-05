"""Unit and integration tests for the Second evaluation harness.

This module tests automatic metrics, LLM judge behavior, baseline generation, and report building.
It verifies swap consistency, caching, retry backoff, and end-to-end evaluation runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.baselines import (
    RuleBaselineGenerator,
    TransformersBaselineGenerator,
    create_baseline_generator,
    generate_baseline,
    load_baseline,
)
from eval.judge import (
    GeminiClient,
    GeminiJudge,
    JudgeCache,
    PairwiseComparison,
    PairwiseSummary,
    RubricScore,
    RubricSummary,
    extract_json_from_text,
    get_gemini_api_key,
    load_env_file,
    run_judge_eval,
)
from eval.metrics import (
    AggregateMetrics,
    EvalMetrics,
    ItemMetric,
    clean_completion_text,
    count_words,
    ensure_prompt_text,
    evaluate_completions,
    evaluate_file,
    evaluate_item,
    is_in_length_band,
)
from eval.run_eval import generate_markdown_report, run_eval


# Sample prompt text for testing
SAMPLE_PROMPT_MOVE = (
    "task: explain_move\n"
    "level: club\n"
    "detail: short\n\n"
    "FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4\n"
    "White to move, move 4.\n"
    "Move played: Ng5.\n"
    "Verdict: Good.\n"
    "White's winning chances: 54% → 53%.\n"
    "Best move: d3.\n"
    "Best line: d3 Bc5 c3 d6.\n"
    "Facts:\n"
    "- Ng5 attacks the pawn on f7 together with the bishop on c4.\n"
    "- Black's best reply is d5.\n"
    "- Material: equal.\n"
)

SAMPLE_PROMPT_POSITION = (
    "task: explain_position\n"
    "level: beginner\n"
    "detail: short\n\n"
    "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n"
    "Black to move, move 1.\n"
    "Black's winning chances: 50%.\n"
    "Best move: e5.\n"
    "Best line: e5 Nf3 Nc6.\n"
    "Facts:\n"
    "- White controls central squares with the e4 pawn.\n"
    "- Material: equal.\n"
)


# =====================================================================
# 1. Automatic Metrics Tests (Phase 4.2)
# =====================================================================

def test_clean_completion_text_removes_thinking() -> None:
    """Strip thinking blocks and return clean response."""
    raw = "<think>\nConsider Ng5 or d3.\n</think>\nWhite plays Ng5 to attack f7."
    cleaned = clean_completion_text(raw)
    assert cleaned == "White plays Ng5 to attack f7."
    assert "<think>" not in cleaned


def test_count_words_and_length_band() -> None:
    """Verify word counting and target detail band boundaries."""
    text_short = "White plays Ng5 to create a direct attack on f7."
    assert count_words(text_short) == 10
    assert not is_in_length_band("short", count_words(text_short))

    words_40 = "word " * 40
    assert count_words(words_40) == 40
    assert is_in_length_band("short", count_words(words_40))
    assert not is_in_length_band("long", count_words(words_40))

    words_100 = "word " * 100
    assert is_in_length_band("long", count_words(words_100))
    assert not is_in_length_band("short", count_words(words_100))


def test_evaluate_item_clean_pass() -> None:
    """Evaluate clean completion that satisfies all hard checks."""
    completion = (
        "The move Ng5 is a good move. "
        "White places active pressure on the vulnerable f7 square. "
        "Black must defend carefully against the combined threat of the bishop and knight. "
        "This maintains strong development and active initiative in the opening."
    )
    item = {
        "id": "t1",
        "user_message": SAMPLE_PROMPT_MOVE,
    }
    result = evaluate_item(item, completion=completion)
    assert result.id == "t1"
    assert result.task == "explain_move"
    assert result.level == "club"
    assert result.detail == "short"
    assert result.passed_all_hard
    assert len(result.failed_checks) == 0
    assert result.reward > 0.0
    assert result.in_length_band
    assert result.verdict_accurate is True
    assert result.verdict_covered is True


def test_evaluate_item_hard_check_illegal_move_fails() -> None:
    """Fail hard check when completion contains an illegal move."""
    # White cannot play e8=Q from initial position or in move 4 Italian
    completion = (
        "White should play e8=Q here immediately. "
        "This gives a decisive advantage on the board."
    )
    item = {
        "id": "t2",
        "user_message": SAMPLE_PROMPT_MOVE,
    }
    result = evaluate_item(item, completion=completion)
    assert not result.passed_all_hard
    assert "legal_moves" in result.failed_checks
    assert result.reward == -1.0


def test_evaluate_item_beginner_jargon_fails() -> None:
    """Fail hard check when beginner text contains prohibited jargon."""
    completion = (
        "White has an evaluation of +1.5 at depth 20 according to the engine. "
        "This prevents any tactical activity."
    )
    item = {
        "id": "t3",
        "user_message": SAMPLE_PROMPT_POSITION,
    }
    result = evaluate_item(item, completion=completion)
    assert not result.passed_all_hard
    assert "beginner_jargon" in result.failed_checks
    assert result.reward == -1.0


def test_evaluate_completions_aggregation() -> None:
    """Verify aggregated metrics across passing and failing items."""
    good_text = (
        "The move Ng5 is a good idea. "
        "White targets the f7 pawn together with the bishop on c4. "
        "Black can respond with d5 to challenge the diagonal. "
        "Both players contest the center actively."
    )
    bad_text = "White plays Qd8 which wins everything."

    records = [
        {"id": "item1", "user_message": SAMPLE_PROMPT_MOVE, "completion": good_text},
        {"id": "item2", "user_message": SAMPLE_PROMPT_MOVE, "completion": bad_text},
    ]

    metrics = evaluate_completions(records)
    agg = metrics.aggregate

    assert agg.total_items == 2
    assert agg.hard_fail_rate == 50.0
    assert agg.hard_check_fail_rates["legal_moves"] == 50.0
    assert agg.mean_reward < 1.0
    assert agg.verdict_items_count == 2
    assert "level" in agg.breakdowns
    assert "task" in agg.breakdowns
    assert "detail" in agg.breakdowns


def test_evaluate_file(tmp_path: Path) -> None:
    """Load items from a JSONL file and run evaluation."""
    file_path = tmp_path / "test_outputs.jsonl"
    record = {
        "id": "f1",
        "user_message": SAMPLE_PROMPT_MOVE,
        "completion": "The move Ng5 is a good continuation that attacks the f7 pawn.",
    }
    file_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    metrics = evaluate_file(file_path)
    assert metrics.aggregate.total_items == 1
    assert metrics.items[0].id == "f1"


# =====================================================================
# 2. LLM Judge Tests (Phase 4.3)
# =====================================================================

def test_extract_json_from_text() -> None:
    """Extract JSON object with and without markdown formatting."""
    plain = '{"winner": "A", "reason": "Better clarity."}'
    assert extract_json_from_text(plain)["winner"] == "A"

    markdown_fenced = '```json\n{"winner": "B", "reason": "Accurate."}\n```'
    assert extract_json_from_text(markdown_fenced)["winner"] == "B"

    embedded = 'Here is the verdict: {"winner": "tie", "reason": "Equal."} Thank you.'
    assert extract_json_from_text(embedded)["winner"] == "tie"


def test_judge_swap_consistency_candidate_wins() -> None:
    """Declare candidate winner when both orders pick candidate."""
    judge = GeminiJudge(client=MagicMock())
    # Order 1 (cand=A, opp=B): returns A -> candidate
    # Order 2 (opp=A, cand=B): returns B -> candidate
    judge._call_pairwise_prompt = MagicMock(side_effect=[("A", "cand better"), ("B", "cand better")])

    result = judge.compare_pair(
        prompt_text=SAMPLE_PROMPT_MOVE,
        level="club",
        candidate_text="Candidate text",
        opponent_text="Opponent text",
        item_id="pair_1",
    )
    assert result.winner == "candidate"
    assert result.swap_agreed is True
    assert result.order1_vote == "candidate"
    assert result.order2_vote == "candidate"


def test_judge_swap_consistency_opponent_wins() -> None:
    """Declare opponent winner when both orders pick opponent."""
    judge = GeminiJudge(client=MagicMock())
    # Order 1 (cand=A, opp=B): returns B -> opponent
    # Order 2 (opp=A, cand=B): returns A -> opponent
    judge._call_pairwise_prompt = MagicMock(side_effect=[("B", "opp better"), ("A", "opp better")])

    result = judge.compare_pair(
        prompt_text=SAMPLE_PROMPT_MOVE,
        level="club",
        candidate_text="Candidate text",
        opponent_text="Opponent text",
        item_id="pair_2",
    )
    assert result.winner == "opponent"
    assert result.swap_agreed is True
    assert result.order1_vote == "opponent"
    assert result.order2_vote == "opponent"


def test_judge_swap_consistency_disagreement_becomes_tie() -> None:
    """Resolve disagreeing picks across orders to a tie."""
    judge = GeminiJudge(client=MagicMock())
    # Order 1: picks A (candidate). Order 2: picks A (opponent). Positional bias!
    judge._call_pairwise_prompt = MagicMock(side_effect=[("A", "A was first"), ("A", "A was first")])

    result = judge.compare_pair(
        prompt_text=SAMPLE_PROMPT_MOVE,
        level="club",
        candidate_text="Candidate text",
        opponent_text="Opponent text",
        item_id="pair_3",
    )
    assert result.winner == "tie"
    assert result.swap_agreed is False
    assert result.order1_vote == "candidate"
    assert result.order2_vote == "opponent"


def test_judge_swap_consistency_one_tie_becomes_tie() -> None:
    """Resolve partial tie pick to a final tie."""
    judge = GeminiJudge(client=MagicMock())
    # Order 1: candidate. Order 2: tie.
    judge._call_pairwise_prompt = MagicMock(side_effect=[("A", "cand"), ("tie", "both good")])

    result = judge.compare_pair(
        prompt_text=SAMPLE_PROMPT_MOVE,
        level="club",
        candidate_text="Candidate text",
        opponent_text="Opponent text",
        item_id="pair_4",
    )
    assert result.winner == "tie"
    assert result.swap_agreed is False


def test_judge_rubric_scoring() -> None:
    """Score explanation rubric dimensions and compute overall average."""
    mock_client = MagicMock()
    mock_client.generate_json.return_value = {
        "correctness": 4,
        "clarity": 5,
        "usefulness": 4,
        "level_fit": 5,
        "reason": "Clear and accurate advice.",
    }
    judge = GeminiJudge(client=mock_client)
    score = judge.score_rubric(
        prompt_text=SAMPLE_PROMPT_MOVE,
        level="club",
        explanation="White plays Ng5 with strong initiative.",
    )
    assert score.correctness == 4
    assert score.clarity == 5
    assert score.usefulness == 4
    assert score.level_fit == 5
    assert score.overall == 4.5
    assert "Clear" in score.reason


def test_judge_cache_hits_and_misses(tmp_path: Path) -> None:
    """Verify cache writes new values and returns cached entries."""
    cache_file = tmp_path / "cache.jsonl"
    cache = JudgeCache(cache_file)

    assert cache.get("key123") is None

    cache.set("key123", {"winner": "A"}, meta={"id": "test"})
    assert cache.get("key123") == {"winner": "A"}

    # Reload cache from disk
    reloaded_cache = JudgeCache(cache_file)
    assert reloaded_cache.get("key123") == {"winner": "A"}


def test_gemini_client_retry_and_backoff(tmp_path: Path) -> None:
    """Retry API requests upon 429 status and succeed when 200 is returned."""
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.text = "Rate limited"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": '{"winner": "A", "reason": "Strong."}'}]
                }
            }
        ]
    }

    cache_file = tmp_path / "gemini_retry_cache.jsonl"
    client = GeminiClient(api_key="test_key", cache=JudgeCache(cache_file), base_delay=0.01)
    with patch("eval.judge.requests.post", side_effect=[mock_resp_429, mock_resp_200]) as mock_post:
        result = client.generate_json("unique test prompt 999", task_type="pairwise")
        assert result["winner"] == "A"
        assert mock_post.call_count == 2


# =====================================================================
# 3. Baselines Generator Tests (Phase 4.4)
# =====================================================================

def test_rule_baseline_generator_move_and_position() -> None:
    """Generate deterministic explanations for move and position tasks."""
    gen = RuleBaselineGenerator()

    # Move task
    out_move = gen.generate(SAMPLE_PROMPT_MOVE, {})
    assert "Ng5" in out_move
    assert "good" in out_move.lower()
    assert len(out_move.split()) >= 10

    # Position task
    out_pos = gen.generate(SAMPLE_PROMPT_POSITION, {})
    assert "position" in out_pos.lower()
    assert len(out_pos.split()) >= 8


def test_generate_baseline_resumes(tmp_path: Path) -> None:
    """Resume baseline generation and avoid re-evaluating existing items."""
    testset_file = tmp_path / "testset.jsonl"
    item1 = {"id": "item_1", "user_message": SAMPLE_PROMPT_MOVE}
    item2 = {"id": "item_2", "user_message": SAMPLE_PROMPT_POSITION}
    testset_file.write_text(json.dumps(item1) + "\n" + json.dumps(item2) + "\n", encoding="utf-8")

    out_dir = tmp_path / "baselines"
    out_dir.mkdir(parents=True)
    baseline_file = out_dir / "my_base.jsonl"

    # Pre-populate item_1
    pre_record = {
        "id": "item_1",
        "prompt": SAMPLE_PROMPT_MOVE,
        "completion": "Existing completion.",
    }
    baseline_file.write_text(json.dumps(pre_record) + "\n", encoding="utf-8")

    records = generate_baseline(
        baseline_name="my_base",
        backend="rule-baseline",
        testset_path=testset_file,
        output_dir=out_dir,
    )
    assert len(records) == 2
    # Verify file contains both item_1 and item_2
    final_items = load_baseline("my_base", baselines_dir=out_dir)
    assert len(final_items) == 2
    assert final_items[0]["id"] == "item_1"
    assert final_items[1]["id"] == "item_2"


# =====================================================================
# 4. Evaluation Runner & Reports Tests (Phase 4 Done Criteria)
# =====================================================================

def test_generate_markdown_report_structure() -> None:
    """Verify Markdown report sections adhere to required structure."""
    record = {
        "id": "ex1",
        "user_message": SAMPLE_PROMPT_MOVE,
        "completion": "The move Ng5 is a good idea. White targets the f7 pawn directly.",
    }
    metrics = evaluate_completions([record])
    judge_data = {
        "rubric": {
            "summary": {
                "mean_correctness": 4.5,
                "mean_clarity": 4.0,
                "mean_usefulness": 4.0,
                "mean_level_fit": 4.5,
                "mean_overall": 4.25,
            }
        },
        "pairwise": {
            "summaries": {
                "base-qwen-2b": {
                    "total_pairs": 1,
                    "candidate_wins": 1,
                    "opponent_wins": 0,
                    "ties": 0,
                    "candidate_win_rate": 100.0,
                    "candidate_win_tie_rate": 100.0,
                    "swap_agreement_rate": 100.0,
                }
            }
        },
    }

    report = generate_markdown_report(
        model_name="test_model",
        metrics=metrics,
        judge_results=judge_data,
    )
    assert "# Evaluation Report: test_model" in report
    assert "## 1. Executive Summary" in report
    assert "## 2. Hard Check Hallucination Metrics" in report
    assert "## 3. Soft Score Breakdown" in report
    assert "## 4. Subgroup Breakdowns" in report
    assert "## 5. LLM Judge Results" in report
    assert "base-qwen-2b" in report
    assert "## 6. Sample Outputs" in report


def test_run_eval_end_to_end_with_rule_backend(tmp_path: Path) -> None:
    """Execute end-to-end evaluation harness with rule backend and skipped judge."""
    testset_file = tmp_path / "testset.jsonl"
    item = {
        "id": "run_1",
        "task": "explain_move",
        "level": "club",
        "detail": "short",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "move_played": "Ng5",
        "verdict": "Good",
        "facts": ["Ng5 attacks f7."],
        "user_message": SAMPLE_PROMPT_MOVE,
    }
    testset_file.write_text(json.dumps(item) + "\n", encoding="utf-8")

    out_dir = tmp_path / "results"
    rep_dir = tmp_path / "reports"
    base_dir = tmp_path / "baselines"

    # Create baseline opponent
    generate_baseline(
        baseline_name="base_opponent",
        backend="rule-baseline",
        testset_path=testset_file,
        output_dir=base_dir,
    )

    eval_result = run_eval(
        model_name="test_rule_model",
        model_path="dummy_path",
        backend="rule-baseline",
        testset_path=testset_file,
        output_dir=out_dir,
        report_dir=rep_dir,
        baselines_dir=base_dir,
        skip_judge=True,
    )

    assert eval_result["model_name"] == "test_rule_model"
    assert eval_result["total_items"] == 1
    assert (out_dir / "test_rule_model_eval.json").is_file()
    assert (rep_dir / "test_rule_model.md").is_file()

    # Read generated markdown report
    md_content = (rep_dir / "test_rule_model.md").read_text(encoding="utf-8")
    assert "Evaluation Report: test_rule_model" in md_content
    assert "skipped LLM judge evaluation" in md_content


def test_transformers_baseline_generator_passes_enable_thinking_false() -> None:
    """Verify that generate passes enable_thinking=False when supported."""
    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "White plays Ng5."

    mock_inputs = {"input_ids": MagicMock(shape=[1, 3])}
    mock_tokenizer.return_value.to.return_value = mock_inputs
    mock_model.device = "cpu"
    mock_model.generate.return_value = [MagicMock()]

    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
        gen = TransformersBaselineGenerator("dummy-model", device="cpu")
        output = gen.generate("Sample prompt", {})

        assert output == "White plays Ng5."
        mock_tokenizer.apply_chat_template.assert_called_once()
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert kwargs.get("enable_thinking") is False


def test_transformers_baseline_generator_fallback_when_enable_thinking_raises_type_error() -> None:
    """Verify fallback when tokenizer does not accept enable_thinking."""
    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    def side_effect(*args: Any, **kwargs: Any) -> str:
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        return "fallback formatted prompt"

    mock_tokenizer.apply_chat_template.side_effect = side_effect
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "White plays Ng5."

    mock_inputs = {"input_ids": MagicMock(shape=[1, 3])}
    mock_tokenizer.return_value.to.return_value = mock_inputs
    mock_model.device = "cpu"
    mock_model.generate.return_value = [MagicMock()]

    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
        gen = TransformersBaselineGenerator("dummy-model", device="cpu")
        output = gen.generate("Sample prompt", {})

        assert output == "White plays Ng5."
        assert mock_tokenizer.apply_chat_template.call_count == 2


def test_transformers_baseline_generator_loads_peft_adapter(tmp_path: Path) -> None:
    """Verify loading from PEFT adapter checkpoint when adapter_config.json exists."""
    adapter_dir = tmp_path / "adapter_ckpt"
    adapter_dir.mkdir()
    config_file = adapter_dir / "adapter_config.json"
    config_file.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-2B"}), encoding="utf-8")

    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    mock_peft = MagicMock()

    mock_base_model = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_base_model
    mock_peft_model = MagicMock()
    mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model
    mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers, "peft": mock_peft}):
        gen = TransformersBaselineGenerator(str(adapter_dir), device="cpu")

        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3.5-2B",
            torch_dtype=mock_torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        mock_peft.PeftModel.from_pretrained.assert_called_once_with(mock_base_model, str(adapter_dir))
        assert gen.model == mock_peft_model
