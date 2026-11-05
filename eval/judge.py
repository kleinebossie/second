"""Gemini Flash LLM judge module.

This module evaluates explanation quality with Gemini Flash.
It performs rubric scoring and blind pairwise comparisons with swap consistency.
It caches API responses to prevent duplicate calls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

import requests

from eval.metrics import clean_completion_text, ensure_prompt_text

LOGGER = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CACHE_PATH = Path("eval/results/judge_cache.jsonl")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def load_env_file(env_path: Path | str | None = None) -> dict[str, str]:
    """Load environment variables from a .env file if present.

    Search the given path, working directory, or parent repository directory.
    Do not overwrite existing environment variables.
    Return loaded key-value pairs.
    """
    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(Path(env_path))
    else:
        candidates.append(Path(".env"))
        candidates.append(Path(__file__).resolve().parent.parent / ".env")

    loaded: dict[str, str] = {}
    for candidate in candidates:
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8")
                for line in content.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if "=" in stripped:
                        key, val = stripped.split("=", 1)
                        clean_key = key.strip().removeprefix("export ").strip()
                        clean_val = val.strip().strip("'\"")
                        if clean_key and clean_key not in os.environ:
                            os.environ[clean_key] = clean_val
                        loaded[clean_key] = clean_val
                break
            except Exception as exc:
                LOGGER.warning("Could not read .env file at %s: %s", candidate, exc)
    return loaded


def get_gemini_api_key(required: bool = False) -> str | None:
    """Retrieve Gemini API key from environment or .env file.

    When key is missing and required is true, raise a ValueError.
    Return the API key string or None.
    """
    load_env_file()
    key = os.environ.get("GEMINI_API_KEY")
    if not key and required:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set GEMINI_API_KEY in the environment or in a .env file."
        )
    return key


def compute_cache_key(model: str, task_type: str, payload_data: dict[str, Any]) -> str:
    """Compute a deterministic hash key for judge requests.

    Return a SHA-256 hex digest string.
    """
    serialized = json.dumps(
        {"model": model, "task": task_type, "data": payload_data},
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class JudgeCache:
    """Persistent JSONL cache for LLM judge calls."""

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_PATH) -> None:
        """Initialize cache store from a JSONL file path."""
        self.path = Path(cache_path)
        self._cache: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load cached entries from disk if file exists."""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        record = json.loads(line_str)
                        k = record.get("key")
                        resp = record.get("response")
                        if k and resp is not None:
                            self._cache[k] = resp
        except Exception as exc:
            LOGGER.warning("Failed to load judge cache from %s: %s", self.path, exc)

    def get(self, key: str) -> dict[str, Any] | None:
        """Return cached response dict or None."""
        return self._cache.get(key)

    def set(
        self,
        key: str,
        response: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Store response in memory and append entry to disk."""
        self._cache[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "response": response,
            "meta": meta or {},
            "timestamp": time.time(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Parse JSON object from model output text.

    Strip markdown fences if present.
    Return parsed JSON dictionary.
    """
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text).strip()

    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", clean_text, re.DOTALL)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data

    raise ValueError(f"Could not extract JSON object from judge output: {text}")


class GeminiClient:
    """REST API client for Gemini with exponential backoff and caching."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_JUDGE_MODEL,
        cache: JudgeCache | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> None:
        """Initialize Gemini client parameters."""
        self.api_key = api_key or get_gemini_api_key(required=False)
        self.model = model
        self.cache = cache or JudgeCache()
        self.max_retries = max_retries
        self.base_delay = base_delay

    def generate_json(
        self,
        prompt: str,
        task_type: str,
        cache_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send prompt to Gemini and return parsed JSON response.

        Check cache before calling API.
        Retry on status 429 and 5xx with exponential backoff.
        Write new results to cache.
        """
        payload_data = {"prompt": prompt}
        cache_key = compute_cache_key(self.model, task_type, payload_data)

        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            return cached_response

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to call the Gemini API. "
                "Provide an API key or use cached evaluation results."
            )

        url = GEMINI_API_URL.format(model=self.model)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        body = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=60)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    candidates = resp_json.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"No candidates returned by Gemini: {resp_json}")
                    part_text = candidates[0]["content"]["parts"][0]["text"]
                    parsed = extract_json_from_text(part_text)
                    self.cache.set(cache_key, parsed, meta=cache_meta)
                    return parsed

                if resp.status_code in (429, 500, 502, 503, 504):
                    delay = self.base_delay * (2 ** attempt)
                    LOGGER.warning(
                        "Gemini API rate limit or error (%s). Retry in %.1f seconds.",
                        resp.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                raise RuntimeError(
                    f"Gemini API request failed with status {resp.status_code}: {resp.text}"
                )
            except Exception as exc:
                last_error = exc
                delay = self.base_delay * (2 ** attempt)
                LOGGER.warning("Gemini API call failed (%s). Retry in %.1f s.", exc, delay)
                time.sleep(delay)

        raise RuntimeError(
            f"Gemini API failed after {self.max_retries} retries: {last_error}"
        )


@dataclass(frozen=True)
class RubricScore:
    """Judge rubric rating for a single explanation."""

    correctness: int
    clarity: int
    usefulness: int
    level_fit: int
    overall: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert rubric score to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RubricSummary:
    """Summary of rubric scores across test items."""

    mean_correctness: float
    mean_clarity: float
    mean_usefulness: float
    mean_level_fit: float
    mean_overall: float
    total_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        """Convert rubric summary to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PairwiseComparison:
    """Pairwise match outcome with swap consistency check."""

    item_id: str
    order1_winner: str
    order1_vote: str
    order1_reason: str
    order2_winner: str
    order2_vote: str
    order2_reason: str
    winner: str
    swap_agreed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert pairwise comparison to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PairwiseSummary:
    """Aggregated match statistics against a baseline opponent."""

    opponent_name: str
    total_pairs: int
    candidate_wins: int
    opponent_wins: int
    ties: int
    candidate_win_rate: float
    candidate_win_tie_rate: float
    swap_agreement_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert pairwise summary to dictionary."""
        return asdict(self)


class GeminiJudge:
    """LLM Judge using Gemini Flash for rubric and pairwise evaluations."""

    def __init__(
        self,
        client: GeminiClient | None = None,
        model: str = DEFAULT_JUDGE_MODEL,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
    ) -> None:
        """Initialize judge with Gemini client and cache."""
        self.model = model
        cache = JudgeCache(cache_path)
        self.client = client or GeminiClient(model=model, cache=cache)

    def score_rubric(
        self,
        prompt_text: str,
        level: str,
        explanation: str,
        item_id: str = "",
    ) -> RubricScore:
        """Score explanation quality on a 1-5 scale across four dimensions.

        Dimensions are correctness, clarity, usefulness, and level fit.
        Return a RubricScore instance.
        """
        judge_prompt = (
            "You are an expert chess coach.\n"
            f"Score a chess explanation for a player at level: {level}.\n\n"
            "Rate the explanation on four criteria from 1 (very poor) to 5 (excellent):\n"
            "1. correctness (1-5): Moves, squares, and facts must match the chess position.\n"
            "2. clarity (1-5): The text must be simple, direct, and easy to understand.\n"
            "3. usefulness (1-5): The explanation must teach the main tactical or positional idea.\n"
            f"4. level_fit (1-5): Vocabulary and sentence length must fit player level: {level}.\n\n"
            f"Position and facts:\n{prompt_text}\n\n"
            f"Explanation:\n{explanation}\n\n"
            "Respond with a JSON object in this format:\n"
            '{"correctness": 1, "clarity": 1, "usefulness": 1, "level_fit": 1, "reason": "Short summary."}'
        )

        meta = {"item_id": item_id, "type": "rubric", "level": level}
        resp = self.client.generate_json(judge_prompt, task_type="rubric", cache_meta=meta)

        def clamp_score(val: Any) -> int:
            try:
                num = int(val)
                return max(1, min(5, num))
            except (ValueError, TypeError):
                return 3

        correctness = clamp_score(resp.get("correctness", 3))
        clarity = clamp_score(resp.get("clarity", 3))
        usefulness = clamp_score(resp.get("usefulness", 3))
        level_fit = clamp_score(resp.get("level_fit", 3))
        overall = round((correctness + clarity + usefulness + level_fit) / 4.0, 2)
        reason = str(resp.get("reason", ""))

        return RubricScore(
            correctness=correctness,
            clarity=clarity,
            usefulness=usefulness,
            level_fit=level_fit,
            overall=overall,
            reason=reason,
        )

    def score_rubric_batch(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[RubricSummary, list[dict[str, Any]]]:
        """Score a collection of completions using the rubric judge.

        Calculate average scores per dimension and set overall average.
        Return RubricSummary and item details.
        """
        if not records:
            empty_summary = RubricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)
            return empty_summary, []

        scores: list[dict[str, Any]] = []
        for rec in records:
            item_id = str(rec.get("id", "unknown"))
            prompt_str = ensure_prompt_text(rec)
            level = str(rec.get("level", "club"))
            raw_text = str(rec.get("completion") or rec.get("output") or rec.get("text") or "")
            cleaned = clean_completion_text(raw_text)

            score = self.score_rubric(
                prompt_text=prompt_str,
                level=level,
                explanation=cleaned,
                item_id=item_id,
            )
            score_dict = score.to_dict()
            score_dict["id"] = item_id
            scores.append(score_dict)

        count = len(scores)
        mean_correctness = round(sum(s["correctness"] for s in scores) / count, 2)
        mean_clarity = round(sum(s["clarity"] for s in scores) / count, 2)
        mean_usefulness = round(sum(s["usefulness"] for s in scores) / count, 2)
        mean_level_fit = round(sum(s["level_fit"] for s in scores) / count, 2)
        mean_overall = round(sum(s["overall"] for s in scores) / count, 2)

        summary = RubricSummary(
            mean_correctness=mean_correctness,
            mean_clarity=mean_clarity,
            mean_usefulness=mean_usefulness,
            mean_level_fit=mean_level_fit,
            mean_overall=mean_overall,
            total_evaluated=count,
        )
        return summary, scores

    def _call_pairwise_prompt(
        self,
        prompt_text: str,
        level: str,
        exp_a: str,
        exp_b: str,
        meta: dict[str, Any],
    ) -> tuple[str, str]:
        """Run single pairwise evaluation prompt.

        Return normalized winner in (A, B, tie) and reason.
        """
        judge_prompt = (
            "You are an expert chess coach.\n"
            f"Compare two chess explanations for a player at level: {level}.\n\n"
            "Evaluation criteria:\n"
            "1. Correctness: Moves, squares, and facts must match the chess position.\n"
            "2. Clarity: The text must be simple, direct, and easy to understand.\n"
            "3. Usefulness: The explanation must teach the main tactical or positional idea.\n"
            f"4. Level Fit: Vocabulary and sentence length must fit player level: {level}.\n\n"
            f"Position and facts:\n{prompt_text}\n\n"
            f"Explanation A:\n{exp_a}\n\n"
            f"Explanation B:\n{exp_b}\n\n"
            "Select the better explanation.\n"
            "If both explanations are equal in quality, select tie.\n"
            "Respond with a JSON object in this format:\n"
            '{"winner": "A", "reason": "Short explanation in one or two sentences."}\n'
            'Allowed values for winner are "A", "B", or "tie".'
        )

        resp = self.client.generate_json(
            judge_prompt,
            task_type="pairwise",
            cache_meta=meta,
        )
        raw_winner = str(resp.get("winner", "tie")).strip()
        norm_winner = "tie"
        if raw_winner in ("A", "a", "Explanation A", "explanation A"):
            norm_winner = "A"
        elif raw_winner in ("B", "b", "Explanation B", "explanation B"):
            norm_winner = "B"
        reason = str(resp.get("reason", ""))
        return norm_winner, reason

    def compare_pair(
        self,
        prompt_text: str,
        level: str,
        candidate_text: str,
        opponent_text: str,
        item_id: str = "",
    ) -> PairwiseComparison:
        """Run swap-consistent pairwise comparison between two explanations.

        Evaluate Order 1 (candidate as A, opponent as B).
        Evaluate Order 2 (opponent as A, candidate as B).
        If both orders pick the same underlying model, declare that winner.
        If the picks disagree or either ties, declare tie.
        Return PairwiseComparison.
        """
        # Order 1: A=candidate, B=opponent
        meta1 = {"item_id": item_id, "order": 1, "type": "pairwise"}
        win1, reason1 = self._call_pairwise_prompt(
            prompt_text,
            level,
            candidate_text,
            opponent_text,
            meta=meta1,
        )
        if win1 == "A":
            vote1 = "candidate"
        elif win1 == "B":
            vote1 = "opponent"
        else:
            vote1 = "tie"

        # Order 2: A=opponent, B=candidate
        meta2 = {"item_id": item_id, "order": 2, "type": "pairwise"}
        win2, reason2 = self._call_pairwise_prompt(
            prompt_text,
            level,
            opponent_text,
            candidate_text,
            meta=meta2,
        )
        if win2 == "A":
            vote2 = "opponent"
        elif win2 == "B":
            vote2 = "candidate"
        else:
            vote2 = "tie"

        # Swap consistency rule
        swap_agreed = (vote1 == vote2)
        if swap_agreed:
            final_winner = vote1
        else:
            final_winner = "tie"

        return PairwiseComparison(
            item_id=item_id,
            order1_winner=win1,
            order1_vote=vote1,
            order1_reason=reason1,
            order2_winner=win2,
            order2_vote=vote2,
            order2_reason=reason2,
            winner=final_winner,
            swap_agreed=swap_agreed,
        )

    def compare_opponent_batch(
        self,
        pairs: list[dict[str, Any]],
        opponent_name: str,
    ) -> tuple[PairwiseSummary, list[PairwiseComparison]]:
        """Run swap-consistent pairwise evaluation against one opponent.

        Each pair dict must contain prompt, level, candidate text, and opponent text.
        Calculate candidate win rate, tie rate, and swap agreement rate.
        Return PairwiseSummary and list of PairwiseComparison items.
        """
        if not pairs:
            empty_summary = PairwiseSummary(
                opponent_name=opponent_name,
                total_pairs=0,
                candidate_wins=0,
                opponent_wins=0,
                ties=0,
                candidate_win_rate=0.0,
                candidate_win_tie_rate=0.0,
                swap_agreement_rate=0.0,
            )
            return empty_summary, []

        results: list[PairwiseComparison] = []
        for pair in pairs:
            item_id = str(pair.get("id", "unknown"))
            prompt_str = ensure_prompt_text(pair)
            level = str(pair.get("level", "club"))
            cand = clean_completion_text(str(pair.get("candidate", "")))
            opp = clean_completion_text(str(pair.get("opponent", "")))

            comp = self.compare_pair(
                prompt_text=prompt_str,
                level=level,
                candidate_text=cand,
                opponent_text=opp,
                item_id=item_id,
            )
            results.append(comp)

        total = len(results)
        w = sum(1 for r in results if r.winner == "candidate")
        l = sum(1 for r in results if r.winner == "opponent")
        t = sum(1 for r in results if r.winner == "tie")
        agreed = sum(1 for r in results if r.swap_agreed)

        win_rate = round(100.0 * w / total, 2)
        win_tie_rate = round(100.0 * (w + t) / total, 2)
        agree_rate = round(100.0 * agreed / total, 2)

        summary = PairwiseSummary(
            opponent_name=opponent_name,
            total_pairs=total,
            candidate_wins=w,
            opponent_wins=l,
            ties=t,
            candidate_win_rate=win_rate,
            candidate_win_tie_rate=win_tie_rate,
            swap_agreement_rate=agree_rate,
        )
        return summary, results


def run_judge_eval(
    candidate_records: list[dict[str, Any]],
    opponent_records: dict[str, list[dict[str, Any]]] | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Execute complete judge evaluation for a model.

    Run rubric scoring on candidate completions.
    Run blind pairwise comparisons against all supplied baseline opponents.
    Return full evaluation dictionary.
    """
    judge = GeminiJudge(
        client=GeminiClient(api_key=api_key, model=judge_model, cache=JudgeCache(cache_path)),
        model=judge_model,
        cache_path=cache_path,
    )

    rubric_summary, rubric_items = judge.score_rubric_batch(candidate_records)

    pairwise_summaries: dict[str, dict[str, Any]] = {}
    pairwise_details: dict[str, list[dict[str, Any]]] = {}

    if opponent_records:
        candidate_map = {str(r.get("id")): r for r in candidate_records}
        for opp_name, opp_list in opponent_records.items():
            opp_map = {str(r.get("id")): r for r in opp_list}
            matched_pairs: list[dict[str, Any]] = []

            for item_id, cand_rec in candidate_map.items():
                if item_id in opp_map:
                    opp_rec = opp_map[item_id]
                    matched_pairs.append({
                        "id": item_id,
                        "prompt": cand_rec.get("prompt") or cand_rec.get("user_message"),
                        "level": cand_rec.get("level", "club"),
                        "candidate": cand_rec.get("completion") or cand_rec.get("output"),
                        "opponent": opp_rec.get("completion") or opp_rec.get("output"),
                    })

            if matched_pairs:
                summary, comps = judge.compare_opponent_batch(matched_pairs, opp_name)
                pairwise_summaries[opp_name] = summary.to_dict()
                pairwise_details[opp_name] = [c.to_dict() for c in comps]

    return {
        "model": judge_model,
        "rubric": {
            "summary": rubric_summary.to_dict(),
            "items": rubric_items,
        },
        "pairwise": {
            "summaries": pairwise_summaries,
            "details": pairwise_details,
        },
    }


def main() -> None:
    """CLI entry point for Gemini judge."""
    parser = argparse.ArgumentParser(description="Evaluate explanation quality with Gemini judge.")
    parser.add_argument("--eval-results", type=str, required=True, help="Path to evaluation results JSONL")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model name")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH), help="Path to cache JSONL")
    args = parser.parse_args()

    print(f"Second LLM Judge ({args.judge_model})")
    records: list[dict[str, Any]] = []
    with open(args.eval_results, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    results = run_judge_eval(
        candidate_records=records,
        judge_model=args.judge_model,
        cache_path=args.cache_path,
    )
    print("Rubric Summary:")
    print(json.dumps(results["rubric"]["summary"], indent=2))


if __name__ == "__main__":
    main()
