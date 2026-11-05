"""Storage module for the human rating arena.

This module records comparison votes to local files.
It synchronizes recorded votes with a private Hugging Face dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
import uuid

from huggingface_hub import HfApi


@dataclass
class Vote:
    """Record of a single human preference vote."""

    timestamp: str
    session_id: str
    rater_id: str
    item_id: str
    comparison_set: str
    fen: str
    move_san: str
    level: str
    detail: str
    model_a: str
    model_b: str
    choice: str
    winner_model: str
    vote_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert the vote to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vote:
        """Create a Vote instance from a dictionary."""
        return cls(
            vote_id=data.get("vote_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            session_id=data.get("session_id", "default_session"),
            rater_id=data.get("rater_id", "anonymous"),
            item_id=data.get("item_id", ""),
            comparison_set=data.get("comparison_set", "unknown"),
            fen=data.get("fen", ""),
            move_san=data.get("move_san", ""),
            level=data.get("level", "club"),
            detail=data.get("detail", "short"),
            model_a=data.get("model_a", ""),
            model_b=data.get("model_b", ""),
            choice=data.get("choice", "same"),
            winner_model=data.get("winner_model", "tie"),
            comment=data.get("comment", ""),
        )


def save_vote_locally(vote: Vote, filepath: Path | str = "votes.jsonl") -> Path:
    """Save a vote record to a local JSONL file.

    When the parent directory does not exist, create the directory.
    Append the serialized vote record as a new line.
    """
    target_path = Path(filepath)
    if target_path.parent and not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(vote.to_dict()) + "\n")

    return target_path


def load_local_votes(filepath: Path | str = "votes.jsonl") -> list[Vote]:
    """Load vote records from a local JSONL file.

    When the file does not exist, return an empty list.
    When a line is empty or starts with a comment mark, skip the line.
    """
    target_path = Path(filepath)
    if not target_path.exists():
        return []

    votes: list[Vote] = []
    with target_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                continue
            try:
                data = json.loads(clean_line)
                votes.append(Vote.from_dict(data))
            except json.JSONDecodeError:
                continue

    return votes


def upload_votes_to_hub(
    filepath: Path | str = "votes.jsonl",
    repo_id: str | None = None,
    token: str | None = None,
    api: Any | None = None,
) -> bool:
    """Upload the votes file to a private Hugging Face dataset.

    When repo_id is None, read RATING_DATASET_REPO from the environment.
    When token is None, read HF_TOKEN or HF_ACCESS_TOKEN from the environment.
    When credentials or repository settings are missing, return False.
    When the local vote file does not exist, return False.
    """
    target_repo = repo_id or os.environ.get("RATING_DATASET_REPO")
    target_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")

    if not target_repo or not target_token:
        return False

    target_path = Path(filepath)
    if not target_path.exists():
        return False

    client = api if api is not None else HfApi(token=target_token)

    try:
        iso_now = datetime.now(timezone.utc).isoformat()
        client.upload_file(
            path_or_fileobj=str(target_path),
            path_in_repo="votes.jsonl",
            repo_id=target_repo,
            repo_type="dataset",
            commit_message=f"Sync human rating votes at {iso_now}",
        )
        return True
    except Exception as exc:
        print(f"Failed to upload votes to Hugging Face: {exc}")
        return False


def compute_summary_stats(votes: list[Vote]) -> dict[str, Any]:
    """Compute summary statistics grouped by comparison set.

    Calculate total votes, winner counts, tie counts, and win rates.
    Include aggregated values across all comparison sets.
    """
    stats: dict[str, Any] = {
        "by_set": {},
        "overall": {
            "total_votes": 0,
            "ties": 0,
            "winner_counts": {},
            "win_rates": {},
            "decisive_win_rates": {},
            "tie_rate": 0.0,
        },
    }

    if not votes:
        return stats

    for vote in votes:
        set_name = vote.comparison_set or "unknown"
        if set_name not in stats["by_set"]:
            stats["by_set"][set_name] = {
                "total_votes": 0,
                "ties": 0,
                "winner_counts": {},
                "win_rates": {},
                "decisive_win_rates": {},
                "tie_rate": 0.0,
            }

        set_stats = stats["by_set"][set_name]
        overall = stats["overall"]

        set_stats["total_votes"] += 1
        overall["total_votes"] += 1

        if vote.winner_model == "tie":
            set_stats["ties"] += 1
            overall["ties"] += 1
        else:
            winner = vote.winner_model
            set_stats["winner_counts"][winner] = set_stats["winner_counts"].get(winner, 0) + 1
            overall["winner_counts"][winner] = overall["winner_counts"].get(winner, 0) + 1

    def _finalize_rates(group: dict[str, Any]) -> None:
        total = group["total_votes"]
        ties = group["ties"]
        decisive_total = total - ties

        group["tie_rate"] = round(ties / total, 4) if total > 0 else 0.0

        for model, wins in group["winner_counts"].items():
            group["win_rates"][model] = round(wins / total, 4) if total > 0 else 0.0
            group["decisive_win_rates"][model] = (
                round(wins / decisive_total, 4) if decisive_total > 0 else 0.0
            )

    for group in stats["by_set"].values():
        _finalize_rates(group)

    _finalize_rates(stats["overall"])

    return stats


def format_summary_markdown(stats: dict[str, Any]) -> str:
    """Format summary statistics into a markdown table.

    Return a clean markdown representation for display in the interface.
    """
    rows: list[str] = [
        "| Comparison Set | Total Votes | Wins (Leading Model) | Second Wins | Ties | Tie Rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    by_set: dict[str, Any] = stats.get("by_set", {})
    if not by_set:
        return "No votes recorded yet."

    for set_name, data in by_set.items():
        total = data["total_votes"]
        ties = data["ties"]
        tie_rate = f"{data['tie_rate'] * 100:.1f}%"

        winners = data["winner_counts"]
        second_wins = 0
        other_summary = "None"
        for model_name, win_count in winners.items():
            if "second" in model_name.lower():
                second_wins += win_count
            else:
                other_summary = f"{model_name}: {win_count}"

        rows.append(
            f"| `{set_name}` | {total} | {other_summary} | Second: {second_wins} | {ties} | {tie_rate} |"
        )

    overall = stats.get("overall", {})
    if overall.get("total_votes", 0) > 0:
        total = overall["total_votes"]
        ties = overall["ties"]
        tie_rate = f"{overall['tie_rate'] * 100:.1f}%"
        second_wins = sum(
            c for m, c in overall["winner_counts"].items() if "second" in m.lower()
        )
        competitor_wins = sum(
            c for m, c in overall["winner_counts"].items() if "second" not in m.lower()
        )
        rows.append(
            f"| **Overall Total** | **{total}** | **Competitors: {competitor_wins}** | **Second: {second_wins}** | **{ties}** | **{tie_rate}** |"
        )

    return "\n".join(rows)
