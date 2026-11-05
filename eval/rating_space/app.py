"""Human rating application for the Second chess project.

This module provides blind A/B evaluation of chess explanations.
It supports local testing and deployment to Hugging Face Spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any
import uuid

import chess

from eval.rating_space.board import extract_position_metadata, render_board_svg
from eval.rating_space.storage import (
    Vote,
    compute_summary_stats,
    format_summary_markdown,
    load_local_votes,
    save_vote_locally,
    upload_votes_to_hub,
)

try:
    import gradio as gr
except ImportError:
    gr = None

SET_CHOICES: list[str] = [
    "All Comparison Sets",
    "Second vs Base Qwen",
    "Second vs Gemini Flash",
    "Second vs Chess.com Coach",
]

SET_MAPPING: dict[str, str] = {
    "All Comparison Sets": "all",
    "Second vs Base Qwen": "second_vs_base_qwen",
    "Second vs Gemini Flash": "second_vs_gemini_flash",
    "Second vs Chess.com Coach": "second_vs_chesscom",
    "all": "all",
    "second_vs_base_qwen": "second_vs_base_qwen",
    "second_vs_gemini_flash": "second_vs_gemini_flash",
    "second_vs_chesscom": "second_vs_chesscom",
}


@dataclass
class ComparisonPair:
    """Single pair of explanations for a chess position."""

    id: str
    comparison_set: str
    fen: str
    move_san: str
    model_1_name: str
    model_1_text: str
    model_2_name: str
    model_2_text: str
    move_uci: str | None = None
    level: str = "club"
    detail: str = "short"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonPair:
        """Create a ComparisonPair instance from a dictionary."""
        item_id = str(data.get("id") or data.get("pair_id") or uuid.uuid4())
        comparison_set = str(data.get("comparison_set") or data.get("set") or "general")
        fen = str(data.get("fen") or chess.STARTING_FEN)
        move_san = str(data.get("move_san") or data.get("move_played") or "")
        move_uci = data.get("move_uci")
        level = str(data.get("level") or "club")
        detail = str(data.get("detail") or "short")

        m1_name = str(
            data.get("model_1_name")
            or data.get("model_second_name")
            or data.get("second_name")
            or "Second"
        )
        m1_text = str(
            data.get("model_1_text")
            or data.get("model_second_text")
            or data.get("second_text")
            or ""
        )

        m2_name = str(
            data.get("model_2_name")
            or data.get("model_competitor_name")
            or data.get("competitor_name")
            or "Competitor"
        )
        m2_text = str(
            data.get("model_2_text")
            or data.get("model_competitor_text")
            or data.get("competitor_text")
            or ""
        )

        return cls(
            id=item_id,
            comparison_set=comparison_set,
            fen=fen,
            move_san=move_san,
            move_uci=move_uci,
            level=level,
            detail=detail,
            model_1_name=m1_name,
            model_1_text=m1_text,
            model_2_name=m2_name,
            model_2_text=m2_text,
        )


@dataclass
class ActiveItem:
    """Blind presentation order for a comparison pair."""

    pair: ComparisonPair
    model_a_name: str
    model_b_name: str
    explanation_a: str
    explanation_b: str


def prepare_blind_item(pair: ComparisonPair, seed: int | None = None) -> ActiveItem:
    """Randomize the presentation order of two candidate explanations.

    When seed is provided, use the seed for deterministic order.
    The evaluator does not see the true model identities.
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    first_is_model_1 = rng.choice([True, False])

    if first_is_model_1:
        return ActiveItem(
            pair=pair,
            model_a_name=pair.model_1_name,
            model_b_name=pair.model_2_name,
            explanation_a=pair.model_1_text,
            explanation_b=pair.model_2_text,
        )
    return ActiveItem(
        pair=pair,
        model_a_name=pair.model_2_name,
        model_b_name=pair.model_1_name,
        explanation_a=pair.model_2_text,
        explanation_b=pair.model_1_text,
    )


def resolve_winner(choice: str, model_a: str, model_b: str) -> str:
    """Resolve the winning model name from rater choice.

    When choice is 'A', return model_a name.
    When choice is 'B', return model_b name.
    When choice is 'same' or 'tie', return 'tie'.
    When choice is invalid, raise a ValueError.
    """
    clean_choice = choice.strip()
    if clean_choice == "A":
        return model_a
    if clean_choice == "B":
        return model_b
    if clean_choice in {"same", "tie", "Same / Tie"}:
        return "tie"
    raise ValueError(f"Unknown choice: {choice}. Expected 'A', 'B', or 'same'.")


def filter_pairs(pairs: list[ComparisonPair], set_name: str) -> list[ComparisonPair]:
    """Filter comparison pairs by comparison set.

    When set_name is 'all' or empty, return all pairs.
    When set_name matches a key in SET_MAPPING, resolve the canonical name.
    """
    canonical = SET_MAPPING.get(set_name, set_name.lower().strip())
    if canonical in {"all", ""}:
        return list(pairs)

    return [p for p in pairs if p.comparison_set.lower() == canonical]


def _read_pairs_file(path: Path) -> list[ComparisonPair]:
    """Read comparison pairs from a JSONL file."""
    items: list[ComparisonPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            try:
                data = json.loads(clean)
                items.append(ComparisonPair.from_dict(data))
            except Exception:
                continue
    return items


def try_download_private_pairs(
    repo_id: str | None = None,
    token: str | None = None,
    filename: str = "chesscom_pairs.jsonl",
) -> list[ComparisonPair]:
    """Download private evaluation pairs from a Hugging Face dataset.

    When repo_id or token is missing, return an empty list.
    When the file cannot be downloaded, return an empty list.
    """
    target_repo = repo_id or os.environ.get("RATING_DATASET_REPO")
    target_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")

    if not target_repo or not target_token:
        return []

    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=target_repo,
            filename=filename,
            repo_type="dataset",
            token=target_token,
        )
        return _read_pairs_file(Path(downloaded_path))
    except Exception as exc:
        print(f"Could not download private pairs from Hugging Face: {exc}")
        return []


def load_pairs(
    filepath: Path | str | None = None,
    include_private: bool = True,
) -> list[ComparisonPair]:
    """Load comparison pairs from JSONL files.

    When filepath is provided, load pairs from that path.
    When filepath is None, load from pairs.jsonl or sample_pairs.jsonl.
    When include_private is True, append pairs from chesscom_pairs.jsonl if present.
    """
    pairs: list[ComparisonPair] = []

    if filepath is not None:
        target_path = Path(filepath)
        if target_path.exists():
            pairs.extend(_read_pairs_file(target_path))
        return pairs

    base_dir = Path(__file__).parent
    main_file = base_dir / "pairs.jsonl"
    if not main_file.exists():
        main_file = base_dir / "sample_pairs.jsonl"

    if main_file.exists():
        pairs.extend(_read_pairs_file(main_file))

    if include_private:
        private_file = base_dir / "chesscom_pairs.jsonl"
        if private_file.exists():
            pairs.extend(_read_pairs_file(private_file))
        else:
            downloaded = try_download_private_pairs()
            pairs.extend(downloaded)

    return pairs


def record_vote(
    active_item: ActiveItem,
    choice: str,
    rater_id: str = "anonymous",
    session_id: str = "",
    votes_path: Path | str = "votes.jsonl",
    sync_hf: bool = True,
    comment: str = "",
) -> Vote:
    """Record a vote and persist the vote record.

    Save the vote to the local JSONL file.
    When sync_hf is True, upload the updated votes file to Hugging Face.
    """
    winner = resolve_winner(choice, active_item.model_a_name, active_item.model_b_name)
    pair = active_item.pair
    vote = Vote.from_dict(
        {
            "session_id": session_id or str(uuid.uuid4()),
            "rater_id": rater_id or "anonymous",
            "item_id": pair.id,
            "comparison_set": pair.comparison_set,
            "fen": pair.fen,
            "move_san": pair.move_san,
            "level": pair.level,
            "detail": pair.detail,
            "model_a": active_item.model_a_name,
            "model_b": active_item.model_b_name,
            "choice": choice,
            "winner_model": winner,
            "comment": comment,
        }
    )

    save_vote_locally(vote, filepath=votes_path)

    if sync_hf:
        upload_votes_to_hub(filepath=votes_path)

    return vote


class RatingSession:
    """Manage the state of a human rating session."""

    def __init__(
        self,
        pairs: list[ComparisonPair],
        comparison_set: str = "all",
        rater_id: str = "anonymous",
        session_id: str | None = None,
        votes_path: Path | str = "votes.jsonl",
        sync_hf: bool = True,
        seed: int | None = None,
    ) -> None:
        self.all_pairs = list(pairs)
        self.comparison_set = comparison_set
        self.rater_id = rater_id
        self.session_id = session_id or str(uuid.uuid4())
        self.votes_path = Path(votes_path)
        self.sync_hf = sync_hf
        self.seed = seed
        self.votes_count = 0
        self.current_index = 0
        self.filtered_pairs = filter_pairs(self.all_pairs, self.comparison_set)
        self.active_item: ActiveItem | None = None
        self._load_current_active()

    def _load_current_active(self) -> None:
        """Load the active blind item for the current index."""
        if 0 <= self.current_index < len(self.filtered_pairs):
            pair = self.filtered_pairs[self.current_index]
            self.active_item = prepare_blind_item(pair, seed=self.seed)
        else:
            self.active_item = None

    def set_comparison_set(self, set_name: str) -> None:
        """Filter pairs by comparison set and reset index to zero."""
        self.comparison_set = set_name
        self.filtered_pairs = filter_pairs(self.all_pairs, self.comparison_set)
        self.current_index = 0
        self._load_current_active()

    def set_rater_id(self, rater_id: str) -> None:
        """Update the rater identifier for future votes."""
        clean = rater_id.strip()
        self.rater_id = clean if clean else "anonymous"

    def record_choice(self, choice: str, comment: str = "") -> Vote | None:
        """Record the rater choice for the current active item.

        Advance the session to the next pair after recording.
        When no active item exists, return None.
        """
        if self.active_item is None:
            return None

        vote = record_vote(
            active_item=self.active_item,
            choice=choice,
            rater_id=self.rater_id,
            session_id=self.session_id,
            votes_path=self.votes_path,
            sync_hf=self.sync_hf,
            comment=comment,
        )
        self.votes_count += 1
        self.current_index += 1
        self._load_current_active()
        return vote

    def skip(self) -> None:
        """Skip the current pair without recording a vote."""
        if self.active_item is not None:
            self.current_index += 1
            self._load_current_active()

    def get_progress_text(self) -> str:
        """Return the current progress status string."""
        total = len(self.filtered_pairs)
        if total == 0:
            return "No items available in this set."
        curr = min(self.current_index + 1, total)
        return f"Item {curr} of {total} | Total votes recorded: {self.votes_count}"

    def get_board_svg(self) -> str:
        """Render the board SVG for the current active item."""
        if self.active_item is None:
            return render_board_svg(chess.STARTING_FEN)
        pair = self.active_item.pair
        return render_board_svg(
            fen=pair.fen,
            move_san=pair.move_san,
            move_uci=pair.move_uci,
        )

    def get_metadata_markdown(self) -> str:
        """Return position metadata formatted as Markdown."""
        if self.active_item is None:
            return "No position selected."
        pair = self.active_item.pair
        meta = extract_position_metadata(pair.fen, pair.move_san)
        move_str = f"{meta['move_number']}. {meta['move_san']}"
        lines = [
            f"### Position Details",
            f"- **Move**: `{move_str}`",
            f"- **Side to move**: {meta['side_to_move']}",
            f"- **Target level**: `{pair.level}`",
            f"- **Detail mode**: `{pair.detail}`",
        ]
        if meta["is_check"]:
            lines.append("- **Status**: ⚠️ In Check")
        return "\n".join(lines)


def _render_session_view(session: RatingSession, status: str = "") -> tuple[Any, ...]:
    """Generate display outputs for the current session state."""
    board_svg = session.get_board_svg()
    meta_md = session.get_metadata_markdown()
    if session.active_item is not None:
        text_a = f"### Explanation A\n\n{session.active_item.explanation_a}"
        text_b = f"### Explanation B\n\n{session.active_item.explanation_b}"
    else:
        text_a = "### Review Finished\n\nYou reviewed all positions in this set."
        text_b = "### Thank You\n\nYour ratings improve the Second explanation model."

    progress_md = f"**{session.get_progress_text()}**"
    votes = load_local_votes(session.votes_path)
    stats_md = format_summary_markdown(compute_summary_stats(votes))

    return (
        session,
        board_svg,
        meta_md,
        text_a,
        text_b,
        progress_md,
        status,
        stats_md,
    )


def create_ui(
    pairs_path: Path | str | None = None,
    votes_path: Path | str = "votes.jsonl",
    sync_hf: bool = True,
) -> gr.Blocks:
    """Create the Gradio user interface for blind human rating.

    When Gradio is not installed, raise a RuntimeError.
    """
    if gr is None:
        raise RuntimeError("Gradio is not installed in the current environment.")

    pairs = load_pairs(pairs_path)

    with gr.Blocks(title="Second Human Rating Arena") as demo:
        gr.Markdown(
            "# ♟️ Second Human Rating Arena\n"
            "Compare chess explanations blindly. Select which explanation teaches the move better."
        )

        session_state = gr.State(
            lambda: RatingSession(
                pairs=pairs,
                votes_path=votes_path,
                sync_hf=sync_hf,
            )
        )

        with gr.Row():
            set_dropdown = gr.Dropdown(
                choices=SET_CHOICES,
                value="All Comparison Sets",
                label="Comparison Set",
                interactive=True,
            )
            rater_input = gr.Textbox(
                value="anonymous",
                label="Rater Nickname / ID",
                placeholder="Enter your name or handle",
                interactive=True,
            )
            progress_display = gr.Markdown(value="Loading arena items...")

        with gr.Row():
            with gr.Column(scale=1):
                board_html = gr.HTML()
                meta_display = gr.Markdown()

            with gr.Column(scale=2):
                with gr.Row():
                    box_a = gr.Markdown()
                    box_b = gr.Markdown()

                with gr.Row():
                    btn_a = gr.Button("Prefer A", variant="primary")
                    btn_b = gr.Button("Prefer B", variant="primary")
                    btn_same = gr.Button("Same / Tie", variant="secondary")
                    btn_skip = gr.Button("Skip", variant="stop")

                status_display = gr.Markdown(value="")

        with gr.Accordion("Rating Statistics", open=False):
            stats_display = gr.Markdown(value="No votes recorded yet.")
            refresh_btn = gr.Button("Refresh Statistics", variant="secondary")

        output_targets = [
            session_state,
            board_html,
            meta_display,
            box_a,
            box_b,
            progress_display,
            status_display,
            stats_display,
        ]

        def on_init(session: RatingSession) -> tuple[Any, ...]:
            return _render_session_view(session, status="Ready for rating.")

        def on_vote_a(session: RatingSession) -> tuple[Any, ...]:
            vote = session.record_choice("A")
            msg = f"Recorded preference for A (Item: `{vote.item_id}`)." if vote else "No active item."
            return _render_session_view(session, status=msg)

        def on_vote_b(session: RatingSession) -> tuple[Any, ...]:
            vote = session.record_choice("B")
            msg = f"Recorded preference for B (Item: `{vote.item_id}`)." if vote else "No active item."
            return _render_session_view(session, status=msg)

        def on_vote_same(session: RatingSession) -> tuple[Any, ...]:
            vote = session.record_choice("same")
            msg = f"Recorded tie vote (Item: `{vote.item_id}`)." if vote else "No active item."
            return _render_session_view(session, status=msg)

        def on_skip(session: RatingSession) -> tuple[Any, ...]:
            session.skip()
            return _render_session_view(session, status="Skipped item.")

        def on_set_change(session: RatingSession, selected: str) -> tuple[Any, ...]:
            session.set_comparison_set(SET_MAPPING.get(selected, "all"))
            return _render_session_view(session, status=f"Switched set to: {selected}")

        def on_rater_change(session: RatingSession, new_id: str) -> RatingSession:
            session.set_rater_id(new_id)
            return session

        def on_refresh(session: RatingSession) -> str:
            votes = load_local_votes(session.votes_path)
            return format_summary_markdown(compute_summary_stats(votes))

        demo.load(
            fn=on_init,
            inputs=[session_state],
            outputs=output_targets,
        )

        btn_a.click(
            fn=on_vote_a,
            inputs=[session_state],
            outputs=output_targets,
        )
        btn_b.click(
            fn=on_vote_b,
            inputs=[session_state],
            outputs=output_targets,
        )
        btn_same.click(
            fn=on_vote_same,
            inputs=[session_state],
            outputs=output_targets,
        )
        btn_skip.click(
            fn=on_skip,
            inputs=[session_state],
            outputs=output_targets,
        )
        set_dropdown.change(
            fn=on_set_change,
            inputs=[session_state, set_dropdown],
            outputs=output_targets,
        )
        rater_input.change(
            fn=on_rater_change,
            inputs=[session_state, rater_input],
            outputs=[session_state],
        )
        refresh_btn.click(
            fn=on_refresh,
            inputs=[session_state],
            outputs=[stats_display],
        )

    return demo


def build_app(pairs_path: Path | str | None = None) -> Any:
    """Build and return the rating application."""
    return create_ui(pairs_path=pairs_path)


if __name__ == "__main__":
    app = build_app()
    app.launch()
