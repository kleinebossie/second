"""Gradio demo Space for Second-1.0-2B chess explanations.

Users can paste a PGN game or FEN position, choose a move, and generate a coaching explanation.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import os
from pathlib import Path
import re
import shutil
import sys
import threading
from typing import Any

import chess
import chess.engine
import chess.pgn
import chess.svg

try:
    import gradio as gr
except ImportError:
    gr = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SPACE_ROOT = Path(__file__).resolve().parent
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from core.python import (
    SYSTEM_MESSAGE,
    build_user_message,
    classify_verdict,
    format_qwen_chat_prompt,
)

DEFAULT_HF_REPO = "joebos/Second-1.0-2B-GGUF"
DEFAULT_GGUF_FILE = "Second-1.0-2B-Q4_K_M.gguf"
POSITION_ONLY_LABEL = "Position Overview (no specific move)"

PRESET_EXAMPLES: dict[str, tuple[str, str]] = {
    "Opening Blunder (Scholars / Fried Liver setup)": (
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 Nxe4",
        "4... Nxe4 (ply 8)",
    ),
    "Tactical Fork FEN (After 4. Ng5)": (
        "r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4",
        "4... d5 (d7d5)",
    ),
    "Position Overview (Italian Game)": (
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        POSITION_ONLY_LABEL,
    ),
}

_MODEL_LOCK = threading.Lock()
_LLAMA_INSTANCE: Any = None


@dataclass
class MoveEntry:
    """Single selectable move extracted from a PGN or FEN position."""

    label: str
    fen_before: str
    move_uci: str
    move_san: str


@dataclass
class ParsedChessInput:
    """Parsed chess input containing base FEN and selectable moves."""

    mode: str
    base_fen: str
    entries: list[MoveEntry]
    default_label: str


def _format_ply_label(ply_index: int, fullmove: int, turn: chess.Color, san: str) -> str:
    """Build a human-readable dropdown label for a PGN move."""
    prefix = f"{fullmove}." if turn == chess.WHITE else f"{fullmove}..."
    return f"{prefix} {san} (ply {ply_index})"


def parse_chess_input(raw_text: str) -> ParsedChessInput:
    """Parse a PGN string or FEN string into selectable moves.

    When input matches a valid FEN, return Position Overview plus legal moves.
    When input parses as a PGN with moves, return each played move in order.
    When input is invalid, raise ValueError with a clear message.
    """
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Paste a PGN or FEN first.")

    # Attempt FEN parsing directly when input looks like a single-line FEN.
    if "\n" not in text and "/" in text and len(text.split()) >= 4:
        try:
            board = chess.Board(text)
            entries: list[MoveEntry] = []
            for move in sorted(board.legal_moves, key=lambda m: board.san(m)):
                san = board.san(move)
                entries.append(
                    MoveEntry(
                        label=f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'} {san} ({move.uci()})",
                        fen_before=board.fen(),
                        move_uci=move.uci(),
                        move_san=san,
                    )
                )
            return ParsedChessInput(
                mode="fen",
                base_fen=board.fen(),
                entries=entries,
                default_label=POSITION_ONLY_LABEL,
            )
        except ValueError:
            pass

    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise ValueError("Could not read input as PGN or FEN.")

    board = game.board()
    entries = []
    ply_index = 0
    for node in game.mainline():
        move = node.move
        ply_index += 1
        san = board.san(move)
        label = _format_ply_label(ply_index, board.fullmove_number, board.turn, san)
        entries.append(
            MoveEntry(
                label=label,
                fen_before=board.fen(),
                move_uci=move.uci(),
                move_san=san,
            )
        )
        board.push(move)

    if not entries:
        if game.errors:
            raise ValueError("Invalid PGN or FEN input.")
        if "FEN" in game.headers:
            return parse_chess_input(game.headers["FEN"])
        raise ValueError("Could not find any chess moves or a valid FEN.")

    return ParsedChessInput(
        mode="pgn",
        base_fen=entries[-1].fen_before,
        entries=entries,
        default_label=entries[-1].label,
    )


def find_stockfish_binary() -> str | None:
    """Locate a working Stockfish binary on the host system."""
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    for name in ("stockfish", "/usr/games/stockfish", "/usr/bin/stockfish"):
        resolved = shutil.which(name) if "/" not in name else (name if Path(name).is_file() else None)
        if resolved:
            return resolved

    local_fallback = (
        PROJECT_ROOT
        / "spikes"
        / "colab-stockfish"
        / "stockfish_bin"
        / "stockfish"
        / "stockfish-linux-x86-64-universal"
    )
    if local_fallback.is_file():
        return str(local_fallback)

    return None


def _engine_infos_to_eval_dict(board: chess.Board, infos: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    """Convert python-chess analysis info list to Second EngineEval dictionary."""
    pvs: list[dict[str, Any]] = []
    for info in infos:
        score = info["score"].white()
        if score.is_mate():
            score_dict = {"type": "mate", "value": int(score.mate() or 0)}
        else:
            score_dict = {"type": "cp", "value": int(score.score() or 0)}
        pv_moves = [m.uci() for m in info.get("pv", [])[:6]]
        pvs.append({"score": score_dict, "moves": pv_moves})

    return {
        "fen": board.fen(),
        "depth": depth,
        "pvs": pvs,
    }


def analyze_with_stockfish(
    fen_before: str,
    move_uci: str | None = None,
    depth: int = 14,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run MultiPV Stockfish analysis before and optionally after a move."""
    binary = find_stockfish_binary()
    if not binary:
        raise RuntimeError(
            "Stockfish binary not found. Install stockfish or set STOCKFISH_PATH."
        )

    board_before = chess.Board(fen_before)
    limit = chess.engine.Limit(depth=depth)

    with chess.engine.SimpleEngine.popen_uci(binary) as engine:
        info_before = engine.analyse(board_before, limit, multipv=3)
        eval_before = _engine_infos_to_eval_dict(board_before, info_before, depth)

        eval_after: dict[str, Any] | None = None
        if move_uci:
            board_after = board_before.copy()
            board_after.push(chess.Move.from_uci(move_uci))
            info_after = engine.analyse(board_after, limit, multipv=2)
            eval_after = _engine_infos_to_eval_dict(board_after, info_after, depth)

    return eval_before, eval_after


def clean_model_output(raw_text: str) -> str:
    """Strip any stray thinking tags and clean whitespace from model output."""
    text = raw_text or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def render_demo_board_svg(
    fen_before: str,
    move_uci: str | None = None,
    best_move_uci: str | None = None,
) -> str:
    """Render SVG chess board after an optional move and highlight the best move."""
    board = chess.Board(fen_before)
    last_move: chess.Move | None = None
    orientation = board.turn

    if move_uci:
        last_move = chess.Move.from_uci(move_uci)
        board.push(last_move)

    arrows: list[chess.svg.Arrow] = []
    if best_move_uci and best_move_uci != move_uci:
        try:
            best_m = chess.Move.from_uci(best_move_uci)
            arrows.append(chess.svg.Arrow(best_m.from_square, best_m.to_square, color="#34d399"))
        except ValueError:
            pass

    return chess.svg.board(
        board=board,
        orientation=orientation,
        lastmove=last_move,
        arrows=arrows,
        size=380,
    )


def get_llama_model() -> Any:
    """Load Second-1.0-2B Q4_K_M GGUF once in a thread-safe singleton."""
    global _LLAMA_INSTANCE
    if _LLAMA_INSTANCE is not None:
        return _LLAMA_INSTANCE

    with _MODEL_LOCK:
        if _LLAMA_INSTANCE is not None:
            return _LLAMA_INSTANCE

        from llama_cpp import Llama

        local_gguf = os.environ.get("SECOND_GGUF_PATH")
        if local_gguf and Path(local_gguf).is_file():
            _LLAMA_INSTANCE = Llama(
                model_path=local_gguf,
                n_ctx=1024,
                n_threads=max(2, os.cpu_count() or 2),
                verbose=False,
            )
        else:
            _LLAMA_INSTANCE = Llama.from_pretrained(
                repo_id=os.environ.get("SECOND_HF_REPO", DEFAULT_HF_REPO),
                filename=os.environ.get("SECOND_GGUF_FILE", DEFAULT_GGUF_FILE),
                n_ctx=1024,
                n_threads=max(2, os.cpu_count() or 2),
                verbose=False,
            )
        return _LLAMA_INSTANCE


def run_demo_inference(prompt_text: str, detail: str) -> str:
    """Generate a greedy coaching explanation using Second-1.0-2B GGUF."""
    llm = get_llama_model()
    max_tokens = 320 if detail == "long" else 180
    result = llm(
        prompt_text,
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    raw = str(result["choices"][0]["text"])
    return clean_model_output(raw)


def on_parse_input(raw_text: str, override_choice: str | None = None) -> tuple[Any, str, str]:
    """Update move dropdown and initial board SVG when PGN or FEN changes."""
    try:
        parsed = parse_chess_input(raw_text)
    except ValueError as exc:
        empty_svg = render_demo_board_svg(chess.STARTING_FEN)
        return gr.update(choices=[POSITION_ONLY_LABEL], value=POSITION_ONLY_LABEL), empty_svg, f"Input error: {exc}"

    choices = [POSITION_ONLY_LABEL] + [e.label for e in parsed.entries]
    selected = override_choice if override_choice in choices else parsed.default_label
    default_entry = next((e for e in parsed.entries if e.label == selected), None)
    svg = render_demo_board_svg(
        default_entry.fen_before if default_entry else parsed.base_fen,
        move_uci=default_entry.move_uci if default_entry else None,
    )
    return gr.update(choices=choices, value=selected), svg, f"Loaded {len(parsed.entries)} move option(s)."


def on_explain(
    raw_text: str,
    selected_label: str,
    level: str,
    detail: str,
) -> tuple[str, str, str]:
    """Analyze the selected move or position and generate Second's explanation."""
    parsed = parse_chess_input(raw_text)
    entry = next((e for e in parsed.entries if e.label == selected_label), None)

    if entry is None:
        task = "explain_position"
        fen_before = parsed.base_fen
        move_uci = None
    else:
        task = "explain_move"
        fen_before = entry.fen_before
        move_uci = entry.move_uci

    eval_before, eval_after = analyze_with_stockfish(fen_before, move_uci=move_uci)
    best_move_uci: str | None = None
    pvs = eval_before.get("pvs", [])
    if pvs and pvs[0].get("moves"):
        best_move_uci = pvs[0]["moves"][0]

    user_msg = build_user_message(
        task=task,
        level=level,
        detail=detail,
        fen=fen_before,
        eval_before=eval_before,
        eval_after=eval_after,
        move_uci=move_uci,
    )
    prompt = format_qwen_chat_prompt(user_msg, SYSTEM_MESSAGE)
    svg = render_demo_board_svg(fen_before, move_uci=move_uci, best_move_uci=best_move_uci)

    explanation = run_demo_inference(prompt, detail=detail)
    header = (
        f"### {entry.move_san} — {classify_verdict(eval_before, eval_after, move_uci)}"
        if entry and move_uci
        else "### Position Overview"
    )
    return svg, f"{header}\n\n{explanation}", user_msg


def build_demo() -> Any:
    """Construct the Gradio Blocks application."""
    if gr is None:
        raise ImportError("Gradio is required to run the demo Space.")

    first_name = next(iter(PRESET_EXAMPLES))
    first_input, first_choice = PRESET_EXAMPLES[first_name]
    parsed_init = parse_chess_input(first_input)
    init_choices = [POSITION_ONLY_LABEL] + [e.label for e in parsed_init.entries]
    init_entry = next((e for e in parsed_init.entries if e.label == first_choice), parsed_init.entries[-1])
    init_svg = render_demo_board_svg(init_entry.fen_before, init_entry.move_uci)

    with gr.Blocks(title="Second — Local AI Chess Coach") as demo:
        gr.Markdown(
            "# Second (`Second-1.0-2B`)\n"
            "Paste a **PGN** or **FEN**, choose a move, and read a Stockfish-grounded explanation."
        )

        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=list(PRESET_EXAMPLES.keys()),
                    value=first_name,
                    label="Example Positions",
                )
                chess_input = gr.Textbox(
                    value=first_input,
                    lines=4,
                    label="PGN or FEN",
                    placeholder="Paste a PGN game or FEN position...",
                )
                parse_btn = gr.Button("Load Moves", variant="secondary")
                move_choice = gr.Dropdown(
                    choices=init_choices,
                    value=init_entry.label,
                    label="Move to Explain",
                )
                with gr.Row():
                    level = gr.Radio(["beginner", "club"], value="club", label="Coach Style")
                    detail = gr.Radio(["short", "long"], value="short", label="Length")
                explain_btn = gr.Button("Explain Move", variant="primary")
                status_md = gr.Markdown(f"Loaded {len(parsed_init.entries)} move option(s).")

            with gr.Column(scale=1):
                board_svg = gr.HTML(value=init_svg, label="Board")
                explanation_md = gr.Markdown("Click **Explain Move** to run Second-1.0-2B.")
                with gr.Accordion("Prompt & Engine Facts (Debug)", open=False):
                    facts_box = gr.Textbox(lines=10, interactive=False, label="Prompt sent to Second")

        def _load_preset(name: str) -> tuple[str, Any, str, str]:
            text, target = PRESET_EXAMPLES[name]
            dd, svg, msg = on_parse_input(text, override_choice=target)
            return text, dd, svg, msg

        preset.change(_load_preset, inputs=[preset], outputs=[chess_input, move_choice, board_svg, status_md])
        parse_btn.click(on_parse_input, inputs=[chess_input], outputs=[move_choice, board_svg, status_md])
        explain_btn.click(
            on_explain,
            inputs=[chess_input, move_choice, level, detail],
            outputs=[board_svg, explanation_md, facts_box],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
