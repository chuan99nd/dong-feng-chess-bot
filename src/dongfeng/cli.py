"""Dong Feng command-line interface.

A small `Typer <https://typer.tiangolo.com/>`_ app that makes the engine
ecosystem usable from a terminal:

* ``version``  — print the package version.
* ``board``    — pretty-print a position (via ``rich``).
* ``selfplay`` — play an engine against itself and print the game.
* ``play``     — play a human (typing ICCS moves) against an engine.
* ``ucci``     — run the UCCI/UCI protocol adapter on real stdin/stdout, so the
  bot is a drop-in engine for any GUI or tournament manager.

Run as ``dfc <command>`` (see ``[project.scripts]``) or ``python -m dongfeng.cli``.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import STARTING_FEN, Board, GameResult, Move, new_board
from .engines import PikafishEngine, RandomEngine
from .protocol.engine import Engine, SearchLimits
from .protocol.ucci import ProtocolAdapter

app = typer.Typer(
    name="dfc",
    help="Dong Feng — Xiangqi (Chinese chess) engine toolkit.",
    add_completion=False,
    no_args_is_help=True,
)

_console = Console()

# -- engine factory ---------------------------------------------------------

_ENGINE_CHOICES = ("random", "pikafish")


def _make_engine(name: str, seed: int | None = None) -> Engine:
    """Build an :class:`Engine` by short name (``random`` / ``pikafish``)."""
    key = name.lower()
    if key == "random":
        return RandomEngine(seed=seed)
    if key == "pikafish":
        return PikafishEngine()
    raise typer.BadParameter(f"unknown engine {name!r}; choose one of {', '.join(_ENGINE_CHOICES)}")


def _result_text(result: GameResult) -> str:
    return {
        GameResult.RED_WIN: "Red wins",
        GameResult.BLACK_WIN: "Black wins",
        GameResult.DRAW: "Draw",
        GameResult.ONGOING: "Ongoing",
    }[result]


def _board_text(board: Board) -> str:
    """Return the board's ASCII rendering as a clean multi-line string.

    The core backend's ``ascii()`` may return a ``repr`` of a list of row
    strings (a quirk of the underlying ``text_view()``); if so, evaluate it back
    into rows and join with newlines. Otherwise the string is used as-is.
    """
    raw = board.ascii()
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        import ast

        try:
            rows = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return raw
        if isinstance(rows, list) and all(isinstance(r, str) for r in rows):
            return "\n".join(rows)
    return raw


def _render_board(board: Board) -> Panel:
    """Wrap the board's ASCII rendering in a titled ``rich`` panel."""
    body = _board_text(board)
    subtitle = f"{board.turn.value} to move"
    return Panel(body, title="Xiangqi", subtitle=subtitle, expand=False)


# -- commands ---------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the Dong Feng package version."""
    from . import __version__

    _console.print(f"dongfeng {__version__}")


@app.command()
def board(
    fen: str = typer.Option(STARTING_FEN, "--fen", help="Position to display (Xiangqi FEN)."),
) -> None:
    """Pretty-print a Xiangqi position from a FEN string."""
    b = new_board(fen)
    _console.print(_render_board(b))


@app.command()
def selfplay(
    engine: str = typer.Option("random", "--engine", help="Engine name for both sides."),
    max_moves: int = typer.Option(200, "--max-moves", help="Max plies before adjudicating a draw."),
    seed: int | None = typer.Option(None, "--seed", help="Seed for reproducible random play."),
) -> None:
    """Play an engine against itself and print the moves and result."""
    board_state = new_board(STARTING_FEN)
    red = _make_engine(engine, seed=seed)
    black = _make_engine(engine, seed=None if seed is None else seed + 1)
    for e in (red, black):
        e.new_game()

    table = Table(title=f"Self-play: {engine} vs {engine}")
    table.add_column("#", justify="right")
    table.add_column("Side")
    table.add_column("Move")

    played: list[Move] = []
    ply = 0
    while ply < max_moves and not board_state.is_game_over():
        mover = red if board_state.turn.value == "red" else black
        mover.set_position(STARTING_FEN, played)
        move = mover.bestmove(SearchLimits(movetime_ms=100))
        board_state.push(move)
        played.append(move)
        ply += 1
        table.add_row(str(ply), "Red" if ply % 2 == 1 else "Black", move.iccs)

    _console.print(table)
    result = board_state.result()
    _console.print(_render_board(board_state))
    _console.print(f"[bold]Result:[/bold] {_result_text(result)} after {ply} plies")


@app.command()
def play(
    fen: str = typer.Option(STARTING_FEN, "--fen", help="Starting position (Xiangqi FEN)."),
    engine: str = typer.Option("random", "--engine", help="Engine to play against."),
) -> None:
    """Play against an engine: you type ICCS moves; the engine replies.

    You play the side to move in ``--fen`` (Red in the standard start). Enter a
    4-char ICCS move like ``h2e2``; type ``quit`` to resign, ``moves`` to list
    the legal moves.
    """
    board_state = new_board(fen)
    opponent = _make_engine(engine)
    opponent.new_game()
    human = board_state.turn
    played: list[Move] = []

    _console.print(_render_board(board_state))
    while not board_state.is_game_over():
        if board_state.turn == human:
            raw = _console.input(f"[bold]{human.value}[/bold] move (ICCS, or 'quit'/'moves'): ")
            command = raw.strip().lower()
            if command in ("quit", "exit", "resign"):
                _console.print("You resigned.")
                return
            if command == "moves":
                legal = " ".join(sorted(m.iccs for m in board_state.legal_moves()))
                _console.print(legal or "(no legal moves)")
                continue
            try:
                move = Move.from_iccs(command)
            except ValueError as exc:
                _console.print(f"[red]Invalid move:[/red] {exc}")
                continue
            if not board_state.is_legal(move):
                _console.print("[red]Illegal move in this position.[/red]")
                continue
        else:
            opponent.set_position(fen, played)
            move = opponent.bestmove(SearchLimits(movetime_ms=500))
            _console.print(f"{engine} plays [bold]{move.iccs}[/bold]")

        board_state.push(move)
        played.append(move)
        _console.print(_render_board(board_state))

    _console.print(f"[bold]Result:[/bold] {_result_text(board_state.result())}")


@app.command()
def ucci(
    engine: str = typer.Option("random", "--engine", help="Engine to expose over the protocol."),
) -> None:
    """Run the UCCI/UCI protocol adapter on real stdin/stdout.

    This turns Dong Feng into a drop-in Xiangqi engine: point any UCCI or UCI
    (Pikafish-dialect) GUI or tournament manager at ``dfc ucci`` and it will
    handshake, accept positions, and reply with ``bestmove``.
    """
    adapter = ProtocolAdapter(_make_engine(engine))
    adapter.run(sys.stdin, sys.stdout)


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` ``dfc``)."""
    app()


if __name__ == "__main__":
    main()
