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

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import STARTING_FEN, Board, GameResult, Move, new_board
from .data import build_shards, iter_games_in
from .engines import PikafishEngine, RandomEngine
from .protocol.engine import Engine, SearchLimits
from .protocol.ucci import ProtocolAdapter
from .tokenizer import BoardTokenizer, MoveTokenizer

app = typer.Typer(
    name="dfc",
    help="Dong Feng — Xiangqi (Chinese chess) engine toolkit.",
    add_completion=False,
    no_args_is_help=True,
)

_console = Console()

# -- engine factory ---------------------------------------------------------

_ENGINE_CHOICES = ("random", "pikafish", "neural")


def _make_engine(name: str, seed: int | None = None) -> Engine:
    """Build an :class:`Engine` by short name (``random`` / ``pikafish`` / ``neural``).

    ``neural`` loads a trained checkpoint from ``$DONGFENG_CKPT`` (or a random-init
    model if unset); it needs the optional ``model`` extra (torch).
    """
    key = name.lower()
    if key == "random":
        return RandomEngine(seed=seed)
    if key == "pikafish":
        return PikafishEngine()
    if key == "neural":
        from .inference.transformer_engine import TransformerEngine  # noqa: PLC0415

        return TransformerEngine(os.environ.get("DONGFENG_CKPT") or None)
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


# -- data pipeline (M1) -----------------------------------------------------

data_app = typer.Typer(
    name="data",
    help="Corpus ingestion + tokenization (M1).",
    no_args_is_help=True,
)
app.add_typer(data_app)


def _manifest_path() -> Path:
    """Resolve the artifacts manifest (env override, else ./manifest.json)."""
    return Path(os.environ.get("DONGFENG_MANIFEST", "manifest.json"))


def _load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"tokenizers": [], "datasets": [], "checkpoints": [], "runs": []}


def _save_manifest(m: dict[str, Any]) -> None:
    with open(_manifest_path(), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
        fh.write("\n")


def _register_tokenizers(m: dict[str, Any]) -> None:
    """Ensure the manifest lists the built-in tokenizers (id + vocab_size)."""
    known = {t.get("id") for t in m.get("tokenizers", [])}
    for tok, kind in ((MoveTokenizer(), "move"), (BoardTokenizer(), "board")):
        if tok.id not in known:
            m.setdefault("tokenizers", []).append(
                {"id": tok.id, "kind": kind, "vocab_size": tok.vocab_size}
            )


def _upsert_dataset(m: dict[str, Any], entry: dict[str, Any]) -> None:
    """Insert or replace a dataset entry (matched by id) in the manifest."""
    datasets = m.setdefault("datasets", [])
    for i, d in enumerate(datasets):
        if d.get("id") == entry["id"]:
            datasets[i] = entry
            return
    datasets.append(entry)


@data_app.command("ingest")
def data_ingest(
    path: str = typer.Argument(..., help="Game file or directory (.pgn/.xqf/.cbf/.cbl/.cbr/.txt)."),
    out: str = typer.Option(..., "--out", help="Output dir for shards (under data/)."),
    dataset_id: str = typer.Option(..., "--id", help="Unique dataset id for the manifest."),
    shard_size: int = typer.Option(1_048_576, "--shard-size", help="Target token ids per shard."),
) -> None:
    """Parse games, tokenize to autoregressive shards, and index in the manifest."""
    created = datetime.now(UTC).isoformat(timespec="seconds")
    stats = build_shards(
        iter_games_in(path), out, tokenizer=MoveTokenizer(), shard_size=shard_size, created=created
    )
    m = _load_manifest()
    _register_tokenizers(m)
    _upsert_dataset(
        m,
        {
            "id": dataset_id,
            "path": out,
            "source": Path(path).name,
            "format": "mixed",
            "num_games": stats.num_games,
            "num_samples": stats.num_samples,
            "tokenizer": stats.tokenizer,
            "created": created,
            "notes": f"{stats.num_tokens} tokens across {stats.num_shards} shard(s); "
            f"{stats.skipped_games} game(s) skipped",
        },
    )
    _save_manifest(m)

    table = Table(title=f"Ingested dataset {dataset_id!r}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for label, value in (
        ("Games", stats.num_games),
        ("Plies (samples)", stats.num_samples),
        ("Tokens", stats.num_tokens),
        ("Shards", stats.num_shards),
        ("Skipped games", stats.skipped_games),
        ("Tokenizer", f"{stats.tokenizer} (vocab {stats.vocab_size})"),
    ):
        table.add_row(label, str(value))
    _console.print(table)
    _console.print(f"Shards + dataset_meta.json written to [bold]{out}[/bold]")


@data_app.command("stats")
def data_stats(
    dataset_id: str | None = typer.Option(None, "--id", help="Show only this dataset."),
) -> None:
    """Show indexed datasets and tokenizers from the manifest."""
    m = _load_manifest()
    datasets = m.get("datasets", [])
    if dataset_id is not None:
        datasets = [d for d in datasets if d.get("id") == dataset_id]
        if not datasets:
            _console.print(f"[red]No dataset with id {dataset_id!r}.[/red]")
            raise typer.Exit(1)

    if not datasets:
        _console.print("No datasets indexed yet. Run [bold]dfc data ingest[/bold].")
    else:
        table = Table(title="Datasets")
        for col in ("id", "source", "games", "samples", "tokenizer", "created"):
            table.add_column(col)
        for d in datasets:
            table.add_row(
                str(d.get("id")),
                str(d.get("source")),
                str(d.get("num_games")),
                str(d.get("num_samples")),
                str(d.get("tokenizer")),
                str(d.get("created")),
            )
        _console.print(table)

    tokenizers = m.get("tokenizers", [])
    if tokenizers:
        ttable = Table(title="Tokenizers")
        for col in ("id", "kind", "vocab_size"):
            ttable.add_column(col)
        for t in tokenizers:
            ttable.add_row(str(t.get("id")), str(t.get("kind")), str(t.get("vocab_size")))
        _console.print(ttable)


@data_app.command("tokenize")
def data_tokenize(
    text: str = typer.Argument(
        ..., help="ICCS move(s) like 'h2e2 h9g7', or a FEN with --board."
    ),
    board_mode: bool = typer.Option(
        False, "--board", help="Treat input as a FEN (BoardTokenizer)."
    ),
) -> None:
    """Demo encode/decode round-trip for the move or board tokenizer."""
    if board_mode:
        tok: Any = BoardTokenizer()
        ids = tok.encode(text)
        _console.print(f"[bold]{tok.id}[/bold] vocab={tok.vocab_size} tokens={len(ids)}")
        _console.print(f"ids: {ids}")
        _console.print(f"decode: {tok.decode(ids)}")
    else:
        tok = MoveTokenizer()
        ids = tok.encode(text)
        _console.print(f"[bold]{tok.id}[/bold] vocab={tok.vocab_size} tokens={len(ids)}")
        _console.print(f"ids: {ids}")
        _console.print(f"decode: {tok.decode(ids)}")


# -- training + eval (M2) ---------------------------------------------------


@app.command()
def train(
    data_dir: str = typer.Option(..., "--data", help="Dir of tokenized shards (dfc data ingest)."),
    out: str = typer.Option(..., "--out", help="Output dir for the checkpoint."),
    checkpoint_id: str = typer.Option(..., "--id", help="Checkpoint id for the manifest."),
    n_layer: int = typer.Option(4, "--layers"),
    n_embd: int = typer.Option(256, "--width"),
    n_head: int = typer.Option(4, "--heads"),
    block_size: int = typer.Option(256, "--block"),
    batch_size: int = typer.Option(64, "--batch"),
    lr: float = typer.Option(3e-4, "--lr"),
    max_steps: int = typer.Option(2000, "--steps"),
    warmup: int = typer.Option(100, "--warmup"),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps | cuda"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Behavior-cloning pretrain of the decoder-only transformer on move shards (M2)."""
    from .model import TransformerConfig, TransformerPolicy  # noqa: PLC0415
    from .training.base import TrainConfig  # noqa: PLC0415
    from .training.loop import bc_pretrain, resolve_device  # noqa: PLC0415

    tok = MoveTokenizer()
    model = TransformerPolicy(
        TransformerConfig(
            vocab_size=tok.vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
        )
    )
    resolved = resolve_device(device)
    _console.print(
        f"Training [bold]{checkpoint_id}[/bold]: {model.num_params():,} params on "
        f"[bold]{resolved}[/bold] for {max_steps} steps (vocab {tok.vocab_size})"
    )
    tcfg = TrainConfig(
        data_dir=Path(data_dir),
        out_dir=Path(out),
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup,
        max_steps=max_steps,
        device=device,
        seed=seed,
        checkpoint_every=max(max_steps // 10, 1),
    )
    ckpt = bc_pretrain(model, tcfg)

    created = datetime.now(UTC).isoformat(timespec="seconds")
    m = _load_manifest()
    checkpoints = m.setdefault("checkpoints", [])
    entry = {
        "id": checkpoint_id,
        "path": str(ckpt),
        "arch": f"transformer-{n_layer}L-{n_embd}d",
        "params": model.num_params(),
        "step": max_steps,
        "tokenizer": tok.id,
        "metrics": {},
        "created": created,
    }
    for i, c in enumerate(checkpoints):
        if c.get("id") == checkpoint_id:
            checkpoints[i] = entry
            break
    else:
        checkpoints.append(entry)
    _save_manifest(m)
    _console.print(f"[bold green]Saved[/bold green] checkpoint to {ckpt}")


eval_app = typer.Typer(
    name="eval", help="Strength & accuracy evaluation (M2+).", no_args_is_help=True
)
app.add_typer(eval_app)


@eval_app.command("accuracy")
def eval_accuracy(
    ckpt: str = typer.Option(..., "--ckpt", help="Checkpoint path."),
    data: str = typer.Option(..., "--data", help="Held-out games file/dir."),
    max_positions: int = typer.Option(1000, "--positions"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """Top-1 move-match accuracy of a checkpoint against held-out games."""
    from .eval import move_accuracy  # noqa: PLC0415
    from .inference.transformer_engine import TransformerEngine  # noqa: PLC0415

    engine = TransformerEngine(ckpt, device=device)
    res = move_accuracy(engine, iter_games_in(data), max_positions=max_positions)
    _console.print(
        f"top-1 accuracy: [bold]{res.top1_acc:.1%}[/bold] "
        f"({res.top1}/{res.positions} positions)"
    )


@eval_app.command("arena")
def eval_arena(
    ckpt: str = typer.Option(..., "--ckpt", help="Checkpoint path for the neural engine."),
    games: int = typer.Option(20, "--games"),
    opponent: str = typer.Option("random", "--opponent"),
    device: str = typer.Option("cpu", "--device"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Play the neural engine vs a baseline and report W/D/L + estimated Elo."""
    from .eval import play_match  # noqa: PLC0415
    from .inference.transformer_engine import TransformerEngine  # noqa: PLC0415

    neural = TransformerEngine(ckpt, device=device)
    baseline = _make_engine(opponent, seed=seed)
    res = play_match(neural, baseline, games=games, limits=SearchLimits(movetime_ms=10))
    _console.print(
        f"neural vs {opponent}: [bold]{res.wins}W-{res.draws}D-{res.losses}L[/bold] "
        f"(score {res.score:.1%}, Elo diff {res.elo_diff:+.0f})"
    )


@app.command()
def web(
    engine: str = typer.Option("neural", "--engine", help="random | neural."),
    ckpt: str | None = typer.Option(None, "--ckpt", help="Neural checkpoint (or $DONGFENG_CKPT)."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    no_open: bool = typer.Option(False, "--no-open", help="Do not auto-open the browser."),
) -> None:
    """Launch the browser UI to play against an engine (open http://host:port)."""
    from .serve import serve  # noqa: PLC0415

    serve(host=host, port=port, engine=engine, checkpoint=ckpt, open_browser=not no_open)


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` ``dfc``)."""
    app()


if __name__ == "__main__":
    main()
