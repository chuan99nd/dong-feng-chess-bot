"""Tests for the UCCI/UCI protocol adapter using in-memory streams."""

from __future__ import annotations

import io

from dongfeng.core import STARTING_FEN, new_board
from dongfeng.engines import RandomEngine
from dongfeng.protocol.ucci import ProtocolAdapter


def _run(script: str) -> list[str]:
    """Feed ``script`` (newline-separated commands) through the adapter.

    Returns the adapter's output as a list of non-empty stripped lines.
    """
    adapter = ProtocolAdapter(RandomEngine(seed=0))
    out = io.StringIO()
    adapter.run(io.StringIO(script), out)
    return [line for line in out.getvalue().splitlines() if line.strip()]


def _start_legal_iccs() -> set[str]:
    return {m.iccs for m in new_board(STARTING_FEN).legal_moves()}


def test_uci_handshake_and_bestmove() -> None:
    """A full UCI session yields id lines, uciok, readyok, and a legal bestmove."""
    lines = _run("uci\nisready\nucinewgame\nposition startpos\ngo movetime 10\nquit\n")

    assert any(line.startswith("id name ") for line in lines)
    assert any(line.startswith("id author ") for line in lines)
    assert "uciok" in lines
    assert "readyok" in lines

    bestmoves = [line for line in lines if line.startswith("bestmove ")]
    assert len(bestmoves) == 1
    move = bestmoves[0].split()[1]
    assert move in _start_legal_iccs()


def test_ucci_handshake_uses_ucciok() -> None:
    """The UCCI dialect answers the handshake with ``ucciok`` (not ``uciok``)."""
    lines = _run("ucci\nisready\nposition startpos\ngo movetime 10\nquit\n")
    assert "ucciok" in lines
    assert "uciok" not in lines
    assert any(line.startswith("bestmove ") for line in lines)


def test_position_fen_with_moves() -> None:
    """``position fen ... moves ...`` is parsed and yields a legal bestmove."""
    fen = STARTING_FEN
    lines = _run(f"uci\nposition fen {fen} moves h2e2 h9g7\ngo depth 2\nquit\n")

    board = new_board(fen)
    for iccs in ("h2e2", "h9g7"):
        board.push(next(m for m in board.legal_moves() if m.iccs == iccs))
    legal_after = {m.iccs for m in board.legal_moves()}

    bestmoves = [line for line in lines if line.startswith("bestmove ")]
    assert len(bestmoves) == 1
    assert bestmoves[0].split()[1] in legal_after


def test_go_without_flags_still_returns_bestmove() -> None:
    """A bare ``go`` (no limits) still produces a single legal bestmove."""
    lines = _run("uci\nposition startpos\ngo\nquit\n")
    bestmoves = [line for line in lines if line.startswith("bestmove ")]
    assert len(bestmoves) == 1
    assert bestmoves[0].split()[1] in _start_legal_iccs()


def test_setoption_seed_is_forwarded() -> None:
    """Setting the Seed option makes the (otherwise random) bestmove deterministic."""

    def best_after_seed() -> str:
        adapter = ProtocolAdapter(RandomEngine())
        out = io.StringIO()
        script = "uci\nsetoption name Seed value 42\nposition startpos\ngo movetime 5\nquit\n"
        adapter.run(io.StringIO(script), out)
        lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("bestmove ")]
        return lines[0].split()[1]

    assert best_after_seed() == best_after_seed()
