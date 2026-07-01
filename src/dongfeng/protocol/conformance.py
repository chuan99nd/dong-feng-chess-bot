"""Engine conformance harness.

:func:`run_conformance` exercises a factory that produces :class:`Engine`
instances and returns a list of human-readable failure messages — an empty list
means the engine conforms. Move legality is validated against the real rules
backend via :func:`dongfeng.core.new_board`.

This lets any bot (ours or third-party) be checked against the universal contract
before it is trusted in a match or arena.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core import Move, new_board
from ..core.fen import STARTING_FEN
from .engine import Analysis, Engine, EngineInfo, SearchLimits

# A short, unambiguous opening line in ICCS (from the starting position):
#   Red central cannon, Black horse.
_OPENING_MOVES = [Move.from_iccs("h2e2"), Move.from_iccs("h9g7")]

# Modest default limits so conformance stays fast regardless of engine strength.
_LIMITS = SearchLimits(depth=4, movetime_ms=200)


def _legal_iccs(fen: str, moves: list[Move]) -> set[str]:
    """Return the set of legal ICCS move strings after applying ``moves`` to ``fen``."""
    board = new_board(fen)
    for m in moves:
        board.push(m)
    return {m.iccs for m in board.legal_moves()}


def run_conformance(make_engine: Callable[[], Engine]) -> list[str]:
    """Run the engine conformance suite.

    Args:
        make_engine: Zero-arg factory returning a fresh :class:`Engine`.

    Returns:
        A list of failure messages. Empty means the engine passed every check.
    """
    failures: list[str] = []

    # --- id() returns an EngineInfo ---------------------------------------
    try:
        engine = make_engine()
        info = engine.id()
        if not isinstance(info, EngineInfo):
            failures.append(f"id() must return EngineInfo, got {type(info).__name__}")
        elif not info.name:
            failures.append("id().name must be a non-empty string")
    except Exception as exc:  # noqa: BLE001 - report any failure as a message
        failures.append(f"id() raised {type(exc).__name__}: {exc}")

    # --- bestmove() from the starting position is legal -------------------
    start_legal = _legal_iccs(STARTING_FEN, [])
    try:
        engine = make_engine()
        engine.new_game()
        engine.set_position(STARTING_FEN, [])
        mv = engine.bestmove(_LIMITS)
        if not isinstance(mv, Move):
            failures.append(f"bestmove() must return a Move, got {type(mv).__name__}")
        elif mv.iccs not in start_legal:
            failures.append(f"bestmove() from start returned illegal move: {mv.iccs}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"bestmove() from start raised {type(exc).__name__}: {exc}")

    # --- bestmove() still legal after a couple of moves ------------------
    after_legal = _legal_iccs(STARTING_FEN, _OPENING_MOVES)
    try:
        engine = make_engine()
        engine.new_game()
        engine.set_position(STARTING_FEN, _OPENING_MOVES)
        mv = engine.bestmove(_LIMITS)
        if not isinstance(mv, Move):
            failures.append(f"bestmove() after moves must return a Move, got {type(mv).__name__}")
        elif mv.iccs not in after_legal:
            failures.append(
                f"bestmove() after {[m.iccs for m in _OPENING_MOVES]} "
                f"returned illegal move: {mv.iccs}"
            )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"bestmove() after moves raised {type(exc).__name__}: {exc}")

    # --- new_game() then bestmove() works --------------------------------
    try:
        engine = make_engine()
        engine.new_game()
        engine.set_position(STARTING_FEN, [])
        mv = engine.bestmove(_LIMITS)
        if not isinstance(mv, Move) or mv.iccs not in start_legal:
            failures.append("new_game() then bestmove() did not yield a legal move")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"new_game() then bestmove() raised {type(exc).__name__}: {exc}")

    # --- analyze() returns an Analysis whose best is a legal move ---------
    try:
        engine = make_engine()
        engine.new_game()
        engine.set_position(STARTING_FEN, [])
        analysis = engine.analyze(_LIMITS)
        if not isinstance(analysis, Analysis):
            failures.append(f"analyze() must return an Analysis, got {type(analysis).__name__}")
        elif not analysis.moves:
            failures.append("analyze() returned an Analysis with no moves")
        else:
            best_move = analysis.best.move
            if not isinstance(best_move, Move):
                failures.append(
                    f"analyze().best.move must be a Move, got {type(best_move).__name__}"
                )
            elif best_move.iccs not in start_legal:
                failures.append(f"analyze().best from start is an illegal move: {best_move.iccs}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"analyze() raised {type(exc).__name__}: {exc}")

    return failures
