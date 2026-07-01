"""A seedable random-legal-mover baseline engine.

:class:`RandomEngine` is the simplest possible conforming
:class:`~dongfeng.protocol.engine.Engine`: it picks uniformly at random among the
legal moves of the current position. It exists as a smoke-test opponent, a
conformance target, and a floor to measure real bots against.

Legality comes entirely from the core rules backend
(:func:`dongfeng.core.new_board`); this engine contains no Xiangqi rules of its
own. Randomness is via :class:`random.Random`, seedable with
``set_option("Seed", "<int>")`` for reproducible games.
"""

from __future__ import annotations

import random

from ..core import STARTING_FEN, Board, Move, new_board
from ..protocol.engine import Analysis, EngineInfo, ScoredMove, SearchLimits

# How many random legal moves to surface in an :class:`Analysis`.
_ANALYSIS_MOVES = 3


class RandomEngine:
    """A conforming :class:`~dongfeng.protocol.engine.Engine` that plays randomly.

    The internal board is (re)built by :meth:`set_position`; construction sets it
    to the standard starting position so the engine is usable immediately.
    """

    _board: Board
    _rng: random.Random
    _seed: int | None

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._board = new_board(STARTING_FEN)

    # -- identity -----------------------------------------------------------

    def id(self) -> EngineInfo:
        options = {"Seed": "" if self._seed is None else str(self._seed)}
        return EngineInfo(
            name="Dong Feng Random",
            author="Dong Feng contributors",
            options=options,
        )

    # -- lifecycle ----------------------------------------------------------

    def new_game(self) -> None:
        # No per-game state beyond the board itself; reset it to the start so a
        # fresh game does not inherit a stale position.
        self._board = new_board(STARTING_FEN)

    def set_position(self, fen: str, moves: list[Move]) -> None:
        board = new_board(fen)
        for m in moves:
            board.push(m)
        self._board = board

    # -- search -------------------------------------------------------------

    def analyze(self, limits: SearchLimits) -> Analysis:
        # ``limits`` is accepted for protocol compatibility; a random mover does
        # no real search, so limits do not affect the (instant) result.
        _ = limits
        legal = self._board.legal_moves()
        if not legal:
            return Analysis(moves=[], depth=0, nodes=0, time_ms=0)
        picks = self._rng.sample(legal, k=min(_ANALYSIS_MOVES, len(legal)))
        scored = [ScoredMove(move=m) for m in picks]
        return Analysis(moves=scored, depth=0, nodes=len(legal), time_ms=0)

    def bestmove(self, limits: SearchLimits) -> Move:
        _ = limits
        legal = self._board.legal_moves()
        if not legal:
            raise ValueError("no legal moves in the current position")
        return self._rng.choice(legal)

    # -- options / control --------------------------------------------------

    def set_option(self, name: str, value: str) -> None:
        if name.lower() == "seed":
            seed: int | None
            if value == "":
                seed = None
            else:
                try:
                    seed = int(value)
                except ValueError as exc:
                    raise ValueError(f"Seed must be an integer, got {value!r}") from exc
            self._seed = seed
            self._rng = random.Random(seed)
        # Unknown options are silently ignored, matching engine-protocol norms.

    def stop(self) -> None:
        # Search is instantaneous; nothing to interrupt.
        return None
