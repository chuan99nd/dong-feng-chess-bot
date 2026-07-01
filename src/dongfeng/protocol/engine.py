"""The universal engine (bot) contract for Dong Feng.

This module defines the vocabulary and the :class:`Engine` Protocol that *every*
bot in the ecosystem implements — our own neural / search bots, a Pikafish
wrapper, a random-mover baseline, or any third-party engine. Coding all callers
(match runners, arenas, the CLI, the MCP server) against this single Protocol
means any conforming engine is a drop-in for any other.

All positions are exchanged as FEN strings and all moves as ICCS
:class:`~dongfeng.core.types.Move` objects, so engines never leak backend types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..core.types import Move


@dataclass(slots=True)
class EngineInfo:
    """Static identification for an engine.

    Attributes:
        name: Human-readable engine name.
        author: Author / origin.
        options: Declared option name -> current (string) value.
    """

    name: str
    author: str
    options: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SearchLimits:
    """Bounds on a single search.

    Any subset may be set; an engine honors whichever limits it supports. All
    ``None`` means "search at the engine's own discretion".

    Attributes:
        movetime_ms: Fixed thinking time for this move, in milliseconds.
        depth: Maximum search depth in plies.
        nodes: Maximum nodes to search.
        wtime_ms: Red (White) clock remaining, in milliseconds.
        btime_ms: Black clock remaining, in milliseconds.
    """

    movetime_ms: int | None = None
    depth: int | None = None
    nodes: int | None = None
    wtime_ms: int | None = None
    btime_ms: int | None = None


@dataclass(slots=True)
class ScoredMove:
    """A candidate move annotated with evaluation signals.

    Attributes:
        move: The move.
        score_cp: Evaluation in centipawns from the side-to-move's perspective.
        win_prob: Win probability in ``[0, 1]`` (e.g. from engine WDL output).
        policy_prob: Prior/policy probability in ``[0, 1]`` (for distillation).
        pv: Principal variation starting with ``move``.
    """

    move: Move
    score_cp: int | None = None
    win_prob: float | None = None
    policy_prob: float | None = None
    pv: list[Move] = field(default_factory=list)


@dataclass(slots=True)
class Analysis:
    """The result of analyzing a position.

    Attributes:
        moves: Scored candidate moves, best first by convention.
        depth: Depth reached, in plies.
        nodes: Nodes searched.
        time_ms: Wall-clock time spent, in milliseconds.
    """

    moves: list[ScoredMove]
    depth: int = 0
    nodes: int = 0
    time_ms: int = 0

    @property
    def best(self) -> ScoredMove:
        """The best scored move.

        Raises:
            IndexError: if the analysis contains no moves.
        """
        if not self.moves:
            raise IndexError("analysis has no moves")
        return self.moves[0]


@runtime_checkable
class Engine(Protocol):
    """The universal Xiangqi bot contract.

    Any bot — ours or third-party — implements these methods. Positions are FEN
    strings, moves are ICCS :class:`~dongfeng.core.types.Move` objects. A typical
    lifecycle is::

        engine.new_game()
        engine.set_position(fen, moves)
        analysis = engine.analyze(SearchLimits(depth=20))
        move = analysis.best.move
        # or, when only the move is needed:
        move = engine.bestmove(SearchLimits(movetime_ms=1000))
    """

    def id(self) -> EngineInfo:
        """Return static engine identification (name, author, options)."""
        ...

    def new_game(self) -> None:
        """Reset any per-game state (transposition tables, history, etc.)."""
        ...

    def set_position(self, fen: str, moves: list[Move]) -> None:
        """Set the root position to ``fen`` then apply ``moves`` in order."""
        ...

    def analyze(self, limits: SearchLimits) -> Analysis:
        """Search the current position under ``limits`` and return an :class:`Analysis`."""
        ...

    def bestmove(self, limits: SearchLimits) -> Move:
        """Search the current position under ``limits`` and return the best move."""
        ...

    def set_option(self, name: str, value: str) -> None:
        """Set an engine option by name to a string ``value``."""
        ...

    def stop(self) -> None:
        """Request that any in-progress search stop as soon as possible."""
        ...
