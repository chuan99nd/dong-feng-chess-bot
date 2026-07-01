"""Data-layer contracts: game sources and training samples.

The data layer turns raw human/engine game records into a stream of typed
:class:`Sample` objects (one per position-to-move decision) that the tokenizer and
training loop consume. A :class:`GameSource` abstracts *where* games come from
(a PGN dump, an XQF archive, a scraped online corpus) behind a single iterator, so
the ingestion and dataset code is decoupled from any one file format.

Roadmap: concrete sources and the sharded dataset land in milestone **M1**. This
module defines only the stable contracts.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..core.types import Color, GameResult, Move


@dataclass(slots=True)
class Sample:
    """A single supervised training example: a position and the move played.

    One :class:`Sample` corresponds to one ply of a game — the position before the
    move (as a FEN) and the move made from it. Optional fields carry distillation /
    outcome signals used by later training stages (see
    :mod:`dongfeng.training`).

    Attributes:
        fen: The position before the move, as a Xiangqi FEN (see
            :mod:`dongfeng.core.fen`).
        move: The move played from ``fen``, in ICCS coordinates.
        turn: The side to move in ``fen``.
        result: The final game result (used to filter to winning-side moves, or as
            a value target). ``ONGOING`` when the outcome is unknown.
        win_prob: Optional engine WDL-derived win probability in ``[0, 1]`` for the
            side to move (distillation value target).
        weight: Optional sample weight for loss reweighting.
    """

    fen: str
    move: Move
    turn: Color
    result: GameResult = GameResult.ONGOING
    win_prob: float | None = None
    weight: float = 1.0


@dataclass(slots=True)
class Game:
    """A single parsed game: a start position plus the moves played.

    Attributes:
        start_fen: The initial position (usually the standard start; a FEN allows
            handicap / puzzle positions too).
        moves: The moves played, in order, in ICCS coordinates.
        result: The final result of the game.
        metadata: Free-form record metadata (event, players, date, source, ...).
    """

    start_fen: str
    moves: list[Move]
    result: GameResult = GameResult.ONGOING
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class GameSource(Protocol):
    """A stream of parsed games from some backing store or file format.

    Implementations wrap a concrete source (PGN, XQF, DhtmlXQ, a database export)
    and yield :class:`Game` objects. Consumers (ingestion, dataset building) code
    only against this Protocol.
    """

    def iter_games(self) -> Iterator[Game]:
        """Yield each :class:`Game` from this source, in source order."""
        ...
