"""Match & rating: play engines against each other and estimate Elo (stub).

Roadmap milestone: **M4**.

This module will run head-to-head matches between two
:class:`~dongfeng.protocol.engine.Engine` instances (our neural engine, a Pikafish
wrapper, a random baseline, ...), track wins/draws/losses under Xiangqi rules
(remember: stalemate is a LOSS for the side to move, see
:class:`dongfeng.core.types.GameResult`), and estimate a relative Elo difference
from the results.

Planned pieces (finalized in M4):

* :class:`MatchResult` — aggregate W/D/L and a derived Elo estimate with a
  confidence interval.
* :func:`play_match` — play ``n`` games between two engines, alternating colors,
  each move produced via the universal ``Engine.bestmove`` under given limits, with
  the board driven by :func:`dongfeng.core.new_board` for legality and termination.
* :func:`estimate_elo` — convert a W/D/L score into an Elo difference
  (``-400 * log10(1/score - 1)``) with an error bar.

Until M4, the entry points below raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol.engine import Engine, SearchLimits


@dataclass(slots=True)
class MatchResult:
    """Aggregate outcome of an engine-vs-engine match.

    Attributes:
        wins: Games won by the first engine.
        draws: Drawn games.
        losses: Games lost by the first engine.
        elo_diff: Estimated Elo difference (first minus second), or ``None`` if not
            yet computed.
    """

    wins: int = 0
    draws: int = 0
    losses: int = 0
    elo_diff: float | None = None


def play_match(
    engine_a: Engine,
    engine_b: Engine,
    *,
    games: int,
    limits: SearchLimits,
) -> MatchResult:
    """Play ``games`` games between two engines and return the aggregate result.

    Raises:
        NotImplementedError: always — planned for milestone M4.
    """
    raise NotImplementedError("play_match is planned: M4")


def estimate_elo(result: MatchResult) -> float:
    """Estimate the Elo difference (engine A minus engine B) from a match result.

    Raises:
        NotImplementedError: always — planned for milestone M4.
    """
    raise NotImplementedError("estimate_elo is planned: M4")
