"""Move-prediction accuracy against held-out games.

Replays each game and, at every ply, asks the engine for its move given the
history so far, comparing it to the move actually played. Top-1 accuracy is the
standard behavior-cloning quality signal (how often the engine's chosen move
matches the human/reference move).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..data.base import Game
from ..protocol.engine import Engine, SearchLimits


@dataclass(slots=True)
class AccuracyResult:
    """Aggregate move-prediction accuracy."""

    positions: int = 0
    top1: int = 0

    @property
    def top1_acc(self) -> float:
        return self.top1 / self.positions if self.positions else 0.0


def move_accuracy(
    engine: Engine,
    games: Iterable[Game],
    *,
    max_positions: int = 2_000,
    limits: SearchLimits | None = None,
) -> AccuracyResult:
    """Return top-1 move-match accuracy of ``engine`` over up to ``max_positions`` plies."""
    limits = limits or SearchLimits(movetime_ms=50)
    res = AccuracyResult()
    for game in games:
        played: list = []
        for move in game.moves:
            engine.set_position(game.start_fen, list(played))
            try:
                pred = engine.bestmove(limits)
            except ValueError:
                break  # no legal move (terminal) — stop this game
            res.positions += 1
            if pred == move:
                res.top1 += 1
            played.append(move)
            if res.positions >= max_positions:
                return res
    return res
