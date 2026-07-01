"""Match & rating: play engines against each other and estimate Elo.

Runs head-to-head games between two :class:`~dongfeng.protocol.engine.Engine`
instances, alternating colors, with the board driven by
:func:`dongfeng.core.new_board` for legality and termination (in Xiangqi a side
with no legal move LOSES). Elo difference is derived from the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core import STARTING_FEN, GameResult, new_board
from ..core.types import Color
from ..protocol.engine import Engine, SearchLimits


@dataclass(slots=True)
class MatchResult:
    """Aggregate outcome of an engine-vs-engine match (from engine A's view)."""

    wins: int = 0
    draws: int = 0
    losses: int = 0
    elo_diff: float | None = None

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Points per game for engine A (win=1, draw=0.5)."""
        n = self.games
        return (self.wins + 0.5 * self.draws) / n if n else 0.0


def _play_one(red: Engine, black: Engine, limits: SearchLimits, max_plies: int) -> GameResult:
    """Play a single game; return the result. A side that has no/illegal move loses."""
    board = new_board(STARTING_FEN)
    red.new_game()
    black.new_game()
    played: list = []
    for _ in range(max_plies):
        if board.is_game_over():
            return board.result()
        mover = red if board.turn is Color.RED else black
        mover.set_position(STARTING_FEN, list(played))
        move = mover.bestmove(limits)
        if not board.is_legal(move):  # safety: an engine that emits an illegal move forfeits
            return GameResult.BLACK_WIN if board.turn is Color.RED else GameResult.RED_WIN
        board.push(move)
        played.append(move)
    return GameResult.DRAW  # hit the ply cap


def play_match(
    engine_a: Engine,
    engine_b: Engine,
    *,
    games: int,
    limits: SearchLimits,
    max_plies: int = 300,
) -> MatchResult:
    """Play ``games`` games between two engines (alternating colors) and aggregate."""
    result = MatchResult()
    for g in range(games):
        a_is_red = g % 2 == 0
        red, black = (engine_a, engine_b) if a_is_red else (engine_b, engine_a)
        outcome = _play_one(red, black, limits, max_plies)
        if outcome is GameResult.DRAW:
            result.draws += 1
        else:
            red_won = outcome is GameResult.RED_WIN
            a_won = red_won if a_is_red else not red_won
            if a_won:
                result.wins += 1
            else:
                result.losses += 1
    result.elo_diff = estimate_elo(result)
    return result


def estimate_elo(result: MatchResult) -> float:
    """Estimate the Elo difference (engine A minus engine B) from the score.

    Uses ``-400 * log10(1/score - 1)``, clamped for whitewash scores.
    """
    n = result.games
    if n == 0:
        return 0.0
    score = result.score
    score = min(max(score, 0.5 / n), 1.0 - 0.5 / n)  # avoid infinities at 0/1
    return -400.0 * math.log10(1.0 / score - 1.0)
