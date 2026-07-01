"""Dong Feng evaluation: perft, move-accuracy, and engine-vs-engine matches.

Public surface:
    perft                              -- real leaf-node counter over core.Board
    AccuracyResult, move_accuracy      -- top-1 move-match accuracy vs games
    MatchResult, play_match,           -- engine arena & Elo estimation
    estimate_elo
"""

from __future__ import annotations

from .accuracy import AccuracyResult, move_accuracy
from .match import MatchResult, estimate_elo, play_match
from .perft import perft

__all__ = [
    "AccuracyResult",
    "MatchResult",
    "estimate_elo",
    "move_accuracy",
    "perft",
    "play_match",
]
