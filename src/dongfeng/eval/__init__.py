"""Dong Feng evaluation: move-generator perft and engine-vs-engine matches.

Public surface:
    perft                              -- real leaf-node counter over core.Board
    MatchResult, play_match,           -- engine match & Elo estimation (stubs, M4)
    estimate_elo

``perft`` is fully implemented and useful today; the match/rating pieces are stubs
landing in milestone M4.
"""

from __future__ import annotations

from .match import MatchResult, estimate_elo, play_match
from .perft import perft

__all__ = [
    "MatchResult",
    "estimate_elo",
    "perft",
    "play_match",
]
