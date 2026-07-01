"""Tests for the engine arena and Elo estimation (M2)."""

from __future__ import annotations

from dongfeng.engines import RandomEngine
from dongfeng.eval import MatchResult, estimate_elo, play_match
from dongfeng.protocol.engine import SearchLimits


def test_estimate_elo_even_score_is_zero() -> None:
    assert abs(estimate_elo(MatchResult(wins=5, draws=0, losses=5))) < 1e-6


def test_estimate_elo_monotonic() -> None:
    strong = estimate_elo(MatchResult(wins=9, draws=0, losses=1))
    weak = estimate_elo(MatchResult(wins=1, draws=0, losses=9))
    assert strong > 0 > weak


def test_play_match_counts_add_up() -> None:
    a, b = RandomEngine(seed=1), RandomEngine(seed=2)
    res = play_match(a, b, games=4, limits=SearchLimits(movetime_ms=1), max_plies=30)
    assert res.games == 4
    assert res.wins + res.draws + res.losses == 4
    assert res.elo_diff is not None
