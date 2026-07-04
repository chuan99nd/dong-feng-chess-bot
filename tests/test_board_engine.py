"""Tests for BoardTransformerEngine (WP4).

All tests skip when torch is unavailable (the 'model' extra is optional).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping board engine tests")

from dongfeng.core import STARTING_FEN, new_board  # noqa: E402  # type: ignore[import]
from dongfeng.inference.board_engine import BoardTransformerEngine  # noqa: E402
from dongfeng.protocol import run_conformance  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_engine() -> BoardTransformerEngine:
    """Return a random-init BoardTransformerEngine (no checkpoint needed)."""
    return BoardTransformerEngine(checkpoint=None, device="cpu")


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


def test_conformance() -> None:
    """BoardTransformerEngine must satisfy the universal engine conformance suite."""
    failures = run_conformance(make_engine)
    assert failures == [], "conformance failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Bestmove from start is legal
# ---------------------------------------------------------------------------


def test_bestmove_start_legal() -> None:
    """bestmove() from the starting position must return a legal move."""
    from dongfeng.protocol.engine import SearchLimits  # noqa: PLC0415

    engine = make_engine()
    engine.new_game()
    engine.set_position(STARTING_FEN, [])

    board = new_board(STARTING_FEN)
    legal_iccs = {m.iccs for m in board.legal_moves()}

    move = engine.bestmove(SearchLimits())
    assert move.iccs in legal_iccs, f"bestmove returned illegal move: {move.iccs}"


# ---------------------------------------------------------------------------
# Analyze: ordering and win_prob range
# ---------------------------------------------------------------------------


def test_analyze_sorted_desc_policy_prob() -> None:
    """analyze() ScoredMoves must be sorted descending by policy_prob."""
    from dongfeng.protocol.engine import SearchLimits  # noqa: PLC0415

    engine = make_engine()
    engine.new_game()
    engine.set_position(STARTING_FEN, [])

    analysis = engine.analyze(SearchLimits())
    assert analysis.moves, "analyze() returned no moves from start"

    probs = [sm.policy_prob for sm in analysis.moves]
    for i in range(len(probs) - 1):
        assert probs[i] is not None and probs[i + 1] is not None
        assert probs[i] >= probs[i + 1], (  # type: ignore[operator]
            f"policy_prob not sorted desc at index {i}: {probs[i]} < {probs[i + 1]}"
        )


def test_analyze_win_prob_range() -> None:
    """Every ScoredMove must have win_prob in [0, 1]."""
    from dongfeng.protocol.engine import SearchLimits  # noqa: PLC0415

    engine = make_engine()
    engine.new_game()
    engine.set_position(STARTING_FEN, [])

    analysis = engine.analyze(SearchLimits())
    assert analysis.moves, "analyze() returned no moves from start"

    for sm in analysis.moves:
        assert sm.win_prob is not None, "win_prob is None"
        assert 0.0 <= sm.win_prob <= 1.0, f"win_prob out of range: {sm.win_prob}"
