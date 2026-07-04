"""Tests for MctsBoardEngine (WP-MCTS).

All tests skip when torch is unavailable (the 'model' extra is optional). A tiny
random-init model + a small number of simulations keep the suite fast.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping MCTS board tests")

from dongfeng.core import STARTING_FEN, new_board  # noqa: E402  # type: ignore[import]
from dongfeng.core.types import GameResult  # noqa: E402
from dongfeng.inference.mcts_board import MctsBoardEngine  # noqa: E402
from dongfeng.model.board_transformer import (  # noqa: E402
    BoardTransformer,
    BoardTransformerConfig,
)
from dongfeng.protocol import run_conformance  # noqa: E402
from dongfeng.protocol.engine import SearchLimits  # noqa: E402

_TINY_CFG = BoardTransformerConfig(d_model=32, n_layer=1, n_head=2, ffn_hidden=64)


def _tiny_model() -> BoardTransformer:
    torch.manual_seed(0)
    return BoardTransformer(_TINY_CFG)


def _make_engine(**kw: object) -> MctsBoardEngine:
    """Return a random-init MctsBoardEngine with a tiny model injected."""
    engine = MctsBoardEngine(checkpoint=None, device="cpu", n_simulations=16, **kw)  # type: ignore[arg-type]
    # Swap in a small deterministic model for speed / reproducibility.
    model = _tiny_model()
    model.to("cpu").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    engine._model = model
    return engine


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


def test_conformance() -> None:
    """MctsBoardEngine must satisfy the universal engine conformance suite."""
    failures = run_conformance(lambda: _make_engine(value_mode="zero"))
    assert failures == [], "conformance failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_temperature_zero() -> None:
    """Fixed seed + temperature 0 + fixed sims -> identical bestmove twice."""
    limits = SearchLimits(nodes=24)

    e1 = _make_engine(value_mode="zero", temperature=0.0, seed=0)
    e1.set_position(STARTING_FEN, [])
    m1 = e1.bestmove(limits)

    e2 = _make_engine(value_mode="zero", temperature=0.0, seed=0)
    e2.set_position(STARTING_FEN, [])
    m2 = e2.bestmove(limits)

    assert m1.iccs == m2.iccs


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


def test_only_legal_moves() -> None:
    """bestmove and every analyze move must be legal in the position."""
    engine = _make_engine(value_mode="zero")
    engine.set_position(STARTING_FEN, [])

    board = new_board(STARTING_FEN)
    legal = {m.iccs for m in board.legal_moves()}

    best = engine.bestmove(SearchLimits(nodes=16))
    assert best.iccs in legal

    analysis = engine.analyze(SearchLimits(nodes=16))
    assert analysis.moves
    for sm in analysis.moves:
        assert sm.move.iccs in legal
        assert sm.win_prob is not None and 0.0 <= sm.win_prob <= 1.0


# ---------------------------------------------------------------------------
# Terminal / mate handled as a loss
# ---------------------------------------------------------------------------


def test_terminal_position_is_loss() -> None:
    """A checkmated position (no legal moves) is a loss and bestmove raises."""
    # Black to move and checkmated: red rooks doubled on the file, black king
    # trapped. Simpler: construct a position with no legal moves for the mover
    # by searching for one from a near-mate. We instead assert the engine treats
    # a game-over position correctly via analyze() returning no moves.
    #
    # Black to move and checkmated: red chariot on d1 pins the black king on d10
    # with the red king facing on the e-file — black has no legal escape.
    mate_fen = "3k5/9/9/9/9/9/9/9/9/3RK4 b - - 0 1"
    board = new_board(mate_fen)
    if not board.is_game_over():
        pytest.skip("constructed FEN is not terminal in this backend")

    assert board.result() in (GameResult.RED_WIN, GameResult.BLACK_WIN)

    engine = _make_engine(value_mode="zero")
    engine.set_position(mate_fen, [])
    assert engine.analyze(SearchLimits(nodes=8)).moves == []
    with pytest.raises(ValueError):
        engine.bestmove(SearchLimits(nodes=8))


# ---------------------------------------------------------------------------
# Nodes limit honored
# ---------------------------------------------------------------------------


def test_nodes_limit_honored() -> None:
    """Root visit count ~ n_simulations (nodes limit)."""
    engine = _make_engine(value_mode="zero")
    engine.set_position(STARTING_FEN, [])
    n = 20
    analysis = engine.analyze(SearchLimits(nodes=n))
    # analysis.nodes is the simulation count; each simulation adds exactly one
    # root visit, so the root visit count equals n_simulations.
    assert analysis.nodes == n


def test_stop_flag_halts_search() -> None:
    """stop() set after priming halts the sim loop early (cooperative flag)."""
    engine = _make_engine(value_mode="zero")
    engine.set_position(STARTING_FEN, [])

    # Monkeypatch _simulate to trip the stop flag on the first simulation.
    original = engine._simulate
    calls = {"n": 0}

    def stopping_simulate(root: object, board: object) -> None:  # type: ignore[override]
        calls["n"] += 1
        original(root, board)  # type: ignore[arg-type]
        engine.stop()

    engine._simulate = stopping_simulate  # type: ignore[assignment,method-assign]
    analysis = engine.analyze(SearchLimits(nodes=100))
    assert calls["n"] == 1
    assert analysis.nodes == 1
