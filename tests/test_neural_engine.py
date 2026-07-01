"""Tests for the transformer-backed neural engine (M2/M3). Skipped without torch."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dongfeng.core import STARTING_FEN, new_board  # noqa: E402
from dongfeng.inference.transformer_engine import TransformerEngine  # noqa: E402
from dongfeng.protocol.conformance import run_conformance  # noqa: E402
from dongfeng.protocol.engine import SearchLimits  # noqa: E402


def test_neural_engine_passes_conformance() -> None:
    # Random-init model: legal-move masking guarantees a legal bestmove.
    assert run_conformance(lambda: TransformerEngine()) == []


def test_bestmove_is_legal_from_start() -> None:
    engine = TransformerEngine()
    engine.new_game()
    engine.set_position(STARTING_FEN, [])
    move = engine.bestmove(SearchLimits(movetime_ms=10))
    assert new_board(STARTING_FEN).is_legal(move)


def test_analyze_returns_sorted_legal_moves() -> None:
    engine = TransformerEngine()
    engine.set_position(STARTING_FEN, [])
    analysis = engine.analyze(SearchLimits())
    board = new_board(STARTING_FEN)
    probs = [sm.policy_prob for sm in analysis.moves]
    assert len(analysis.moves) == len(board.legal_moves())
    assert all(board.is_legal(sm.move) for sm in analysis.moves)
    assert probs == sorted(probs, reverse=True)  # descending by policy prior


def test_sampling_options_do_not_crash() -> None:
    engine = TransformerEngine()
    engine.set_option("Temperature", "1.0")
    engine.set_option("TopK", "5")
    engine.set_option("Seed", "3")
    engine.set_position(STARTING_FEN, [])
    move = engine.bestmove(SearchLimits())
    assert new_board(STARTING_FEN).is_legal(move)
