"""Move-generation tests against the real rules backend (``dongfeng.core``).

These exercise the concrete :class:`~dongfeng.core.Board` produced by
:func:`dongfeng.core.new_board` plus the real :func:`dongfeng.eval.perft`. If the
``cchess`` rules library is not installed, the whole module is skipped (the board
factory raises :class:`ImportError` with an install hint).
"""

from __future__ import annotations

import pytest

from dongfeng.core import new_board

# The standard starting position has 44 legal moves; the perft series
# (44, 1920, 79666) is the well-known trusted Xiangqi reference and is verified
# against the cchess backend. See dongfeng/eval/perft.py.
_STARTING_LEGAL_MOVES = 44
_PERFT_START = {1: 44, 2: 1920}


@pytest.fixture
def start_board():
    """A board at the standard starting position, or skip if the backend is absent."""
    try:
        return new_board()
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"rules backend unavailable: {exc}")


def test_starting_position_has_legal_moves(start_board):
    """From the start there are moves, and every one is reported legal."""
    moves = start_board.legal_moves()
    assert len(moves) > 0
    assert len(moves) == _STARTING_LEGAL_MOVES
    for move in moves:
        assert start_board.is_legal(move)


def test_all_legal_moves_are_unique(start_board):
    """legal_moves() should not report duplicates."""
    moves = start_board.legal_moves()
    iccs = [m.iccs for m in moves]
    assert len(iccs) == len(set(iccs))


def test_perft_depth_one_equals_legal_move_count(start_board):
    """perft(start, 1) must equal the number of legal moves."""
    from dongfeng.eval import perft

    assert perft(start_board, 1) == len(start_board.legal_moves())


def test_perft_trusted_counts(start_board):
    """perft matches the trusted starting-position reference counts."""
    from dongfeng.eval import perft

    for depth, expected in _PERFT_START.items():
        assert perft(start_board, depth) == expected


def test_perft_depth_zero_is_one(start_board):
    """perft at depth 0 counts the position itself."""
    from dongfeng.eval import perft

    assert perft(start_board, 0) == 1


def test_perft_restores_board(start_board):
    """perft must leave the board unchanged (every push matched by a pop)."""
    from dongfeng.eval import perft

    before = start_board.fen()
    perft(start_board, 2)
    assert start_board.fen() == before


def test_push_pop_round_trip_preserves_fen(start_board):
    """Pushing then popping a move returns to the exact same FEN."""
    before = start_board.fen()
    move = start_board.legal_moves()[0]
    start_board.push(move)
    assert start_board.fen() != before  # the move actually changed the position
    popped = start_board.pop()
    assert popped == move
    assert start_board.fen() == before


def test_push_pop_round_trip_over_many_moves(start_board):
    """A push of every legal move, each immediately popped, always restores the FEN."""
    before = start_board.fen()
    for move in start_board.legal_moves():
        start_board.push(move)
        start_board.pop()
        assert start_board.fen() == before
