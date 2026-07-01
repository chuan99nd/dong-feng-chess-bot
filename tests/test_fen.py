"""FEN round-trip and validation tests.

These cover :data:`dongfeng.core.STARTING_FEN`, :func:`dongfeng.core.validate_fen`,
and the board's :meth:`~dongfeng.core.Board.fen` / :meth:`~dongfeng.core.Board.set_fen`
round-trip. Board-backed tests skip if the ``cchess`` rules library is absent.

Note on the side-to-move token: ``STARTING_FEN`` uses ``r`` (Red), while the
``cchess`` backend emits the WXF-spec alias ``w`` for Red from ``board.fen()``.
Both denote Red and both are accepted by :func:`validate_fen`; the round-trip
comparison below normalizes that one token so it compares the position, not the
dialect.
"""

from __future__ import annotations

import pytest

from dongfeng.core import STARTING_FEN, new_board, validate_fen
from dongfeng.core.fen import side_to_move


def _normalize_side(fen: str) -> str:
    """Return ``fen`` with the side-to-move token normalized to ``r``/``b``.

    ``w`` (WXF-spec Red) is folded to ``r`` (engine-dialect Red) so two FENs that
    differ only in that alias compare equal.
    """
    fields = fen.split()
    fields[1] = side_to_move(fen)  # 'w' -> 'r', 'r' -> 'r', 'b' -> 'b'
    return " ".join(fields)


# --- validate_fen (no backend needed) --------------------------------------


def test_validate_accepts_starting_fen():
    assert validate_fen(STARTING_FEN) is True


def test_validate_accepts_w_side_token():
    """The WXF-spec 'w' (Red) alias is accepted, not just 'r'."""
    w_fen = STARTING_FEN.replace(" r ", " w ")
    assert w_fen != STARTING_FEN
    assert validate_fen(w_fen) is True


def test_validate_accepts_black_to_move():
    b_fen = STARTING_FEN.replace(" r ", " b ")
    assert validate_fen(b_fen) is True


def test_validate_accepts_short_form():
    """The two-field short form '<placement> <side>' is valid."""
    placement, side = STARTING_FEN.split()[0], "r"
    assert validate_fen(f"{placement} {side}") is True


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "not a fen",
        "garbage",
        # wrong number of ranks (9 instead of 10)
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9 r - - 0 1",
        # a rank whose widths don't sum to 9
        "rnbakabnr/8/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r - - 0 1",
        # illegal side-to-move token
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR x - - 0 1",
        # illegal piece letter (Z)
        "rnbakabnr/9/1c5c1/Z1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r - - 0 1",
        # non-'-' castling/en-passant placeholders
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r K - 0 1",
    ],
)
def test_validate_rejects_garbage(garbage):
    assert validate_fen(garbage) is False


# --- board fen()/set_fen() round-trip (needs the rules backend) -------------


@pytest.fixture
def start_board():
    try:
        return new_board()
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"rules backend unavailable: {exc}")


def test_new_board_fen_round_trips_to_starting_fen(start_board):
    """new_board().fen() equals STARTING_FEN up to the r/w side-token alias."""
    fen = start_board.fen()
    assert validate_fen(fen)
    assert _normalize_side(fen) == _normalize_side(STARTING_FEN)


def test_set_fen_starting_fen_works(start_board):
    """set_fen(STARTING_FEN) succeeds and restores the starting position."""
    # Perturb, then reset to the starting FEN.
    start_board.push(start_board.legal_moves()[0])
    start_board.set_fen(STARTING_FEN)
    assert _normalize_side(start_board.fen()) == _normalize_side(STARTING_FEN)


def test_set_fen_then_fen_is_stable(start_board):
    """Re-applying a board's own FEN via set_fen yields the same FEN."""
    fen = start_board.fen()
    start_board.set_fen(fen)
    assert start_board.fen() == fen
