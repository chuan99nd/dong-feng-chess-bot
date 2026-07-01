"""Tests for the per-point board (FEN) tokenizer (M1)."""

from __future__ import annotations

import pytest

from dongfeng.core import STARTING_FEN
from dongfeng.tokenizer import BoardTokenizer


def test_vocab_size() -> None:
    # specials(4) + empty(1) + pieces(14) + sides(2)
    assert BoardTokenizer().vocab_size == 21


def test_encode_length_is_91() -> None:
    tok = BoardTokenizer()
    ids = tok.encode(STARTING_FEN)
    assert len(ids) == 90 + 1  # 90 board points + side token


def test_roundtrip_start_position() -> None:
    tok = BoardTokenizer()
    decoded = tok.decode(tok.encode(STARTING_FEN))
    placement, side = decoded.split()
    assert placement == STARTING_FEN.split()[0]
    assert side == "r"  # STARTING_FEN is red to move


@pytest.mark.parametrize(
    "fen",
    [
        STARTING_FEN,
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b - - 0 1",
        "3k5/9/9/9/9/9/9/9/9/4K4 r",  # sparse endgame position
    ],
)
def test_roundtrip_preserves_placement_and_side(fen: str) -> None:
    tok = BoardTokenizer()
    decoded = tok.decode(tok.encode(fen))
    placement, side = decoded.split()
    assert placement == fen.split()[0]
    # 'w' normalizes to 'r'; otherwise the side token is preserved.
    expected_side = "b" if fen.split()[1] == "b" else "r"
    assert side == expected_side


def test_bad_length_rejected() -> None:
    tok = BoardTokenizer()
    with pytest.raises(ValueError):
        tok.decode([tok.PAD_ID, tok.PAD_ID])


def test_invalid_fen_rejected() -> None:
    tok = BoardTokenizer()
    with pytest.raises(ValueError):
        tok.encode("not a fen")
