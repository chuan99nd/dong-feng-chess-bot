"""Tests for ICCS <-> traditional (Chinese/WXF) notation conversion (M1)."""

from __future__ import annotations

from dongfeng.core import STARTING_FEN, Move, new_board
from dongfeng.core.notation import iccs_to_wxf, wxf_to_iccs


def test_known_conversion_from_start() -> None:
    # Cannon to the centre file: the canonical opening move.
    assert iccs_to_wxf(Move.from_iccs("h2e2"), STARTING_FEN) == "炮二平五"
    assert wxf_to_iccs("炮二平五", STARTING_FEN) == Move.from_iccs("h2e2")


def test_roundtrip_all_legal_start_moves() -> None:
    board = new_board(STARTING_FEN)
    for move in board.legal_moves():
        wxf = iccs_to_wxf(move, STARTING_FEN)
        assert wxf_to_iccs(wxf, STARTING_FEN) == move
