"""Tests for the DhtmlXQ (DPXQ web) parser (M1)."""

from __future__ import annotations

from dongfeng.core import STARTING_FEN, Move, new_board
from dongfeng.data import parse_dhtmlxq

# Standard start position binit (32 pieces, Red order then Black) + three opening
# plies: 炮二平五 (h2e2), 马8进7 (h9g7), 马二进三 (h0g2) in DhtmlXQ move codes.
_BINIT = "09192939495969798917770626466686" + "00102030405060708012720323436383"
_MOVELIST = "7747" + "7062" + "7967"
_DHTMLXQ = (
    "[DhtmlXQ]"
    "[DhtmlXQ_result]和局[/DhtmlXQ_result]"
    "[DhtmlXQ_red]Red Player[/DhtmlXQ_red]"
    f"[DhtmlXQ_binit]{_BINIT}[/DhtmlXQ_binit]"
    f"[DhtmlXQ_movelist]{_MOVELIST}[/DhtmlXQ_movelist]"
)


def test_binit_decodes_to_standard_start() -> None:
    game = next(parse_dhtmlxq(_DHTMLXQ))
    assert game.start_fen.split()[0] == STARTING_FEN.split()[0]
    assert game.start_fen.split()[1] == "r"  # Red moves first


def test_movelist_decodes_to_iccs() -> None:
    game = next(parse_dhtmlxq(_DHTMLXQ))
    assert [m.iccs for m in game.moves] == ["h2e2", "h9g7", "h0g2"]


def test_decoded_game_replays_legally() -> None:
    game = next(parse_dhtmlxq(_DHTMLXQ))
    board = new_board(game.start_fen)
    for move in game.moves:
        assert board.is_legal(move), move.iccs
        board.push(move)


def test_metadata_and_result() -> None:
    from dongfeng.core import GameResult

    game = next(parse_dhtmlxq(_DHTMLXQ))
    assert game.result is GameResult.DRAW  # 和局
    assert game.metadata.get("red") == "Red Player"


def test_first_move_is_known_cannon_opening() -> None:
    game = next(parse_dhtmlxq(_DHTMLXQ))
    assert game.moves[0] == Move.from_iccs("h2e2")
