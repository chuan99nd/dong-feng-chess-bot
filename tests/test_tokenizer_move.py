"""Tests for the flat move-index tokenizer (M1)."""

from __future__ import annotations

from dongfeng.core import STARTING_FEN, Move, new_board
from dongfeng.tokenizer import MoveTokenizer


def test_vocab_is_stable_and_large() -> None:
    a, b = MoveTokenizer(), MoveTokenizer()
    assert a.vocab_size == b.vocab_size
    # Specials (4) + geometric move space (~2086 for a 9x10 xiangqi board).
    assert a.vocab_size > 2000
    assert a.vocab_size == b.vocab_size


def test_encode_decode_roundtrip() -> None:
    tok = MoveTokenizer()
    seq = "h2e2 h9g7 h0g2"
    ids = tok.encode(seq)
    assert len(ids) == 3
    assert tok.UNK_ID not in ids
    assert tok.decode(ids) == seq


def test_move_id_roundtrip() -> None:
    tok = MoveTokenizer()
    m = Move.from_iccs("h2e2")
    tid = tok.encode_move(m)
    assert tid != tok.UNK_ID
    assert tok.id_to_move(tid) == m


def test_all_legal_moves_are_in_vocab() -> None:
    tok = MoveTokenizer()
    # Start position and a position a few plies in must both be fully covered.
    board = new_board(STARTING_FEN)
    positions = [board.clone()]
    for iccs in ("h2e2", "h9g7", "b0c2"):
        board.push(Move.from_iccs(iccs))
        positions.append(board.clone())
    for pos in positions:
        for move in pos.legal_moves():
            assert tok.encode_move(move) != tok.UNK_ID, move.iccs


def test_encode_game_wraps_with_bos_eos() -> None:
    tok = MoveTokenizer()
    moves = [Move.from_iccs("h2e2"), Move.from_iccs("h9g7")]
    ids = tok.encode_game(moves)
    assert ids[0] == tok.BOS_ID
    assert ids[-1] == tok.EOS_ID
    assert len(ids) == len(moves) + 2


def test_unknown_move_maps_to_unk() -> None:
    tok = MoveTokenizer()
    # "a0a0" is a null move: not in the enumerated (from != to) vocabulary.
    assert tok.encode("a0a0") == [tok.UNK_ID]
