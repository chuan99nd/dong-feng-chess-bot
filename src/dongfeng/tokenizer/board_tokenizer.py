"""Xiangqi-FEN board tokenizer — one token per board point.

Roadmap milestone: **M1** (landed).

:class:`BoardTokenizer` encodes a Xiangqi position (a FEN string) into a
fixed-length token sequence and decodes it back. It expands the run-length FEN
placement into an explicit ``9 * 10 = 90``-point grid (rank 9 first, matching FEN
order) and appends a side-to-move token, giving a constant length of **91** tokens
the model can attend over positionally. This is the input format for the
board-conditioned / action-value head used in distillation (see ROADMAP M4).

Vocabulary (canonical "modern engine" FEN dialect):

* ``.`` — an empty point.
* Piece letters ``K A B N R C P`` (Red) and ``k a b n r c p`` (Black):
  General, Advisor, Elephant (B), Horse (N), Chariot (R), Cannon (C), Pawn (P).
* Two side-to-move tokens (Red / Black); ``w`` is accepted as an alias for Red on
  input, matching :func:`dongfeng.core.fen.side_to_move`.
* Special tokens ``PAD``/``BOS``/``EOS``/``UNK``.
"""

from __future__ import annotations

from ..core.fen import side_to_move, validate_fen

_PIECES = "KABNRCPkabnrcp"  # 14 piece symbols
_EMPTY = "."
_SIDE_RED = "<red>"
_SIDE_BLACK = "<black>"
_BOARD_POINTS = 90  # 9 files x 10 ranks


class BoardTokenizer:
    """Reversible mapping between a Xiangqi FEN and a length-91 token sequence.

    Implements the :class:`~dongfeng.tokenizer.base.Tokenizer` protocol. The
    encoding is lossless for the placement + side-to-move fields; the move counters
    (halfmove/fullmove) are intentionally *not* tokenized (they do not affect the
    position), so :meth:`decode` returns the short two-field FEN form.
    """

    #: Tokenizer identity recorded in ``manifest.json`` and on checkpoints.
    id = "board-v1"

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    _NUM_SPECIAL = 4

    def __init__(self) -> None:
        # id layout: [specials..] [empty] [pieces..] [side_red] [side_black]
        symbols = [_EMPTY, *_PIECES, _SIDE_RED, _SIDE_BLACK]
        self._sym_to_id: dict[str, int] = {s: i + self._NUM_SPECIAL for i, s in enumerate(symbols)}
        self._id_to_sym: dict[int, str] = {v: k for k, v in self._sym_to_id.items()}

    # -- Tokenizer protocol -------------------------------------------------

    def encode(self, x: str) -> list[int]:
        """Encode a Xiangqi FEN into 90 point tokens + 1 side token (length 91).

        Raises:
            ValueError: if ``x`` is not a structurally valid Xiangqi FEN.
        """
        if not validate_fen(x):
            raise ValueError(f"invalid FEN: {x!r}")
        placement = x.split()[0]
        tokens: list[int] = []
        for rank in placement.split("/"):
            for ch in rank:
                if ch.isdigit():
                    tokens.extend([self._sym_to_id[_EMPTY]] * int(ch))
                else:
                    tokens.append(self._sym_to_id.get(ch, self.UNK_ID))
        side = side_to_move(x)  # 'r' or 'b'
        tokens.append(self._sym_to_id[_SIDE_RED if side == "r" else _SIDE_BLACK])
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode a length-91 token sequence back into a short-form Xiangqi FEN.

        Returns ``"<placement> <side>"`` (the two fields this tokenizer preserves).

        Raises:
            ValueError: if ``ids`` does not hold exactly 90 point tokens + 1 side.
        """
        if len(ids) != _BOARD_POINTS + 1:
            raise ValueError(f"expected {_BOARD_POINTS + 1} tokens, got {len(ids)}")
        point_syms = [self._id_to_sym.get(i, "?") for i in ids[:_BOARD_POINTS]]
        rows: list[str] = []
        for r in range(10):
            row_syms = point_syms[r * 9 : (r + 1) * 9]
            rows.append(self._compress_row(row_syms))
        side_sym = self._id_to_sym.get(ids[-1], _SIDE_RED)
        side = "b" if side_sym == _SIDE_BLACK else "r"
        return f"{'/'.join(rows)} {side}"

    @property
    def vocab_size(self) -> int:
        """Total number of ids: specials + empty + 14 pieces + 2 sides = 21."""
        return self._NUM_SPECIAL + len(self._sym_to_id)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _compress_row(syms: list[str]) -> str:
        """Run-length compress a row of 9 point symbols back into a FEN rank."""
        out: list[str] = []
        empties = 0
        for s in syms:
            if s == _EMPTY:
                empties += 1
                continue
            if empties:
                out.append(str(empties))
                empties = 0
            out.append(s)
        if empties:
            out.append(str(empties))
        return "".join(out)
