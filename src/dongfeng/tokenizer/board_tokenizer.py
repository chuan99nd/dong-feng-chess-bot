"""Xiangqi-FEN board tokenizer (stub).

Roadmap milestone: **M1**.

:class:`BoardTokenizer` will encode a Xiangqi position (a FEN string, see
:mod:`dongfeng.core.fen`) into a fixed-length token sequence suitable as model
input, and decode such a sequence back into a FEN.

Planned vocabulary design (canonical "modern engine" FEN dialect):

* Piece letters ``K A B N R C P`` (Red) and ``k a b n r c p`` (Black), plus an
  empty-point symbol — one token per board point over the ``9 * 10 = 90`` points.
  Expanding FEN run-length digits (``"9"``, ``"1c5c1"``, ...) into explicit
  per-point tokens gives a constant-length ``90``-point grid the model can attend
  over positionally.
* Side-to-move token: ``r``/``b`` (``w`` is accepted as an alias for Red on input,
  matching :func:`dongfeng.core.fen.side_to_move`).
* Optionally the move counters (halfmove clock, fullmove number); whether to
  include them is decided in M1.

The exact symbol set, ordering (rank 9 first, matching FEN), and whether counters
are tokenized are pinned in milestone M1. Until then, all methods raise
:class:`NotImplementedError`.
"""

from __future__ import annotations


class BoardTokenizer:
    """Tokenizer mapping Xiangqi FEN <-> token sequence (M1 stub).

    Implements the :class:`~dongfeng.tokenizer.base.Tokenizer` protocol. The
    concrete per-point grid encoding is finalized in milestone M1.
    """

    def encode(self, x: str) -> list[int]:
        """Encode a Xiangqi FEN string into a fixed-length token sequence.

        Raises:
            NotImplementedError: always — planned for milestone M1.
        """
        raise NotImplementedError("BoardTokenizer.encode is planned: M1")

    def decode(self, ids: list[int]) -> str:
        """Decode a token sequence back into a Xiangqi FEN string.

        Raises:
            NotImplementedError: always — planned for milestone M1.
        """
        raise NotImplementedError("BoardTokenizer.decode is planned: M1")

    @property
    def vocab_size(self) -> int:
        """Number of distinct board/point token ids.

        Raises:
            NotImplementedError: always — planned for milestone M1.
        """
        raise NotImplementedError("BoardTokenizer.vocab_size is planned: M1")
