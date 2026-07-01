"""ICCS move tokenizer (stub).

Roadmap milestone: **M1**.

:class:`MoveTokenizer` will encode Xiangqi moves as tokens over a fixed
from-square/to-square vocabulary derived from the 9x10 ICCS board.

Planned vocabulary design (see :mod:`dongfeng.core.types` for the ICCS spec):

* The board has ``9 * 10 = 90`` addressable points; files ``a``-``i`` map to
  column indices ``0``-``8`` and ranks ``0``-``9`` map to row indices ``0``-``9``.
* An ICCS move ``"h2e2"`` is a ``(from_square, to_square)`` pair. Two natural
  encodings, to be chosen and pinned in M1:

  1. **Square-pair**: emit two tokens, one per square, over a 90-symbol square
     vocabulary (plus special tokens). Compact and factorizes cleanly for a
     policy head over ``from`` then ``to``.
  2. **Flat move index**: a single token per legal move over the enumerated
     move space. Larger vocabulary but one token per move — convenient for a
     softmax policy head that predicts a whole move at once.

* Xiangqi has **no promotion**, so a move is always exactly 4 ICCS characters and
  the vocabulary needs no promotion symbols (see :class:`dongfeng.core.types.Move`).

Until M1, all methods raise :class:`NotImplementedError`.
"""

from __future__ import annotations


class MoveTokenizer:
    """Tokenizer over the ICCS move space of the 9x10 Xiangqi board (M1 stub).

    Implements the :class:`~dongfeng.tokenizer.base.Tokenizer` protocol. The
    concrete vocabulary and encoding scheme are finalized in milestone M1.
    """

    def encode(self, x: str) -> list[int]:
        """Encode an ICCS move string (e.g. ``"h2e2"``) into token ids.

        Raises:
            NotImplementedError: always — planned for milestone M1.
        """
        raise NotImplementedError("MoveTokenizer.encode is planned: M1")

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into an ICCS move string.

        Raises:
            NotImplementedError: always — planned for milestone M1.
        """
        raise NotImplementedError("MoveTokenizer.decode is planned: M1")

    @property
    def vocab_size(self) -> int:
        """Number of distinct move/square token ids.

        Raises:
            NotImplementedError: always — planned for milestone M1.
        """
        raise NotImplementedError("MoveTokenizer.vocab_size is planned: M1")
