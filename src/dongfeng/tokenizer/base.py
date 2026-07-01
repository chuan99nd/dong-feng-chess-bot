"""Tokenizer contract for Dong Feng.

A :class:`Tokenizer` turns a domain object (a move string, a FEN board, a game
transcript) into a flat sequence of integer token ids and back. It is the bridge
between the symbolic Xiangqi world (FEN / ICCS, see :mod:`dongfeng.core`) and the
tensor world consumed by the models in :mod:`dongfeng.model`.

Roadmap: the concrete tokenizers (:class:`~dongfeng.tokenizer.move_tokenizer.MoveTokenizer`,
:class:`~dongfeng.tokenizer.board_tokenizer.BoardTokenizer`) land in milestone
**M1** alongside the data pipeline. This module only defines the interface so the
rest of the architecture can be typed against it today.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Reversible mapping between domain strings and integer token ids.

    Implementations must be lossless for in-vocabulary input: ``decode(encode(x))``
    should reconstruct ``x`` (up to documented normalization). Every id returned by
    :meth:`encode` must be in ``range(vocab_size)``.
    """

    def encode(self, x: str) -> list[int]:
        """Encode a domain string into a list of token ids.

        Args:
            x: The input string (e.g. an ICCS move ``"h2e2"`` or a FEN board).

        Returns:
            A list of integer token ids, each in ``range(vocab_size)``.
        """
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a domain string.

        Args:
            ids: Token ids previously produced by :meth:`encode`.

        Returns:
            The reconstructed domain string.
        """
        ...

    @property
    def vocab_size(self) -> int:
        """The number of distinct token ids in this tokenizer's vocabulary."""
        ...
