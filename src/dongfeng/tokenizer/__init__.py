"""Dong Feng tokenizers: turn Xiangqi symbols into model token ids.

Public surface:
    Tokenizer         -- the tokenizer Protocol (contract)
    MoveTokenizer     -- ICCS move <-> tokens (stub, M1)
    BoardTokenizer    -- Xiangqi FEN <-> tokens (stub, M1)

The concrete tokenizers are stubs landing in milestone M1; the Protocol is stable
now so models and the data pipeline can be typed against it.
"""

from __future__ import annotations

from .base import Tokenizer
from .board_tokenizer import BoardTokenizer
from .move_tokenizer import MoveTokenizer

__all__ = [
    "BoardTokenizer",
    "MoveTokenizer",
    "Tokenizer",
]
