"""Dong Feng models: policy (and optional value) networks.

Public surface:
    PolicyModel                     -- the model Protocol (contract; optional value head)
    TransformerPolicy, TransformerConfig -- decoder-only transformer (M2)
    BoardTransformer, BoardTransformerConfig -- board-state encoder transformer (M3.5)

Importing :class:`TransformerPolicy` and :class:`BoardTransformer` requires the
optional ``model`` extra (torch); :class:`PolicyModel` (the Protocol) is
dependency-free.
"""

from __future__ import annotations

from .base import PolicyModel
from .board_transformer import BoardTransformer, BoardTransformerConfig
from .transformer import TransformerConfig, TransformerPolicy

__all__ = [
    "PolicyModel",
    "TransformerConfig",
    "TransformerPolicy",
    "BoardTransformer",
    "BoardTransformerConfig",
]
