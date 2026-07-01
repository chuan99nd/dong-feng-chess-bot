"""Dong Feng models: policy (and optional value) networks.

Public surface:
    PolicyModel                     -- the model Protocol (contract; optional value head)
    TransformerPolicy, TransformerConfig -- decoder-only transformer (M2)

Importing :class:`TransformerPolicy` requires the optional ``model`` extra (torch);
:class:`PolicyModel` (the Protocol) is dependency-free.
"""

from __future__ import annotations

from .base import PolicyModel
from .transformer import TransformerConfig, TransformerPolicy

__all__ = [
    "PolicyModel",
    "TransformerConfig",
    "TransformerPolicy",
]
