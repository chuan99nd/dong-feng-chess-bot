"""Dong Feng models: policy (and optional value) networks.

Public surface:
    PolicyModel          -- the model Protocol (contract; optional value head)
    TransformerPolicy    -- decoder-only transformer (stub, M2)

The concrete transformer is a stub landing in milestone M2; the Protocol is stable
now so inference and training can be typed against it.
"""

from __future__ import annotations

from .base import PolicyModel
from .transformer import TransformerPolicy

__all__ = [
    "PolicyModel",
    "TransformerPolicy",
]
