"""Dong Feng inference: run trained models as universal engines.

Public surface:
    TransformerEngine    -- neural Engine implementation (stub, M3)

:class:`TransformerEngine` implements the universal
:class:`dongfeng.protocol.engine.Engine` contract, so the neural model is a
drop-in for any match runner, arena, CLI, or MCP server. It is a stub landing in
milestone M3.
"""

from __future__ import annotations

from .transformer_engine import TransformerEngine

__all__ = [
    "TransformerEngine",
]
