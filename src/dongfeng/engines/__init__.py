"""Dong Feng engines: concrete :class:`~dongfeng.protocol.engine.Engine` bots.

Public surface:
    RandomEngine     -- a seedable random-legal-mover baseline (zero dependencies
                        beyond the core rules backend).
    PikafishEngine   -- wraps an external Pikafish (or any UCI/UCCI) binary via
                        subprocess.

Both conform to the universal :class:`~dongfeng.protocol.engine.Engine` contract
and are drop-in replacements for one another.
"""

from __future__ import annotations

from .pikafish_engine import PikafishEngine
from .random_engine import RandomEngine

__all__ = [
    "PikafishEngine",
    "RandomEngine",
]
