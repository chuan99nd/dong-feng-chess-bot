"""Dong Feng protocol: the universal engine (bot) contract and conformance harness.

Public surface:
    Engine                              -- the universal bot Protocol (contract)
    EngineInfo, SearchLimits            -- engine identity and search bounds
    ScoredMove, Analysis               -- analysis results
    run_conformance                     -- validate any engine factory against the contract
"""

from __future__ import annotations

from .conformance import run_conformance
from .engine import (
    Analysis,
    Engine,
    EngineInfo,
    ScoredMove,
    SearchLimits,
)

__all__ = [
    "Analysis",
    "Engine",
    "EngineInfo",
    "ScoredMove",
    "SearchLimits",
    "run_conformance",
]
