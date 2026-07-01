"""Dong Feng core: rules-agnostic types plus the rules-library-backed board.

Public surface:
    Color, GameResult, Move          -- pure value types (no rules-lib dependency)
    Board                            -- the board Protocol (contract)
    LibBoard, new_board              -- concrete implementation + factory
    STARTING_FEN, validate_fen       -- FEN helpers
"""

from __future__ import annotations

from .board import Board, LibBoard, new_board
from .fen import STARTING_FEN, validate_fen
from .types import Color, GameResult, Move

__all__ = [
    "STARTING_FEN",
    "Board",
    "Color",
    "GameResult",
    "LibBoard",
    "Move",
    "new_board",
    "validate_fen",
]
