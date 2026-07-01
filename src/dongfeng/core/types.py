"""Core, library-agnostic value types for Dong Feng.

Everything here is expressed in pure Python (strings / ints / enums) with no
dependency on any Xiangqi rules library. This keeps the public vocabulary stable
even if the underlying engine or rules backend changes.

Coordinates use ICCS notation: files ``a``-``i`` (left to right), ranks ``0``-``9``
(bottom to top), with ``a0`` being Red's bottom-left corner. A move is a 4-char
string ``<from><to>`` such as ``"h2e2"``. Xiangqi has no promotion, so a move is
always exactly 4 characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ICCS axis alphabets.
_FILES = "abcdefghi"  # 9 files, a-i
_RANKS = "0123456789"  # 10 ranks, 0-9


class Color(Enum):
    """Side to move / piece owner."""

    RED = "red"
    BLACK = "black"

    def other(self) -> Color:
        """Return the opposing color."""
        return Color.BLACK if self is Color.RED else Color.RED


class GameResult(Enum):
    """Terminal (or non-terminal) outcome of a game.

    Note: in Xiangqi, a side with no legal moves (stalemate) *loses*; it is not a
    draw as in Western chess. Backends must map "no legal moves" to a loss for the
    side to move, not to ``DRAW``.
    """

    RED_WIN = "red_win"
    BLACK_WIN = "black_win"
    DRAW = "draw"
    ONGOING = "ongoing"


def _is_iccs_square(sq: str) -> bool:
    """True iff ``sq`` is a valid 2-char ICCS square like ``"h2"``."""
    return len(sq) == 2 and sq[0] in _FILES and sq[1] in _RANKS


@dataclass(frozen=True, slots=True)
class Move:
    """A single Xiangqi move in ICCS coordinates.

    Attributes:
        from_sq: Origin square, e.g. ``"h2"``.
        to_sq: Destination square, e.g. ``"e2"``.
    """

    from_sq: str
    to_sq: str

    def __post_init__(self) -> None:
        if not _is_iccs_square(self.from_sq):
            raise ValueError(f"invalid ICCS from-square: {self.from_sq!r}")
        if not _is_iccs_square(self.to_sq):
            raise ValueError(f"invalid ICCS to-square: {self.to_sq!r}")

    @classmethod
    def from_iccs(cls, s: str) -> Move:
        """Parse a 4-char ICCS move string (e.g. ``"h2e2"``) into a :class:`Move`.

        Raises:
            ValueError: if ``s`` is not exactly 4 ICCS-valid characters. A 5th
                character (a promotion suffix) is rejected, since Xiangqi has no
                promotion.
        """
        if len(s) != 4:
            raise ValueError(f"ICCS move must be exactly 4 chars, got {s!r}")
        return cls(from_sq=s[:2], to_sq=s[2:])

    @property
    def iccs(self) -> str:
        """The move as a 4-char ICCS string, e.g. ``"h2e2"``."""
        return f"{self.from_sq}{self.to_sq}"

    def __str__(self) -> str:
        return self.iccs
