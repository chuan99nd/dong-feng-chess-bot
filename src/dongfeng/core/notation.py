"""Move-notation helpers.

Dong Feng's canonical move notation is ICCS coordinate notation: files ``a``-``i``,
ranks ``0``-``9``, 4-char moves like ``"h2e2"`` with no promotion suffix.

WXF (relative / columnar) notation — e.g. ``C2=5``, ``H2+3``, ``+R+1`` — is
side-relative and requires board/turn context to convert. It is intentionally left
as a separate, not-yet-implemented converter layer.
"""

from __future__ import annotations

from .types import Move

_FILES = "abcdefghi"
_RANKS = "0123456789"


def is_iccs_move(s: str) -> bool:
    """Return ``True`` iff ``s`` is a well-formed 4-char ICCS move string."""
    return len(s) == 4 and s[0] in _FILES and s[1] in _RANKS and s[2] in _FILES and s[3] in _RANKS


def parse_iccs(s: str) -> Move:
    """Parse a 4-char ICCS move string into a :class:`Move`.

    Thin wrapper over :meth:`Move.from_iccs` for symmetry with :func:`format_iccs`.
    """
    return Move.from_iccs(s)


def format_iccs(move: Move) -> str:
    """Format a :class:`Move` as a 4-char ICCS string."""
    return move.iccs


def wxf_to_iccs(wxf: str, fen: str) -> Move:
    """Convert a WXF relative-notation move to an ICCS :class:`Move`.

    WXF notation is side-relative (file numbering is mirrored between Red and
    Black), so board context is required — hence the ``fen`` parameter.

    Planned for milestone **M1**; not yet implemented.
    """
    raise NotImplementedError("planned: M1")


def iccs_to_wxf(move: Move, fen: str) -> str:
    """Convert an ICCS :class:`Move` to WXF relative notation.

    Requires board context (``fen``) to resolve the side-relative file numbering
    and any front/rear (``+``/``-``) disambiguation.

    Planned for milestone **M1**; not yet implemented.
    """
    raise NotImplementedError("planned: M1")
