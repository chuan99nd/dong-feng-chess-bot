"""Move-notation helpers.

Dong Feng's canonical move notation is ICCS coordinate notation: files ``a``-``i``,
ranks ``0``-``9``, 4-char moves like ``"h2e2"`` with no promotion suffix.

Traditional relative notation — the WXF family, rendered by the backend as Chinese
text such as ``炮二平五`` (cannon on file 2 moves to file 5) — is side-relative
(file numbering is mirrored between Red and Black) and requires board/turn context
to convert. The converters below wrap the ``cchess`` backend, which round-trips
ICCS <-> this traditional notation; the ``fen`` argument supplies the required
context. (A Latin WXF form like ``C2=5`` is a straightforward re-rendering of the
same relative move and can be layered on later.)
"""

from __future__ import annotations

from typing import Any

from .types import Move

_FILES = "abcdefghi"
_RANKS = "0123456789"

_INSTALL_HINT = (
    "The 'cchess' rules library is required for notation conversion. "
    "Install it with: uv pip install 'cchess>=1.25,<2'"
)


def _board_from_fen(fen: str) -> tuple[Any, Any]:
    """Return a fresh ``(cchess.ChessBoard, cchess)`` positioned at ``fen``.

    The board is a throwaway: the ``move_*`` calls below mutate it, so callers use
    one board per conversion. Raises :class:`ImportError` (with an install hint) if
    the ``cchess`` backend is missing.
    """
    try:
        import cchess  # noqa: PLC0415  (deferred import by design)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_HINT) from exc
    board = cchess.ChessBoard()
    board.from_fen(fen)
    return board, cchess


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
    """Convert a traditional relative-notation move (e.g. ``炮二平五``) to ICCS.

    The notation is side-relative, so the position (``fen``) is required to resolve
    which piece moves and where.

    Args:
        wxf: The move in traditional Chinese relative notation.
        fen: The position the move is played from (Xiangqi FEN).

    Returns:
        The move as an ICCS :class:`Move`.

    Raises:
        ImportError: if the ``cchess`` backend is unavailable.
        ValueError: if the notation is invalid or ambiguous in this position.
    """
    board, _ = _board_from_fen(fen)
    try:
        mv = board.move_text(wxf)
    except Exception as exc:  # cchess raises its own exception types
        raise ValueError(f"cannot parse {wxf!r} in this position: {exc}") from exc
    if mv is None:
        raise ValueError(f"illegal or ambiguous move {wxf!r} in this position")
    return Move.from_iccs(mv.to_iccs())


def iccs_to_wxf(move: Move, fen: str) -> str:
    """Convert an ICCS :class:`Move` to traditional relative notation.

    Requires board context (``fen``) to resolve the side-relative file numbering and
    any front/rear disambiguation. Returns the Chinese-character rendering produced
    by the backend (e.g. ``炮二平五``).

    Raises:
        ImportError: if the ``cchess`` backend is unavailable.
        ValueError: if ``move`` is illegal in this position.
    """
    board, _ = _board_from_fen(fen)
    try:
        mv = board.move_iccs(move.iccs)
    except Exception as exc:
        raise ValueError(f"cannot play {move.iccs} in this position: {exc}") from exc
    if mv is None:
        raise ValueError(f"illegal move {move.iccs} in this position")
    return mv.to_text()
