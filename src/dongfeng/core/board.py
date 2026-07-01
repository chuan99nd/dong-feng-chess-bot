"""Board abstraction and the concrete rules-library-backed implementation.

The :class:`Board` Protocol is the stable contract the rest of Dong Feng codes
against. :class:`LibBoard` is the concrete implementation, wrapping the PyPI
``cchess`` library (walker8088).

Backend notes (verified against ``cchess`` 1.25.5):

* ``ChessBoard.create_moves()`` yields *pseudo-legal* moves as
  ``((from_col, from_row), (to_col, to_row))`` tuples. They are NOT filtered for
  self-check or flying-general — we filter each candidate through
  ``is_checked_move()`` to produce genuinely legal moves.
* ``ChessBoard.move()`` / ``move_iccs()`` mutate the board but do NOT flip the
  side to move; we call ``next_turn()`` explicitly after each move.
* There is no push/pop undo stack, so :meth:`LibBoard.pop` is implemented with an
  internal stack of ``(board_copy, move)`` snapshots.
* In Xiangqi, "no legal moves" is a LOSS for the side to move (stalemate == loss),
  reflected in :meth:`LibBoard.result`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .fen import STARTING_FEN, side_to_move
from .types import Color, GameResult, Move


@runtime_checkable
class Board(Protocol):
    """Mutable Xiangqi board contract.

    All positions are exchanged as Xiangqi FEN strings and all moves as ICCS
    :class:`Move` objects, keeping callers independent of any rules library.
    """

    def fen(self) -> str:
        """Return the current position as a FEN string."""
        ...

    def set_fen(self, fen: str) -> None:
        """Reset the board to the given FEN. Clears move history."""
        ...

    @property
    def turn(self) -> Color:
        """The side to move."""
        ...

    def legal_moves(self) -> list[Move]:
        """Return all fully-legal moves for the side to move."""
        ...

    def is_legal(self, m: Move) -> bool:
        """Return ``True`` iff ``m`` is legal in the current position."""
        ...

    def push(self, m: Move) -> None:
        """Apply a legal move, flipping the side to move.

        Raises:
            ValueError: if ``m`` is not legal in the current position.
        """
        ...

    def pop(self) -> Move:
        """Undo the last move and return it.

        Raises:
            IndexError: if there is no move to undo.
        """
        ...

    def is_check(self) -> bool:
        """Return ``True`` iff the side to move is in check."""
        ...

    def is_game_over(self) -> bool:
        """Return ``True`` iff the game has ended (mate or no legal moves)."""
        ...

    def result(self) -> GameResult:
        """Return the game result (``ONGOING`` if not over)."""
        ...

    def clone(self) -> Board:
        """Return an independent deep copy of this board (including history)."""
        ...

    def ascii(self) -> str:
        """Return a human-readable ASCII rendering of the board."""
        ...


_INSTALL_HINT = (
    "The 'cchess' rules library is required. Install it with:\n"
    "    uv pip install 'cchess>=1.25,<2'   (or) pip install cchess\n"
    "Note: the import name 'cchess' also collides with the GitHub package "
    "'python-chinese-chess' (windshadow233), which is a DIFFERENT, incompatible "
    "library. Dong Feng targets the PyPI 'cchess' by walker8088."
)


class LibBoard:
    """Concrete :class:`Board` backed by the PyPI ``cchess`` library.

    The ``cchess`` import is deferred to construction time so that importing this
    module never fails. If the library is missing, the constructor raises a clear
    :class:`ImportError` with an install hint.
    """

    # The backend (`cchess`) ships no type stubs, so its objects are typed `Any`
    # here; the public method signatures below remain fully, statically typed.
    _cchess: Any
    _board: Any
    _history: list[tuple[Any, Move]]

    def __init__(self, fen: str | None = None) -> None:
        try:
            import cchess  # noqa: PLC0415  (deferred import by design)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_INSTALL_HINT) from exc

        self._cchess = cchess
        self._board = cchess.ChessBoard()
        # History of (pre-move board copy, move) for undo.
        self._history = []
        self.set_fen(fen if fen is not None else STARTING_FEN)

    # -- position -----------------------------------------------------------

    def fen(self) -> str:
        return self._board.to_full_fen()

    def set_fen(self, fen: str) -> None:
        self._board.from_fen(fen)
        # Normalize the internal side-to-move from the FEN token (accepts w/r/b).
        self._history.clear()

    @property
    def turn(self) -> Color:
        return Color.RED if self._board.get_move_color() == self._cchess.RED else Color.BLACK

    # -- move generation ----------------------------------------------------

    def legal_moves(self) -> list[Move]:
        pos2iccs = self._cchess.pos2iccs
        board = self._board
        out: list[Move] = []
        # create_moves() is PSEUDO-legal; filter self-check / flying-general.
        for from_pos, to_pos in board.create_moves():
            if board.is_checked_move(from_pos, to_pos):
                continue
            out.append(Move.from_iccs(pos2iccs(from_pos, to_pos)))
        return out

    def is_legal(self, m: Move) -> bool:
        iccs = m.iccs
        board = self._board
        if not board.is_valid_iccs_move(iccs):
            return False
        from_pos, to_pos = self._cchess.iccs2pos(iccs)
        # A structurally valid move is legal only if it does not leave own king
        # in check (also covers the flying-general rule).
        return not board.is_checked_move(from_pos, to_pos)

    # -- make / undo --------------------------------------------------------

    def push(self, m: Move) -> None:
        if not self.is_legal(m):
            raise ValueError(f"illegal move for current position: {m.iccs}")
        snapshot = self._board.copy()
        self._board.move_iccs(m.iccs)
        # cchess.move()/move_iccs() do NOT flip the side to move; do it here.
        self._board.next_turn()
        self._history.append((snapshot, m))

    def pop(self) -> Move:
        if not self._history:
            raise IndexError("no move to undo")
        snapshot, move = self._history.pop()
        self._board = snapshot
        return move

    # -- status -------------------------------------------------------------

    def is_check(self) -> bool:
        # cchess's is_checking() answers "is the side to move giving check to the
        # OPPONENT?" To ask "is the side to move IN check?" we flip the mover on a
        # throwaway copy and ask whether that opponent attacks our king.
        probe = self._board.copy()
        probe.next_turn()
        return bool(probe.is_checking())

    def is_game_over(self) -> bool:
        # no_moves() is the authoritative, self-check-filtered terminal test: it is
        # True iff the side to move has zero legal moves (checkmate OR stalemate).
        # In Xiangqi both are terminal (and both are a loss for the side to move).
        return bool(self._board.no_moves())

    def result(self) -> GameResult:
        if not self.is_game_over():
            return GameResult.ONGOING
        # Checkmate or stalemate: the side to move has no way out and LOSES
        # (in Xiangqi stalemate is a loss, not a draw).
        loser = self.turn
        return GameResult.BLACK_WIN if loser is Color.RED else GameResult.RED_WIN

    # -- copy / render ------------------------------------------------------

    def clone(self) -> Board:
        twin = LibBoard.__new__(LibBoard)
        twin._cchess = self._cchess
        twin._board = self._board.copy()
        # Copy each historical snapshot so the clone's undo history is independent.
        twin._history = [(snap.copy(), mv) for snap, mv in self._history]
        return twin

    def ascii(self) -> str:
        view = self._board.text_view()
        if isinstance(view, (list, tuple)):
            return "\n".join(str(row) for row in view)
        return str(view)


def new_board(fen: str | None = None) -> Board:
    """Return a concrete :class:`Board` for the given FEN (default: starting position).

    Args:
        fen: A Xiangqi FEN. When ``None``, the standard starting position is used.

    Raises:
        ImportError: if the ``cchess`` rules library is not installed. The error
            message includes an install hint.
    """
    # Validate/normalize side token early for a clear error on malformed FEN,
    # while still letting the backend do the authoritative parse.
    if fen is not None:
        side_to_move(fen)  # raises ValueError on structurally invalid FEN
    return LibBoard(fen)
