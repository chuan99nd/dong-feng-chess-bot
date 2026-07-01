"""Perft: count leaf nodes of the move tree to a fixed depth.

Perft ("performance test") is the standard correctness check for a move
generator: it counts the number of leaf positions reachable in exactly ``depth``
plies from a position, exercising :meth:`Board.legal_moves`, :meth:`Board.push`,
and :meth:`Board.pop`. A mismatch against a trusted count reveals move-generation
or make/unmake bugs.

This is a real, working implementation built on the core :class:`dongfeng.core.Board`
contract — it needs no model or engine, only the rules backend.

Trusted starting-position perft values (standard Xiangqi, verified against the
``cchess`` backend):

===== =============
depth perft
===== =============
1     44
2     1,920
3     79,666
===== =============
"""

from __future__ import annotations

from ..core.board import Board


def perft(board: Board, depth: int) -> int:
    """Count leaf nodes reachable in exactly ``depth`` plies from ``board``.

    The board is restored to its original state on return (every :meth:`push` is
    matched by a :meth:`pop`).

    Args:
        board: The starting position. Mutated during traversal but restored before
            returning.
        depth: Non-negative ply depth. ``0`` counts the position itself (returns 1);
            ``1`` returns the number of legal moves.

    Returns:
        The number of leaf nodes at exactly ``depth`` plies.

    Raises:
        ValueError: if ``depth`` is negative.
    """
    if depth < 0:
        raise ValueError(f"perft depth must be non-negative, got {depth}")
    if depth == 0:
        return 1

    moves = board.legal_moves()
    if depth == 1:
        return len(moves)

    total = 0
    for move in moves:
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total
