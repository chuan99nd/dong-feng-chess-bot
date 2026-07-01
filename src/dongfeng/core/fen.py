"""Xiangqi FEN helpers.

Canonical dialect (modern engine convention, as used by the starting FEN in the
wild and by the ``cchess`` backend):

* Piece letters: ``K A B N R C P`` (Red, uppercase) and ``k a b n r c p`` (Black).
  ``K``=General, ``A``=Advisor, ``B``=Elephant, ``N``=Horse, ``R``=Chariot,
  ``C``=Cannon, ``P``=Pawn.
* Board: 9 files x 10 ranks (90 points). Rank groups are written top (Black back
  rank) first, bottom (Red back rank) last, separated by ``/``.
* Side to move: ``r`` (or ``w``) for Red, ``b`` for Black. Both ``r`` and ``w``
  are accepted for Red on input.
* Fields 3 and 4 (castling / en-passant) are unused in Xiangqi; when present they
  are ``-``.

This module does *lightweight* structural validation only. Full legality (kings
present, flying-general, etc.) is the rules backend's job.
"""

from __future__ import annotations

# The canonical standard Xiangqi starting position (modern engine dialect,
# Red to move). Note the 'n' = horse and 'b' = elephant in the placement field.
STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r - - 0 1"

# Valid piece letters in the canonical dialect (both colors).
_PIECE_LETTERS = set("KABNRCPkabnrcp")

# Side-to-move tokens accepted on input; 'w' is a tolerant alias for Red.
_SIDE_TOKENS = {"r", "w", "b"}


def _validate_placement(placement: str) -> bool:
    """Validate the piece-placement field (field 1) of a Xiangqi FEN."""
    ranks = placement.split("/")
    if len(ranks) != 10:
        return False
    for rank in ranks:
        if not rank:
            return False
        width = 0
        for ch in rank:
            if ch.isdigit():
                n = int(ch)
                if n < 1 or n > 9:
                    return False
                width += n
            elif ch in _PIECE_LETTERS:
                width += 1
            else:
                return False
        if width != 9:
            return False
    return True


def validate_fen(s: str) -> bool:
    """Return ``True`` iff ``s`` is a structurally valid Xiangqi FEN.

    Accepts both the short form (``"<placement> <side>"``) and the full 6-field
    form (``"<placement> <side> - - <halfmove> <fullmove>"``). Side-to-move may be
    ``r``/``w`` (Red) or ``b`` (Black). This checks structure, not legality.
    """
    if not s or not isinstance(s, str):
        return False
    fields = s.split()
    if len(fields) not in (2, 6):
        return False

    placement, side = fields[0], fields[1]
    if not _validate_placement(placement):
        return False
    if side not in _SIDE_TOKENS:
        return False

    if len(fields) == 6:
        # Fields 3 & 4 are the unused castling / en-passant placeholders.
        if fields[2] != "-" or fields[3] != "-":
            return False
        # Fields 5 & 6 are the halfmove clock and fullmove number.
        halfmove, fullmove = fields[4], fields[5]
        if not (halfmove.isdigit() and fullmove.isdigit()):
            return False
        if int(fullmove) < 1:
            return False

    return True


def side_to_move(s: str) -> str:
    """Return the normalized side-to-move token (``"r"`` or ``"b"``) from a FEN.

    ``"w"`` is normalized to ``"r"`` (Red). Raises :class:`ValueError` on a
    structurally invalid FEN.
    """
    if not validate_fen(s):
        raise ValueError(f"invalid FEN: {s!r}")
    token = s.split()[1]
    return "b" if token == "b" else "r"
