"""ICCS move tokenizer — flat move-index vocabulary over the 9x10 board.

Roadmap milestone: **M1** (landed).

:class:`MoveTokenizer` maps each Xiangqi move to a single integer id (the
"flat move index" scheme; see ADR-0002 and the DESIGN tokenizer section). This is
the flagship tokenizer for the autoregressive move-prediction model: a game
becomes a sequence of move ids, and training is next-move prediction, exactly like
next-token prediction in an LLM.

Vocabulary construction
-----------------------
The vocabulary is the union of every *geometrically possible* move on the empty
9x10 board, independent of position:

* **Straight-line moves** (Chariot, Cannon — and, as subsets, the General,
  Soldier, and the flying-general capture): every same-file and same-rank
  ``(from, to)`` pair.
* **Horse** moves: the eight ``(±1, ±2)`` / ``(±2, ±1)`` L-shapes.
* **Elephant** moves: the four ``(±2, ±2)`` diagonals.
* **Advisor** moves: the four ``(±1, ±1)`` diagonals.

Position-dependent legality (blocking a horse's leg, a cannon's screen, palace
bounds, the river) is *not* encoded here — that is the board's job at decode time
via legal-move masking. The vocabulary only needs to contain every move that
could ever be legal, which the union above guarantees (~2000 moves).

Special tokens ``PAD``/``BOS``/``EOS``/``UNK`` occupy the first ids so a game can
be framed as ``[BOS] m1 m2 ... [EOS]`` and padded in batches.
"""

from __future__ import annotations

from ..core.types import Move

_FILES = "abcdefghi"  # 9 files, a-i  -> column index 0-8
_RANKS = "0123456789"  # 10 ranks, 0-9 -> row index 0-9


def _in_board(f: int, r: int) -> bool:
    return 0 <= f < 9 and 0 <= r < 10


def _sq(f: int, r: int) -> str:
    """Column/row indices -> 2-char ICCS square (e.g. ``(7, 2)`` -> ``"h2"``)."""
    return f"{_FILES[f]}{_RANKS[r]}"


# Offsets for the non-straight-line movers.
_HORSE = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
_ELEPHANT = ((2, 2), (2, -2), (-2, 2), (-2, -2))
_ADVISOR = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _enumerate_move_strings() -> list[str]:
    """Return the sorted list of every geometrically possible ICCS move string."""
    moves: set[str] = set()
    for f in range(9):
        for r in range(10):
            src = _sq(f, r)
            # Straight lines: same file (all other ranks) + same rank (all files).
            for tr in range(10):
                if tr != r:
                    moves.add(src + _sq(f, tr))
            for tf in range(9):
                if tf != f:
                    moves.add(src + _sq(tf, r))
            # Leapers / diagonal movers.
            for offsets in (_HORSE, _ELEPHANT, _ADVISOR):
                for df, dr in offsets:
                    tf, tr = f + df, r + dr
                    if _in_board(tf, tr):
                        moves.add(src + _sq(tf, tr))
    return sorted(moves)


class MoveTokenizer:
    """Reversible mapping between ICCS moves and integer token ids.

    Implements the :class:`~dongfeng.tokenizer.base.Tokenizer` protocol. Ids are
    deterministic and stable across runs (the vocabulary is a sorted enumeration),
    so a checkpoint's ids stay valid for the life of this scheme.
    """

    #: Tokenizer identity recorded in ``manifest.json`` and on checkpoints.
    id = "move-v1"

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    _NUM_SPECIAL = 4

    def __init__(self) -> None:
        self._moves: list[str] = _enumerate_move_strings()
        self._move_to_id: dict[str, int] = {
            iccs: i + self._NUM_SPECIAL for i, iccs in enumerate(self._moves)
        }

    # -- Tokenizer protocol -------------------------------------------------

    def encode(self, x: str) -> list[int]:
        """Encode a whitespace-separated string of ICCS moves into token ids.

        A single move (``"h2e2"``) or a space-joined sequence
        (``"h2e2 h9g7 h0g2"``) are both accepted. Unknown/malformed moves map to
        :data:`UNK_ID` rather than raising, so streaming over messy corpora is
        robust.
        """
        return [self._move_to_id.get(tok, self.UNK_ID) for tok in x.split()]

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into a space-joined ICCS move string.

        Special tokens (pad/bos/eos) are dropped; unknown ids render as ``"<unk>"``.
        """
        out: list[str] = []
        for i in ids:
            if i < self._NUM_SPECIAL:
                if i == self.UNK_ID:
                    out.append("<unk>")
                continue  # drop pad/bos/eos
            idx = i - self._NUM_SPECIAL
            out.append(self._moves[idx] if 0 <= idx < len(self._moves) else "<unk>")
        return " ".join(out)

    @property
    def vocab_size(self) -> int:
        """Total number of ids: specials + enumerated moves (~2086)."""
        return self._NUM_SPECIAL + len(self._moves)

    # -- move-level convenience --------------------------------------------

    def encode_move(self, move: Move) -> int:
        """Return the single token id for a :class:`Move` (``UNK_ID`` if unknown)."""
        return self._move_to_id.get(move.iccs, self.UNK_ID)

    def id_to_move(self, token_id: int) -> Move | None:
        """Return the :class:`Move` for a token id, or ``None`` for special ids."""
        idx = token_id - self._NUM_SPECIAL
        if 0 <= idx < len(self._moves):
            return Move.from_iccs(self._moves[idx])
        return None

    def encode_game(self, moves: list[Move], *, add_special: bool = True) -> list[int]:
        """Encode a whole game as ``[BOS] m1 ... mN [EOS]`` (specials optional).

        This is the unit the autoregressive dataset stores per game.
        """
        ids = [self.encode_move(m) for m in moves]
        if add_special:
            return [self.BOS_ID, *ids, self.EOS_ID]
        return ids
