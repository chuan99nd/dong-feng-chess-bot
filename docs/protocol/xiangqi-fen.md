# Xiangqi FEN

Dong Feng exchanges every position as a Xiangqi FEN string. This document defines
the **canonical dialect** we emit and the tolerant inputs we accept. It matches
`dongfeng.core.fen` (`STARTING_FEN`, `validate_fen`, `side_to_move`).

## Six fields

A Xiangqi FEN is one line of six space-separated fields (chess-FEN-compatible):

```
<placement> <side> <castling> <enpassant> <halfmove> <fullmove>
```

1. **Placement** — 10 rank-groups separated by `/`.
2. **Side to move** — `r`/`w` (Red) or `b` (Black).
3. **Castling** — unused in Xiangqi; always `-`.
4. **En passant** — unused in Xiangqi; always `-`.
5. **Halfmove clock** — plies since the last capture or pawn move.
6. **Fullmove number** — starts at 1, increments after each Black move.

Fields 3 and 4 are meaningless in Xiangqi but must be present (`-`) so
chess-FEN-compatible parsers do not break. Dong Feng also accepts a short
2-field form (`<placement> <side>`) on input; it emits the full 6-field form.

## Canonical starting position

```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r - - 0 1
```

This is `dongfeng.core.fen.STARTING_FEN`. Note `n` = horse and `b` = elephant,
which proves the modern piece-letter convention below.

## Board geometry (9×10)

- The board is **9 files wide × 10 ranks tall** = 90 points. Pieces sit on
  intersections.
- Each rank-group has **9 points** (files `a`..`i`, left→right). Piece letters and
  empty-run digits within a group **must sum to exactly 9**. Max empty-run digit
  is `9` (a fully empty rank is the single char `9`). This differs from chess,
  where a rank is 8 wide.
- Rank-groups are written **top rank first, bottom rank last**: the **first**
  group is Black's back rank (ICCS rank 9); the **last** (10th) group is Red's back
  rank (ICCS rank 0).

### Rank-group ↔ ICCS rank mapping

| FEN group index | ICCS rank | Which side           |
| --------------- | --------- | -------------------- |
| 0 (first)       | 9         | Black back rank      |
| 1               | 8         |                      |
| …               | …         |                      |
| 8               | 1         |                      |
| 9 (last)        | 0         | Red back rank        |

Within a group, characters map to files `a`..`i` left→right; a digit is a run of
that many empty points.

## Piece letters

Uppercase = Red, lowercase = Black.

### Canonical (modern engine) convention — what Dong Feng uses

| Letter | Piece                 |
| ------ | --------------------- |
| K / k  | General (King)        |
| A / a  | Advisor (Guard)       |
| B / b  | Elephant (Bishop)     |
| N / n  | Horse (kNight)        |
| R / r  | Chariot (Rook)        |
| C / c  | Cannon                |
| P / p  | Pawn (Soldier)        |

This is the set used by the canonical starting FEN, xiangqi.js, and the `cchess`
backend.

### Legacy WXF-FEN convention — tolerant input only

The 1990s WXF FEN spec uses **`E` for Elephant** and **`H` for Horse** (keeping
`R` rook and `C` cannon). So `K A E R C H P` instead of `K A B N R C P`. Dong Feng
may accept these when a source is *explicitly* WXF-FEN, but **does not
auto-detect** per character (the two dialects share the same geometry and would be
ambiguous). Default and emitted output is always the modern set.

## Side-to-move dialect

- **Input:** accept `r` **and** `w` as "Red to move" (`w` is the WXF-spec spelling);
  accept `b` as Black.
- **Output:** Dong Feng emits `r` (engine-interop convention). `side_to_move()`
  normalizes any input to `"r"` or `"b"`.

## Validation scope

`dongfeng.core.fen.validate_fen` checks **structure only**: 10 rank-groups, each
summing to 9, valid letters/digits, a valid side token, and (for the 6-field form)
`-`/`-` placeholders plus numeric counters with `fullmove >= 1`. It does **not**
check legality (kings present, flying-general, etc.) — that is the rules backend's
job (`dongfeng.core.board`).

## Gotchas

- Rank width is **9**, not 8; the biggest empty-run digit is `9`.
- FEN is **top-first**: the first `/`-group is **Black's** back rank, not Red's.
- Fields 3 and 4 must be present as `-` even though they are unused.
- Two piece-letter dialects share the same geometry — never mix `B/N` with `E/H`.
- Side-to-move is `r`/`w`/`b`, not always the chess-style `w`/`b`.

## References

- Implementation: `src/dongfeng/core/fen.py`
- Move notation: [iccs-notation.md](iccs-notation.md)
- WXF FEN spec: https://www.wxf-xiangqi.org/images/computer-xiangqi/fen-for-xiangqi-chinese-chess.pdf
