# ICCS move notation

ICCS (Internet/International Chinese Chess Server) coordinate notation is Dong
Feng's **canonical move notation**. It matches `dongfeng.core.types.Move` and
`dongfeng.core.notation`.

## Axes and orientation

- **Files:** letters `a`–`i`, left to right.
- **Ranks:** digits `0`–`9`, bottom to top.
- **Origin `a0`** is Red's bottom-left corner: file `a` is on Red's left, rank `0`
  is Red's home rank.

Reference squares (from the starting position):

| Piece            | Red   | Black |
| ---------------- | ----- | ----- |
| General          | `e0`  | `e9`  |
| Chariots         | `a0`, `i0` | `a9`, `i9` |
| Cannons          | `b2`, `h2` | `b7`, `h7` |

## Move format

A move is a **4-character** string: source square then destination square.

```
h2e2        # from h2 to e2
```

- `<from-file><from-rank><to-file><to-rank>`, each square is 2 chars.
- **No promotion exists in Xiangqi** → a move is *always* exactly 4 characters.
  A 5th character (a promotion suffix like chess's `e7e8q`) is **invalid** and is
  rejected by `Move.from_iccs`.
- Some sources write a dash (`H2-E2`); Dong Feng's canonical form is the compact
  lowercase 4-char string `h2e2`.

## In code

```python
from dongfeng.core import Move

m = Move.from_iccs("h2e2")   # parse; raises ValueError if not 4 valid ICCS chars
m.from_sq                    # "h2"
m.to_sq                      # "e2"
m.iccs                       # "h2e2"
str(m)                       # "h2e2"
```

Helpers in `dongfeng.core.notation`: `is_iccs_move(s)`, `parse_iccs(s)`,
`format_iccs(move)`.

## Example opening line

```python
# 1. Red central cannon (WXF C2=5)  -> h2e2
# 1... Black horse       (WXF H8+7) -> h9g7
# 2. Red right horse     (WXF H2+3) -> h0g2
# 2... Black chariot     (WXF R9=8) -> i9h9
moves = ["h2e2", "h9g7", "h0g2", "i9h9"]
```

## Relationship to WXF relative notation

WXF (columnar/relative) notation — e.g. `C2=5`, `H2+3`, `P7+1`, `+R+1` — is
**side-relative**: file numbers are counted from each player's own right and are
**mirrored** between Red and Black, so converting to/from ICCS **requires knowing
whose turn it is** (hence the `fen` argument in `notation.wxf_to_iccs` /
`iccs_to_wxf`, planned for M1). Operators: `+` advance, `-` retreat, `=` traverse;
after `+`/`-` the trailing digit is a step count for straight movers (R/C/P/K) but
a destination file for diagonal/L movers (N/A/B). Kept in a separate converter
layer; ICCS is the canonical wire format everywhere else.

## Coordinate-convention warning

ICCS ranks are **`0`–`9`**. Some engines/back-ends (Fairy-Stockfish / pyffish) use
ranks **`1`–`10`** (e.g. `h3h10`), which is *not* ICCS. Pikafish's UCI *does* use
ICCS `0`–`9`. Any `1`–`10` back-end must be remapped at the protocol boundary
(subtract 1 from each rank; handle the two-digit `10`). See
[UCCI.md](UCCI.md) normalization rule 1 and [pikafish-uci.md](pikafish-uci.md).

## References

- Implementation: `src/dongfeng/core/types.py`, `src/dongfeng/core/notation.py`
- FEN encoding: [xiangqi-fen.md](xiangqi-fen.md)
