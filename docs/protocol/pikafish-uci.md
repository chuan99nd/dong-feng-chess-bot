# Pikafish UCI dialect (teacher + opponent)

[Pikafish](https://github.com/official-pikafish/Pikafish) is the strongest
open-source Xiangqi engine (Stockfish-derived, NNUE, ~Elo 3950). Dong Feng drives
it over its **UCI-flavored** protocol for two purposes:

1. **Distillation teacher** (M4): extract per-move policy + value labels.
2. **Opponent / reference** for strength evals (M3+).

`dongfeng.engines.pikafish_engine` wraps a Pikafish subprocess and exposes it as
an ordinary `Engine` (see the [Engine Protocol](../../src/dongfeng/protocol/engine.py)).

Pikafish is invoked as a **separate process** (GPL-3.0); it is never linked into
Dong Feng. It needs a `.nnue` net via `EvalFile`.

## Handshake and options

Pikafish uses UCI-style commands (`uci`/`uciok`, not `ucci`/`ucciok`). The wrapper
translates between our `Engine` calls and these lines.

```
uci
setoption name Threads value 8
setoption name Hash value 4096
setoption name EvalFile value pikafish.nnue
setoption name UCI_ShowWDL value true      # adds per-mille W/D/L to info lines
setoption name MultiPV value 5             # top-N moves, each with its own score
isready
position fen <FEN> [moves ...]
go depth 25
```

| Option           | Purpose                                                            |
| ---------------- | ----------------------------------------------------------------- |
| `Threads`        | Search threads.                                                    |
| `Hash`           | Transposition table size (MB).                                     |
| `EvalFile`       | Path to the NNUE net (required).                                   |
| `UCI_ShowWDL`    | Emit `wdl <W> <D> <L>` (per-mille) — a ready-made **value** target.|
| `MultiPV`        | Report the top *N* lines — the raw material for **policy** targets.|

## Reading `info` lines (label extraction)

A typical MultiPV info line:

```
info depth 25 multipv 1 score cp 120 wdl 720 250 30 nodes 1234567 time 900 pv b0c2 h9g7 ...
```

Parse per line:

- `multipv <k>` — rank of this line (1 = best).
- `score cp <x>` — centipawns from the side-to-move's perspective →
  `ScoredMove.score_cp`.
- `score mate <n>` — forced mate in `n` (instead of `cp`).
- `wdl <W> <D> <L>` — per-mille; `win_prob = W / 1000` → `ScoredMove.win_prob`.
- `pv <m1> <m2> ...` — principal variation → `ScoredMove.pv`; `m1` is the move.

Softmax the MultiPV `score_cp` values across the *k* lines to form a policy target
distribution (`ScoredMove.policy_prob`). This mirrors the Xiangqi-R1 recipe.

## Distillation label recipe (M4)

From the Xiangqi-R1 precedent:

- Run Pikafish at **depth ~25** where compute allows (they applied deep search to
  a subset, not every position — tune depth vs. `MultiPV` vs. total compute).
- Label a candidate move **GoodMove** if it is within **100 cp** of the engine's
  best move.
- For a **value** target, prefer per-mille **WDL → win probability** directly.
- For a **policy** target, softmax the top-*k* MultiPV `score_cp` into a
  distribution.
- 5-class positional-advantage bucketing uses centipawn thresholds
  σ_s = 100 and σ_l = 800.

## Coordinate convention (important)

Pikafish's UCI moves use the **ICCS convention: files `a-i`, ranks `0-9`**
(e.g. `b0c2`, `h9g7`). This matches Dong Feng's canonical ICCS notation directly —
**no rank remap is needed** for Pikafish.

Contrast: **Fairy-Stockfish / pyffish** use ranks **`1-10`** (e.g. `h3h10`), which
is *not* ICCS. If pyffish is used later for perpetual/chasing adjudication, its
moves must be remapped (subtract 1 from each rank, handle the two-digit `10`)
before crossing into Dong Feng. See [UCCI.md](UCCI.md) normalization rule 1.

## WDL caveat

`UCI_ShowWDL` numbers are calibrated to Pikafish/Stockfish self-play conditions,
not ground-truth human win rates. They are excellent *relative* value targets;
recalibrate if you need absolute human win probabilities.

## References

- ADR: [0003-protocol-ucci-and-pikafish-uci.md](../adr/0003-protocol-ucci-and-pikafish-uci.md)
- Pikafish: https://github.com/official-pikafish/Pikafish · UCI commands wiki:
  https://github.com/official-pikafish/Pikafish/wiki/UCI-&-Commands
- Xiangqi-R1 (label recipe): https://arxiv.org/html/2507.12215v2
