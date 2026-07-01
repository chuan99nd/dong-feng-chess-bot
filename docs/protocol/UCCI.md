# UCCI — Universal Chinese Chess Interface

UCCI is the native text protocol for Xiangqi engines, analogous to UCI for Western
chess. Dong Feng speaks UCCI so it can play in any UCCI-compatible GUI or arena.
`dfc ucci` starts the adapter (`dongfeng.serve.ucci`), which wraps the in-process
`Engine` Protocol (see [DESIGN.md](../../DESIGN.md) §3, Tier 2).

This page is a working reference for the subset Dong Feng implements. It is not the
full historical spec; it captures the commands, the position/move encoding, and
the Dong-Feng-specific normalization rules the adapter must honor.

## Transport

- Plain text over **stdin/stdout**, one command per line, UTF-8.
- The GUI (client) sends commands to the engine; the engine replies with lines.
- The engine must be responsive: acknowledge `isready` promptly and honor `stop`.

## Session lifecycle

```
GUI  -> ucci                      # request UCCI mode
ENG  <- id name Dong Feng
ENG  <- id author Dong Feng contributors
ENG  <- option ...                # zero or more declared options
ENG  <- ucciok                    # handshake complete
GUI  -> setoption <name> <value>  # optional configuration
GUI  -> isready
ENG  <- readyok
GUI  -> position ...              # set the root position
GUI  -> go ...                    # start searching
ENG  <- info ...                  # optional progress lines
ENG  <- bestmove <move>           # the chosen move
GUI  -> quit
```

## Commands the engine accepts

| Command                                   | Meaning                                                                 |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| `ucci`                                    | Enter UCCI mode; reply with `id` / `option` lines then `ucciok`.        |
| `isready`                                 | Reply `readyok` once ready.                                             |
| `setoption <name> <value>`                | Set an engine option (maps to `Engine.set_option`).                     |
| `position startpos [moves m1 m2 ...]`     | Root = starting position, then apply moves.                             |
| `position fen <FEN> [moves m1 m2 ...]`    | Root = given FEN, then apply moves.                                     |
| `go <limits>`                             | Search under limits (see below); reply `bestmove`.                      |
| `stop`                                    | Stop the current search ASAP; reply `bestmove` with the best so far.    |
| `uccinewgame`                             | Reset per-game state (maps to `Engine.new_game`).                       |
| `quit`                                    | Terminate the engine process.                                          |

### `go` limit tokens (mapped to `SearchLimits`)

| Token             | `SearchLimits` field | Notes                                  |
| ----------------- | -------------------- | -------------------------------------- |
| `depth <n>`       | `depth`              | Max search depth in plies.             |
| `nodes <n>`       | `nodes`              | Max nodes.                             |
| `movetime <ms>`   | `movetime_ms`        | Fixed time for this move.              |
| `time <ms>` (Red) | `wtime_ms`           | Red/"white" clock remaining.           |
| `opptime <ms>`    | `btime_ms`           | Opponent (Black) clock remaining.      |

Any subset may appear; unspecified limits are `None` ("engine's discretion").

## Replies the engine sends

| Reply                                        | Meaning                                              |
| -------------------------------------------- | --------------------------------------------------- |
| `id name <name>` / `id author <author>`      | From `EngineInfo`.                                   |
| `option name <n> type <t> ...`               | A declared option.                                  |
| `ucciok` / `readyok`                         | Handshake / ready acks.                             |
| `info depth <d> score cp <x> pv <moves> ...` | Optional search progress (maps from `Analysis`).    |
| `bestmove <move>`                            | The chosen move in ICCS coordinates.                |
| `nobestmove`                                 | No legal move available (terminal position).        |

`info` may carry `score cp <cp>`, `score mate <n>`, `nodes`, `time`, `pv`, and —
if supported — WDL. These map from `ScoredMove` / `Analysis`.

## Position and move encoding

- **Positions are Xiangqi FEN.** See [xiangqi-fen.md](xiangqi-fen.md). `startpos`
  is shorthand for the canonical starting FEN.
- **Moves are ICCS coordinate strings**, 4 characters, `<from><to>`, files `a-i`,
  ranks `0-9`, e.g. `h2e2`. See [iccs-notation.md](iccs-notation.md).
- **No promotion** — a move is always exactly 4 chars; reject a 5th character.

## Dong Feng normalization rules (invariants)

The adapter enforces these at the boundary so no back-end quirk leaks inward:

1. **ICCS coordinates only, ranks `0-9`.** If any wrapped back-end emits `1-10`
   ranks (e.g. Fairy-Stockfish/pyffish for adjudication), subtract 1 from each
   rank and re-encode before it crosses the boundary. Pikafish already uses `0-9`.
2. **FEN in, FEN out.** Positions are exchanged as FEN; internal `cchess` objects
   never appear in protocol text.
3. **Side-to-move tolerance.** Accept `r` and `w` as "Red to move" on input; emit
   `r` (see the FEN doc).
4. **Legal moves only.** `bestmove` is validated against `Board.legal_moves()`
   before it is sent — the same legal mask the neural engine decodes under.

These are exercised by `tests/test_ucci_adapter.py`.

## References

- ADR: [0003-protocol-ucci-and-pikafish-uci.md](../adr/0003-protocol-ucci-and-pikafish-uci.md)
- Pikafish's UCI dialect: [pikafish-uci.md](pikafish-uci.md)
- FEN / ICCS encodings: [xiangqi-fen.md](xiangqi-fen.md), [iccs-notation.md](iccs-notation.md)
