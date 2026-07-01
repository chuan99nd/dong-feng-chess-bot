# CLAUDE.md — AI-native repo guide for Dong Feng

This file is the map for AI agents working in this repo. Read it before touching
code. It tells you where the contracts live, which commands to run, and — most
importantly — **how to avoid burning tokens** on large files you should query
instead of read.

Dong Feng is an LLM-style Xiangqi (Chinese chess) engine. Concept-level design is
in [DESIGN.md](DESIGN.md); milestones in [ROADMAP.md](ROADMAP.md).

## Architecture map (which package does what)

| Package                    | Responsibility                                                                  | Depends on            |
| -------------------------- | ------------------------------------------------------------------------------ | --------------------- |
| `dongfeng.core`            | Rules-agnostic value types + the rules-library-backed board.                    | `cchess` (deferred)   |
| `dongfeng.protocol`        | The **universal engine contract** + conformance harness. No rules-lib imports. | `dongfeng.core`       |
| `dongfeng.engines`         | Concrete engines: baseline, neural, Pikafish wrapper (M2+).                     | `protocol`, `core`    |
| `dongfeng.data`            | Tokenizer + DPXQ/TianTian corpus pipeline (M1).                                 | `core`                |
| `dongfeng.model`           | Transformer + training/distill/RL loops (M2+).                                  | `data`, `core`        |
| `dongfeng.serve`           | UCCI text adapter, gRPC/HTTP serving (M3/M6).                                   | `protocol`, `engines` |
| `dongfeng.cli`             | The `dfc` Typer CLI — the single human/agent entrypoint.                        | everything above      |

### Where the contracts live (the load-bearing files)

Two Protocols are the spine. Almost everything else is an implementation of, or a
caller of, one of these. **Change these deliberately** — they ripple everywhere.

- **`dongfeng.protocol.engine`** — the `Engine` Protocol (`id`, `new_game`,
  `set_position`, `analyze`, `bestmove`, `set_option`, `stop`) plus the value
  types `EngineInfo`, `SearchLimits`, `Analysis`, `ScoredMove`. Every bot
  implements this; every caller (arena, CLI, UCCI adapter) codes against it.
- **`dongfeng.core.board`** — the `Board` Protocol (`fen`, `set_fen`, `turn`,
  `legal_moves`, `is_legal`, `push`, `pop`, `is_check`, `is_game_over`, `result`,
  `clone`, `ascii`) plus `LibBoard` and the `new_board()` factory.

Vocabulary types (`Color`, `GameResult`, `Move`) live in `dongfeng.core.types`.
Positions are always FEN strings; moves are always ICCS `Move` objects. Backend
types (`cchess`) never leak past `core.board`.

## Key commands

Everything runs through `uv`. There is no `pip install`; use `uv sync`.

| Command                                    | What it does                                              |
| ------------------------------------------ | -------------------------------------------------------- |
| `uv sync --extra dev`                      | Create/refresh the venv with dev tools.                  |
| `uv run dfc selfplay`                      | Engine plays itself; prints the game.                    |
| `uv run dfc play`                          | Play a human-vs-engine game in the terminal.             |
| `uv run dfc ucci`                          | Speak the UCCI/UCI text protocol on stdin/stdout.        |
| `uv run dfc eval ...`                      | Run an Elo arena / strength eval (M2+).                  |
| `uv run dfc data ...`                      | Corpus ingestion + tokenization (M1).                    |
| `uv run dfc train ...` / `dfc distill ...` | Training / teacher distillation (M2/M4).                 |
| `uv run pytest`                            | Full test suite.                                         |
| `uv run pytest -q tests/test_protocol_conformance.py` | Just the engine-contract tests.              |
| `uv run ruff format && uv run ruff check --fix` | Format + lint autofix.                             |
| `uv run pyright`                           | Static type check.                                       |

The editing hooks in `.claude/settings.json` run `ruff format` / `ruff check
--fix` after every Edit/Write, and run the protocol + UCCI-adapter tests whenever
you touch `src/dongfeng/protocol` or `src/dongfeng/engines`. You do not need to
run those manually after edits in those areas.

## Conventions

- **Python 3.11+, full type hints.** `from __future__ import annotations` at the
  top of every module. `pyright` in `standard` mode must pass.
- **Ruff** for format + lint (line length 100; rules `E,F,I,UP,B,SIM`).
- **Import root is `dongfeng`.** Import as `from dongfeng.core import Move`, etc.
- **FEN in, ICCS out.** Never expose `cchess` objects across a module boundary;
  convert at the edge in `core.board`.
- **Xiangqi ≠ chess.** 9×10 board, no promotion (moves are always 4 chars), and
  **no legal moves = a LOSS** for the side to move (stalemate is not a draw).
- **New engines must pass conformance.** Add a factory to
  `tests/test_protocol_conformance.py` and assert `run_conformance(...) == []`.
- **Deferred/optional deps import lazily.** `cchess` is imported inside
  `LibBoard.__init__`; the MCP SDK import in `tools/mcp_server.py` is guarded.
  Modules must import even when optional deps are absent.

## To find X, read Y / run Z

| I want to…                              | Read                                             | Or run                                       |
| --------------------------------------- | ------------------------------------------------ | -------------------------------------------- |
| Understand the engine contract          | `src/dongfeng/protocol/engine.py`                | —                                            |
| Understand the board contract           | `src/dongfeng/core/board.py`                     | —                                            |
| Know the move/position vocabulary       | `src/dongfeng/core/types.py`                     | —                                            |
| Validate an engine against the contract | `src/dongfeng/protocol/conformance.py`           | `uv run dfc protocol-check` (M0) / pytest    |
| See the LLM↔Xiangqi concept mapping     | `DESIGN.md`                                       | —                                            |
| Learn the FEN dialect we use            | `docs/protocol/xiangqi-fen.md`, `core/fen.py`    | —                                            |
| Learn ICCS move notation                | `docs/protocol/iccs-notation.md`, `core/notation.py` | —                                        |
| Speak/serve UCCI                        | `docs/protocol/UCCI.md`                          | `uv run dfc ucci`                            |
| Drive Pikafish as a teacher/opponent    | `docs/protocol/pikafish-uci.md`                  | —                                            |
| Know why an architecture choice was made| `docs/adr/`                                       | —                                            |
| See what milestone something belongs to | `ROADMAP.md`                                       | —                                            |
| Check dataset stats                     | **do not read data files** — see below           | `dfc data stats` / MCP `dataset_stats`       |
| Check a checkpoint's metadata           | **do not read checkpoints** — see below          | `dfc ckpt info` / MCP `checkpoint_info`      |
| See the last eval result                | `manifest.json` (runs)                            | `dfc eval last` / MCP `eval_last`            |
| Query the tokenizer vocabulary          | `manifest.json`                                   | MCP `tokenizer_info`                         |

## Token-saving guidance (IMPORTANT)

Datasets, checkpoints, and run logs are large binary/opaque artifacts. **Do not
`cat`, `Read`, or grep them.** They are deliberately git-ignored (`data/`,
`checkpoints/`, `runs/`, `*.nnue`). Everything an agent needs about them is
indexed in [`manifest.json`](manifest.json), which is the single source of truth.

Query that index the cheap way, in order of preference:

1. **MCP query tools** (`tools/mcp_server.py`, stdio): `dataset_stats`,
   `checkpoint_info`, `eval_last`, `position_query(fen)`, `tokenizer_info`. These
   read `manifest.json` and return small JSON — a few hundred tokens, not
   megabytes.
2. **The `dfc` CLI**: `dfc data stats`, `dfc ckpt info <id>`, `dfc eval last`.
   Same data, human-readable.
3. **`manifest.json` directly** — small and safe to read if you just need the raw
   index.

Rule of thumb: if a file is under `data/`, `checkpoints/`, or `runs/`, or ends in
`.nnue`, query it — never open it. Reading one training shard can cost more tokens
than the entire rest of this repo.
