# Dong Feng 东风 — an LLM-style Xiangqi AI

Dong Feng ("East Wind") is a Xiangqi (Chinese chess) engine built the way modern
language models are built: a **tokenizer** turns positions and moves into a
sequence, an **autoregressive transformer** predicts the next move, and the same
policy is refined by **distillation** from a strong teacher (Pikafish) and by
**self-play RL / DPO**. Legality is guaranteed by **constrained decoding** (legal
move masking) rather than hoping the model learns the rules perfectly.

The result plugs into the real world through a **UCCI/UCI text protocol**, so it
can play in any Xiangqi GUI or arena, and against Pikafish, exactly like a
conventional engine.

> **Status: M0.** The FOUNDATION spine (core types, board, engine protocol,
> conformance) is in place. Training, the neural engine, and serving arrive in
> later milestones — see [ROADMAP.md](ROADMAP.md).

## Why "LLM-style"?

| LLM concept                | Dong Feng equivalent                                    |
| -------------------------- | ------------------------------------------------------- |
| Tokenizer / vocabulary     | FEN + ICCS move tokenizer (`dongfeng.data`)             |
| Pretraining corpus         | DPXQ / TianTian human game scores → `(FEN, move)` pairs |
| Next-token prediction      | Next-**move** prediction (behavior cloning)             |
| SFT                        | Distillation from Pikafish + Elo-conditioning           |
| RLHF                       | Self-play + DPO on preference pairs                     |
| Constrained decoding       | Legal-move masking via `Board.legal_moves()`            |
| Inference server           | UCCI server (`dfc ucci`), gRPC/HTTP later               |
| Evals                      | Elo arena, tactical suites, teacher agreement           |

Full details in [DESIGN.md](DESIGN.md).

## Quickstart

Dong Feng uses [uv](https://docs.astral.sh/uv/) for environment and task running.

```bash
# 1. Install dependencies (creates .venv, resolves cchess/typer/rich, dev extras)
uv sync --extra dev

# 2. Watch the engine play itself
uv run dfc selfplay

# 3. Run the test suite (protocol conformance, board rules, adapters)
uv run pytest

# 4. Use Dong Feng as a UCCI/UCI engine (speaks the text protocol on stdin/stdout)
uv run dfc ucci
```

Point any UCCI-compatible GUI at `uv run dfc ucci` to play against Dong Feng, or
run an arena match against Pikafish. See [docs/protocol/UCCI.md](docs/protocol/UCCI.md).

> Some subcommands (`train`, `distill`, `data`) are milestone features and will
> print a "planned: Mx" notice until their milestone lands. The engine used by
> `selfplay` / `ucci` in M0 is a legal-random baseline; the neural engine arrives
> in M2–M3.

## Repo map

```
dong-feng-chess-bot/
├── src/dongfeng/
│   ├── core/            # rules-agnostic types + rules-library-backed board
│   │   ├── types.py     #   Color, GameResult, Move (ICCS)
│   │   ├── board.py     #   Board Protocol + LibBoard + new_board() factory
│   │   ├── fen.py       #   STARTING_FEN, validate_fen, side_to_move
│   │   └── notation.py  #   ICCS <-> WXF notation helpers
│   ├── protocol/        # the universal engine contract
│   │   ├── engine.py    #   Engine Protocol, EngineInfo, SearchLimits, Analysis, ScoredMove
│   │   └── conformance.py  # run_conformance(): validate any engine against the contract
│   ├── engines/         # concrete engines (baseline, neural, Pikafish wrapper) — M2+
│   ├── data/            # tokenizer + corpus pipeline — M1
│   ├── model/           # transformer + training — M2+
│   ├── serve/           # UCCI adapter, gRPC/HTTP — M3/M6
│   └── cli.py           # `dfc` Typer CLI
├── docs/
│   ├── adr/             # Architecture Decision Records
│   └── protocol/        # UCCI, Pikafish-UCI, FEN, ICCS reference docs
├── tools/
│   └── mcp_server.py    # token-efficient MCP query server over manifest.json
├── .claude/
│   ├── settings.json    # safe-editing hooks (ruff, targeted tests)
│   └── skills/          # agent skills: play-game, selfplay, eval-strength, ...
├── manifest.json        # artifacts index (datasets / checkpoints / runs)
├── README.md            # you are here
├── CLAUDE.md            # AI-native repo guide (read this if you are an agent)
├── DESIGN.md            # full architecture
└── ROADMAP.md           # milestones M0..M6
```

## For AI agents

Read [CLAUDE.md](CLAUDE.md) first. It is the token-efficient map of the repo: the
contracts, the commands, a "to find X, read Y / run Z" table, and guidance on
using the `dfc` CLI and MCP queries instead of reading large data/checkpoint
files.

## The contracts

Two Protocols are the load-bearing interfaces everything else codes against:

- **`dongfeng.core.board.Board`** — the mutable board contract. Positions are FEN
  strings, moves are ICCS `Move` objects. Backed by `cchess` (walker8088).
- **`dongfeng.protocol.engine.Engine`** — the universal bot contract. Every
  engine (baseline, neural, Pikafish wrapper) implements it, so any conforming
  engine is a drop-in for any other. Validate with `run_conformance()`.

## License

MIT. Note the runtime dependency `cchess` (walker8088) is LGPL-3.0; Pikafish, if
used as a teacher/opponent, is GPL-3.0 and is invoked as a separate process, not
linked.
