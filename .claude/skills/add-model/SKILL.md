---
name: add-model
description: Add a new engine to Dong Feng — a new neural checkpoint wrapper, a baseline, or a third-party engine — that conforms to the Engine Protocol. Use when creating or registering a new bot.
---

# add-model

Use this when adding a **new engine** (bot) to the ecosystem: a wrapper around a
new checkpoint, a baseline, or a third-party engine. Every engine must implement
the universal `Engine` Protocol and pass conformance so it is a drop-in for any
other.

## The contract

Implement `dongfeng.protocol.engine.Engine`:

- `id() -> EngineInfo`
- `new_game() -> None`
- `set_position(fen: str, moves: list[Move]) -> None`
- `analyze(limits: SearchLimits) -> Analysis`
- `bestmove(limits: SearchLimits) -> Move`
- `set_option(name: str, value: str) -> None`
- `stop() -> None`

Read the exact signatures in `src/dongfeng/protocol/engine.py`. Positions are FEN
strings; moves are ICCS `Move` objects. Never leak backend types.

## Steps

1. Create the engine module under `src/dongfeng/engines/<name>_engine.py`.
2. Return legal moves only — decode/search under a **legal mask** from
   `Board.legal_moves()` (see the constrained-decoding design in `DESIGN.md`).
3. Add a conformance test in `tests/test_protocol_conformance.py`:

   ```python
   from dongfeng.protocol import run_conformance
   from dongfeng.engines.my_engine import MyEngine

   def test_my_engine_conforms():
       assert run_conformance(lambda: MyEngine(...)) == []
   ```

4. Register the engine name in the CLI/arena factory so `dfc selfplay --red <name>`
   and `dfc ucci --engine <name>` can select it.

## Verify

```bash
# Quick conformance check (the harness validates legality against the real rules)
uv run dfc protocol-check --engine <name>

# Or run the focused test subset (also auto-run by the editing hook)
uv run pytest -q tests/test_protocol_conformance.py
```

## Notes

- Editing anything under `src/dongfeng/protocol` or `src/dongfeng/engines`
  triggers the protocol/adapter test hook automatically (`.claude/settings.json`).
- For a checkpoint-backed engine, see the **checkpoint-inspect** skill to confirm
  the checkpoint's metadata before wrapping it.
