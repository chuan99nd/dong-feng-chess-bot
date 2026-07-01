---
name: protocol-check
description: Verify that an engine conforms to the universal Engine Protocol using the conformance harness. Use after adding or changing an engine, or before trusting a bot in a match/arena.
---

# protocol-check

Use this to confirm an engine satisfies the universal `Engine` contract before it
is trusted in a match, arena, or UCCI session. The harness validates move legality
against the **real** rules backend, so a passing engine provably returns legal
moves.

## Commands

```bash
# Check a registered engine
uv run dfc protocol-check --engine <name>

# Focused test subset (also auto-run by the editing hook on protocol/engines edits)
uv run pytest -q tests/test_protocol_conformance.py
```

## What it checks

`dongfeng.protocol.conformance.run_conformance(make_engine)` runs a factory and
returns a list of failure messages (**empty == conforms**). It verifies:

- `id()` returns a valid `EngineInfo` with a non-empty name.
- `bestmove()` from the **starting position** returns a **legal** `Move`.
- `bestmove()` is still legal **after a couple of opening moves**.
- `new_game()` then `bestmove()` yields a legal move.
- `analyze()` returns an `Analysis` whose `best.move` is legal.

## In code

```python
from dongfeng.protocol import run_conformance
from dongfeng.engines.my_engine import MyEngine

failures = run_conformance(lambda: MyEngine(...))
assert failures == [], failures
```

## When to use

- Right after **adding a new engine** (pair with the **add-model** skill).
- After changing anything in `src/dongfeng/protocol` or `src/dongfeng/engines`.
- Before running an **arena** or exposing an engine over **UCCI**.

## Notes

- Read the exact contract in `src/dongfeng/protocol/engine.py` and the checks in
  `src/dongfeng/protocol/conformance.py`.
- Legality here follows Xiangqi rules, including flying-general and "no legal moves
  = loss".
