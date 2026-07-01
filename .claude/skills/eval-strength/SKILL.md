---
name: eval-strength
description: Measure how strong a Dong Feng checkpoint or engine is — Elo arena vs. baselines/Pikafish, move-accuracy vs. held-out games, tactical/mate suites. Use to compare checkpoints or track training progress.
---

# eval-strength

Use this to answer "how good is this engine/checkpoint?" — for comparing two
checkpoints, tracking progress across training runs, or benchmarking against
Pikafish.

## Commands

```bash
# Show the most recent eval result (cheap; reads manifest.json)
uv run dfc eval last

# Run an Elo arena between two engines
uv run dfc eval arena --red neural --black pikafish --games 100

# Move-accuracy vs. a held-out set of human games (top-1 / top-k)
uv run dfc eval accuracy --checkpoint <id> --dataset <id>

# Tactical / mate-in-N suite
uv run dfc eval tactics --checkpoint <id>
```

## When to use

- Deciding whether a **new checkpoint** beats the previous one.
- Tracking **training progress** across runs (M2+).
- Benchmarking Dong Feng against **Pikafish** (M3+; see
  `docs/protocol/pikafish-uci.md`).

## Reading results — do NOT read `runs/`

Eval outputs are recorded to `runs/` and indexed in `manifest.json`. **Query the
index, never read the run files** (they are large and git-ignored):

- `uv run dfc eval last`, or
- MCP tool `eval_last` (`tools/mcp_server.py`).

## Metrics you'll see

- **Elo** (relative rating vs. opponents), with confidence interval.
- **Top-1 / top-k move accuracy** vs. held-out human games.
- **Legal-move rate** (should be 100% given legal masking).
- **Teacher agreement** (fraction within 100cp of Pikafish's best — the GoodMove
  metric).

## Notes

- The full eval harness lands with M2; `dfc eval last` works against whatever is
  already in `manifest.json`.
