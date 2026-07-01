---
name: selfplay
description: Have Dong Feng play games against itself (or another engine) and print/record them. Use to sanity-check the engine, generate demo games, or produce self-play data for RL.
---

# selfplay

Use this to have engines play each other with no human input — for a quick sanity
check, a demo game, or (later) to generate self-play data for RL/DPO.

## Commands

```bash
# One self-play game from the start, printed to the terminal
uv run dfc selfplay

# Multiple games
uv run dfc selfplay --games 10

# From a specific position
uv run dfc selfplay --fen "<FEN>"

# Control per-move budget and cap game length
uv run dfc selfplay --movetime 500 --max-moves 200

# Pit two named engines against each other (M3+)
uv run dfc selfplay --red neural --black pikafish
```

## When to use

- **Sanity check** after touching the engine or board layer: does it play a full
  legal game to a terminal result?
- **Demo**: show the engine in action.
- **Data generation** (M5): bulk self-play games feed RL/DPO; games and stats are
  recorded to `runs/` and indexed in `manifest.json` (query with the MCP
  `eval_last` / `dfc eval last`, not by reading `runs/`).

## Notes

- Remember Xiangqi's terminal rule: **no legal moves = a loss** for the side to
  move (not a draw). Results are `RED_WIN` / `BLACK_WIN` / `DRAW`.
- In M0 both sides are the legal-random baseline; stronger engines slot in via
  `--red` / `--black` once they exist.
- For strength comparison across many games with Elo, use the **eval-strength**
  skill instead.
