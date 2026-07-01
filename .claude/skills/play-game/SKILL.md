---
name: play-game
description: Play an interactive human-vs-engine Xiangqi game against Dong Feng in the terminal. Use when the user wants to play against the bot, try a position, or watch the engine respond to specific moves.
---

# play-game

Use this when the user wants to **play against Dong Feng** interactively, or step
through a specific position and see the engine's replies.

## Commands

```bash
# Play from the starting position (you are Red by default)
uv run dfc play

# Choose your side
uv run dfc play --side red
uv run dfc play --side black

# Start from a specific position (Xiangqi FEN)
uv run dfc play --fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r - - 0 1"

# Limit engine thinking time / depth per move
uv run dfc play --movetime 1000        # ms per engine move
uv run dfc play --depth 12
```

Moves are entered in **ICCS** coordinate notation, 4 chars, e.g. `h2e2`
(see `docs/protocol/iccs-notation.md`). Type `quit` to exit.

## Notes

- Positions are Xiangqi FEN; see `docs/protocol/xiangqi-fen.md`.
- Only **legal** moves are accepted (the board rejects illegal input).
- In M0 the opponent is the legal-random baseline engine; the neural engine
  (M3) drops in behind the same command with no CLI change.
- To watch two engines instead of playing yourself, use the **selfplay** skill.
