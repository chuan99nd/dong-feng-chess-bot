# 4. Wrap a rules library (cchess) instead of hand-rolling Xiangqi rules

- **Status:** Accepted
- **Date:** 2026-07-01

## Context

Correct Xiangqi rules are subtle: horse-leg blocking, elephant-eye blocking,
cannon-screen captures, palace confinement, the flying-general constraint,
check/mate, and the fact that *no legal moves is a loss* (not a draw). Getting all
of these right — and staying right — is a substantial, bug-prone effort we do not
want on our critical path. Legal-move generation is also the backbone of
constrained decoding, so it must be authoritative.

The options considered (from the rules research):

1. **PyPI `cchess` (walker8088)** — pip-installable, actively maintained,
   LGPL-3.0. Non-python-chess API; `create_moves()` is *pseudo-legal* and must be
   filtered with `is_checked_move()`. All hard rules correct. No push/pop stack.
2. **`python-chinese-chess` (windshadow233)** — clean python-chess-style API with
   built-in legal filtering and push/pop, but **GPL-3.0** and **not on PyPI**
   (git/vendored only). Import name **collides** with (1).
3. **`pyffish` (Fairy-Stockfish)** — engine-grade, best perpetual/chasing
   adjudication, on PyPI, but stateless and uses ranks `1-10` (not ICCS `0-9`).
4. **Hand-roll** the rules ourselves.

## Decision

- **Wrap PyPI `cchess` (walker8088)** behind our own `Board` Protocol
  (`dongfeng.core.board.LibBoard`). Rationale: PyPI-published, actively
  maintained, permissive-ish (LGPL-3.0, invoked as a wrapped dependency),
  all Xiangqi rules correct. Pinned `cchess>=1.25,<2`.
- **Do not hand-roll** the rules — a mature library exists.
- **Encapsulate the backend's sharp edges** inside `LibBoard`:
  - `create_moves()` is pseudo-legal → filter each candidate through
    `is_checked_move()` to get genuinely legal moves (handles self-check *and*
    flying-general).
  - No push/pop stack → maintain an internal snapshot stack for `pop()`.
  - `move()` does not flip the side to move → call `next_turn()` explicitly.
  - "No legal moves" → map to a **loss** for the side to move in `result()`.
- **Never leak `cchess` types** past `core.board`; the rest of the codebase sees
  only FEN strings and ICCS `Move`s.
- **Defer the import** to `LibBoard.__init__` with a clear install hint, so the
  package imports even without the backend present.

The import-name collision with `python-chinese-chess` is a documented hazard; only
the PyPI `cchess` is supported. The repetition/perpetual-check adjudication gap in
pure-Python libs is accepted for now and closed later by layering an engine
(Pikafish / Fairy-Stockfish `is_optional_game_end`) at M5.

## Consequences

- Correct, maintained rules with minimal in-house code; legal-move generation is
  trustworthy enough to base constrained decoding on.
- A single, swappable seam: replacing the backend (or adding pyffish for
  adjudication) means changing only `core.board`, because callers depend on the
  `Board` Protocol, not on `cchess`.
- We inherit `cchess`'s pseudo-legal/no-undo quirks, but they are contained and
  covered by the board test suite. LGPL-3.0 obligations apply to the wrapped
  dependency only.
