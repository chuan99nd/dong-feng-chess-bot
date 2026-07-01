# 2. Autoregressive transformer as the flagship paradigm

- **Status:** Accepted
- **Date:** 2026-07-01

## Context

Dong Feng's thesis is that Xiangqi can be modeled the way LLMs model text: a game
is a sequence, a position is context, a move is the next token. Several model
paradigms could realize a strong Xiangqi bot:

1. **Classical alpha-beta search + handcrafted/NNUE eval** (Pikafish's approach).
2. **AlphaZero-style MCTS + policy/value network** (search-heavy).
3. **Autoregressive transformer policy** trained by behavior cloning, then
   distillation and preference optimization — "grandmaster without search"-style
   single-forward-pass play.

We want maximal reuse of the modern LLM toolchain and a fast, search-free move at
inference time, while keeping the door open to value estimation and optional
search later.

## Decision

The **flagship paradigm is an autoregressive transformer policy** over the move
sequence, with a **reserved action-value (WDL) head** designed in from the start
but populated later (M4 distillation).

- **Primary head:** a distribution over the move vocabulary, **masked to legal
  moves** at decode time (see ADR-0004 and the constrained-decoding design). BC
  pretraining, distillation, and DPO all target this head.
- **Reserved value head:** trained from Pikafish per-mille WDL + centipawn scores.
  Reserved now so `ScoredMove` (`score_cp`, `win_prob`, `policy_prob`) is stable;
  filled at M4. Enables MCTS-free value estimates and optional shallow search
  without an architectural change.

We explicitly do **not** build MCTS or alpha-beta as the flagship. A
search-based engine remains possible later purely as another `Engine`
implementation behind the Protocol.

## Consequences

- **Pros:** reuse of tokenizer/transformer/SFT/DPO/KV-cache tooling; single
  forward pass per move (fast, cheap to serve); clean distillation target; the
  reserved value head future-proofs value/search work.
- **Cons:** a pure policy may be tactically weaker than deep search at equal
  compute; perpetual-check/chasing subtleties need engine-based adjudication
  (deferred, see roadmap M5). Mitigations: teacher distillation, optional
  value-guided shallow search later, legal masking for zero illegal moves.
- The `Engine` Protocol makes this reversible at the *engine* level: if a
  search-based approach proves stronger, it slots in without touching callers.
