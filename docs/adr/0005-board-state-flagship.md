# 5. Board-state transformer as a second, scale-first flagship

- **Status:** Accepted
- **Date:** 2026-07-04

## Context

ADR-0002 established an **autoregressive** transformer over the move *sequence* as
the flagship paradigm: a game is a sequence, a position is context, the next move
is the next token. That model consumes a variable-length move history and predicts
the next move token.

Two practical pressures pushed us to add a second model family rather than only
scaling the autoregressive one:

1. **Hardware portability + scale.** We want the same code to train on an Apple
   M1 (MPS, fp16) for local iteration and on a rented RTX 5090 (CUDA, bf16) for
   the ~1B-parameter target. A fixed-length, position-in / move-out model is
   simpler to batch, checkpoint, and shard across those two backends than a
   variable-length autoregressive one.
2. **A position is enough.** Xiangqi has no move-repetition state that a FEN plus
   side-to-move cannot capture for policy purposes (repetition/perpetual
   adjudication is deferred to M5 regardless). Predicting the next move from the
   *board state alone* removes the history dimension, which makes the input a
   fixed 91-token grid and the target a single move — a clean, dense supervised
   signal that scales predictably.

## Decision

We add a **board-state transformer** (`dongfeng.model.board_transformer`) as a
scale-first flagship alongside the autoregressive policy, sharing the same
`Engine` Protocol and move vocabulary:

- **Input:** a fixed-length **91-token** board encoding — the 90 board points
  (`board-v1` per-point tokenizer) plus one **side-to-move** token at index 90.
- **Architecture:** encoder-only, **bidirectional** (`is_causal=False`), pre-norm
  RMSNorm, SwiGLU FFN, no biases, `F.scaled_dot_product_attention`, and a **2D
  positional embedding** (`col_emb[i % 9] + rank_emb[9 - i // 9]`, with index 90
  getting its own vector). There is no autoregressive masking — the whole board
  is visible at once.
- **Heads:** a **policy** head over the 2554-move `move-v1` vocabulary (masked to
  legal moves at decode time, per ADR-0004) and a scalar **value** head
  (`tanh`-bounded, `win_prob = (value + 1) / 2`). Value targets come from the game
  result and are **masked (127)** when unavailable, so BC on a result-less corpus
  still trains the policy cleanly.
- **Presets:** `m1-dev` (~22M, local iteration), `mid` (~144M), `1b` (~1.02B, the
  cloud target). One config knob (`BoardTransformerConfig.presets()`) spans all
  three so nothing but the preset name changes between laptop and cloud.
- **Device/dtype:** `resolve_device_dtype()` picks cuda→bf16, mps→fp16, cpu→fp32;
  `DONGFENG_FORCE_FP32=1` is the escape hatch for MPS fp16 instability.

This does **not** supersede ADR-0002. Both model families are legitimate flagship
policies behind the same Protocol; the autoregressive one keeps the sequence
framing, the board-state one is the scale-and-portability workhorse.

## Consequences

- **Pros:** fixed-length input is trivial to batch/shard/checkpoint across MPS and
  CUDA; a single preset knob scales 22M→1B with no code change; dense
  position→move supervision; the value head is populated by the same shards
  (result-labeled) with graceful masking when labels are absent; bidirectional
  attention sees the whole board in one pass (no KV cache, one forward per move).
- **Cons:** loses explicit move-history context (fine for Xiangqi policy; a known
  limitation for repetition-sensitive play, deferred to M5); adds a second model
  family to maintain; a 1B run needs ≫ the current ~628K-sample corpus to reach
  Chinchilla-optimal tokens, so 1B is architecture-verified (smoke) but not yet
  trained to convergence (see `docs/training-1b.md`).
- **Reversibility:** both families sit behind the `Engine` Protocol, so callers
  (arena, CLI, UCCI adapter, web UI) are indifferent to which one is loaded. If
  one paradigm clearly wins, the other can be retired without touching callers.
