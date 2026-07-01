# 3. External protocol: UCCI (serve) and Pikafish-UCI (teacher/opponent)

- **Status:** Accepted
- **Date:** 2026-07-01

## Context

Dong Feng must (a) be playable in existing Xiangqi GUIs and arenas, and (b) drive
a strong reference engine (Pikafish) as a teacher for distillation and as an
opponent for evals. Both are text-protocol integrations. The Xiangqi ecosystem has
two relevant conventions:

- **UCCI** (Universal Chinese Chess Interface) — the native Xiangqi engine
  protocol used by Chinese GUIs.
- **UCI (Xiangqi flavor)** — Pikafish, being Stockfish-derived, speaks a
  UCI-style dialect with Xiangqi FEN and coordinate moves, plus useful options
  (`UCI_ShowWDL`, `MultiPV`) that produce distillation labels.

A coordinate hazard exists: ICCS move strings use ranks `0-9`, while some
back-ends (Fairy-Stockfish / pyffish) use ranks `1-10`. Pikafish's own UCI uses
the ICCS `0-9` convention, but the adapter must be explicit to avoid off-by-one
move bugs.

## Decision

- **Serve** Dong Feng over **UCCI** (`dfc ucci`, `dongfeng.serve.ucci`), adapting
  the in-process `Engine` Protocol to the UCCI text protocol on stdin/stdout.
- **Drive Pikafish** over its **UCI dialect** in `engines/pikafish_engine.py`,
  exposing it as an ordinary `Engine`. Use `UCI_ShowWDL=true` and `MultiPV=k` to
  extract per-move WDL/centipawn labels for distillation (M4).
- **Normalize all coordinates to ICCS (`a-i`, `0-9`) at the protocol boundary.**
  Any back-end that emits `1-10` ranks (e.g. Fairy-Stockfish/pyffish, if used for
  adjudication) is remapped in the adapter, never leaked inward.
- Both directions sit in the same **Tier 2** layer (see DESIGN §3); the in-process
  `Engine` Protocol (Tier 1) is the source of truth, and gRPC/HTTP (Tier 3) is a
  future wrapper.

## Consequences

- Dong Feng plays in any UCCI GUI/arena and can be benchmarked against Pikafish
  with no bespoke glue per tool.
- Distillation labels come "for free" from Pikafish's standard UCI options.
- The coordinate-normalization rule is a hard invariant: every move crossing the
  protocol boundary is validated/converted to ICCS, tested in
  `tests/test_ucci_adapter.py`.
- Reference details are captured in `docs/protocol/UCCI.md` and
  `docs/protocol/pikafish-uci.md` so the adapter has an authoritative spec to code
  against.
