# ROADMAP — Dong Feng milestones

Milestones are incremental and each ends in something runnable. The `Engine` and
`Board` Protocols (see [DESIGN.md](DESIGN.md)) are fixed early so every later
milestone is a drop-in behind a stable contract.

Legend: ✅ done · 🔨 in progress · ⬜ planned

---

## M0 — Skeleton + protocol + tooling  ✅ (this milestone)

The foundation everything else plugs into.

- ✅ `dongfeng.core`: `Color`, `GameResult`, `Move` (ICCS); `Board` Protocol,
  `LibBoard` (cchess-backed), `new_board()`; FEN helpers; notation helpers.
- ✅ `dongfeng.protocol`: `Engine` Protocol + value types; `run_conformance()`.
- ✅ AI-native tooling & docs: `README`, `CLAUDE.md`, `DESIGN`, this roadmap;
  ADRs; protocol reference docs (UCCI, Pikafish-UCI, FEN, ICCS); editing hooks
  (`.claude/settings.json`); agent skills; MCP query server; `manifest.json`
  artifacts index; CI (uv + ruff + pyright + pytest).
- ⬜ `dfc` CLI skeleton with `selfplay`, `ucci`, `protocol-check` wired to a
  legal-random baseline engine (stand-in until M2/M3).

**Exit:** `uv sync`, `uv run pytest`, `uv run dfc selfplay`, `uv run dfc ucci`
all work; `run_conformance` passes for the baseline engine; CI is green.

## M1 — Data + tokenizer  ✅

Turn game archives into training-ready token sequences.

- ✅ Ingest XQF/CBF/CBL/CBR/PGN/TXT via `cchess` → `Game` / `(FEN, move)` pairs
  (`data/ingest.py`: per-format parsers + `parse_file`/`iter_games_in` dispatch).
- Corpus sources: DPXQ (XQF), optional TianTian/QQ scrapes, ChessDB position evals.
- ✅ Filtering: `iter_samples(winning_side_only=…)` keeps the winning side's moves
  in decisive games (both sides in draws); illegal/corrupt tails are dropped.
- ✅ Tokenizers: `MoveTokenizer` (flat ICCS move index, 2554 vocab) and
  `BoardTokenizer` (per-point FEN grid, length-91) — both round-trip.
- ✅ `dataset.py`: `build_shards` writes `uint16` autoregressive shards +
  `dataset_meta.json`; stats recorded in `manifest.json`.
- ✅ `dfc data ingest|tokenize|stats`; WXF↔ICCS converter (`core.notation`).

**Exit:** ✅ a reproducible corpus with `dataset_stats` populated in the manifest;
`tokenizer_info` returns a real vocab; round-trip FEN/move encode↔decode tests
(`tests/test_tokenizer_*`, `test_notation`, `test_ingest`, `test_dataset`).

## M2 — Transformer + BC pretrain + eval  ✅

The base policy by behavior cloning.

- ✅ `model/transformer.py`: decoder-only autoregressive `TransformerPolicy`
  (pre-norm GPT, fused causal attention) + reserved value head (ADR-0002).
- ✅ `training/loop.py`: `bc_pretrain` — next-move cross-entropy over the M1
  `uint16` shards, AdamW + warmup/cosine LR, MPS/CUDA/CPU auto-device.
- ✅ Eval harness: `eval/accuracy.py` (top-1 move-match vs held-out games) and
  `eval/match.py` (engine arena + Elo). Legal-move rate is 100% by construction
  (core legal-masking in the neural engine).
- ✅ `inference/transformer_engine.py`: `TransformerEngine` implements the Engine
  Protocol with legal-masked decoding (passes `run_conformance`).
- ✅ `dfc train`, `dfc eval accuracy|arena`; checkpoints indexed in `manifest.json`.
- ✅ Trained on real data: 9,381 vietcotuong.com games (DhtmlXQ) → BC checkpoint.

**Exit:** ✅ a trained checkpoint that plays legal moves and beats the random
baseline; checkpoint indexed for `checkpoint_info`.

## M3 — TransformerEngine → UCCI → play  ⬜

Make the model a first-class, playable engine.

- `engines/neural_engine.py`: wrap the model as an `Engine` with **legal-move
  masked** decoding; pass `run_conformance`.
- `serve/ucci.py`: full UCCI adapter so the neural engine plays in any GUI/arena.
- `engines/pikafish_engine.py`: Pikafish UCI wrapper (opponent + future teacher).
- Arena runner for engine-vs-engine matches.

**Exit:** `dfc ucci` serves the neural engine to a real GUI; it plays a full legal
game vs. Pikafish through the adapter.

## M4 — Scale + distill / Elo  ⬜

Stronger and controllable.

- Distillation (`model/distill.py`): Pikafish teacher over UCI with
  `UCI_ShowWDL=true` + `MultiPV=k`; policy = softmaxed MultiPV, value = per-mille
  WDL; label recipe from the Xiangqi-R1 precedent (depth ~25 where affordable,
  GoodMove = within 100cp of best).
- Populate the reserved action-value head.
- Elo-conditioning: strength token so one model spans a rating range.
- Scale model + corpus; retune training.

**Exit:** measurable Elo gain over the M2 base; Elo-conditioning demonstrably
shifts playing strength; value head correlates with teacher WDL.

## M5 — RL / self-play / DPO  ⬜

Refine beyond imitation.

- Self-play game generation with the neural engine.
- Preference pairs (winning ≻ losing lines, teacher-preferred ≻ dispreferred).
- DPO (`model/rl.py`); optional policy-gradient self-play.
- Repetition/perpetual-check adjudication in self-play (hash-per-ply +
  engine-based `is_optional_game_end` for tournament accuracy — the known gap in
  pure-Python rules libs).

**Exit:** DPO/self-play checkpoint outperforms the distilled M4 model in the arena.

## M6 — gRPC serving + quantization  ⬜

Deployment and efficiency.

- `serve/grpc.py`: batched, remote, multi-tenant inference (Tier 3).
- HTTP endpoint for UIs; health/metrics.
- Quantization / KV-cache decode optimizations for low-latency play.

**Exit:** a remote server plays multiple concurrent arena games on one GPU;
quantized model within a small strength delta of full precision.

---

### Cross-cutting, always-on

- **Protocol conformance** for every new engine (`run_conformance`).
- **Token-efficient artifacts:** datasets/checkpoints/runs stay out of git and are
  queried via the MCP server / `dfc` over `manifest.json` — never read directly.
- **Docs & ADRs** updated when an architecture decision changes.
