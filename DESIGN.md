# DESIGN — Dong Feng architecture

Dong Feng is a Xiangqi (Chinese chess) engine built as a language model over the
game. This document explains the concept mapping, the module decomposition, the
interface tiers, the flagship model paradigm, the tech stack, and the
Xiangqi-specific rules the whole system must respect.

## 1. The LLM → Xiangqi concept mapping

The core bet: a game of Xiangqi is a *sequence*, a position is *context*, and a
move is a *token to predict*. Every stage of the modern LLM recipe has a direct
analogue.

| LLM concept                | Dong Feng realization                                                                                          | Module         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------- |
| **Tokenizer / vocabulary** | Encode a position (FEN) and moves (ICCS) into token ids. Board tokens for the 90 points + piece types; a move-token vocabulary over legal from→to pairs. | `dongfeng.data` |
| **Pretraining corpus**     | DPXQ / TianTian / QQ human game scores parsed (via `cchess`) into `(FEN, next_move)` pairs. Millions of pairs. | `dongfeng.data` |
| **Next-token prediction**  | **Next-move prediction** — behavior cloning (BC) of human/master play. The base pretraining objective.        | `dongfeng.model` |
| **SFT (supervised finetune)** | **Distillation** from a strong teacher (Pikafish): match its top-move / MultiPV policy and WDL value. Plus **Elo-conditioning**: prepend a strength token so the model can imitate a target rating. | `dongfeng.model` |
| **RLHF**                   | **Self-play + DPO**: generate games, form preference pairs (winning line ≻ losing line, or teacher-preferred ≻ dispreferred), optimize with Direct Preference Optimization. Optionally policy-gradient self-play. | `dongfeng.model` |
| **Constrained decoding**   | **Legal-move masking**: at every step the decoder samples only from `Board.legal_moves()`. Illegal moves have probability zero *by construction*, not by hope. | `dongfeng.engines` |
| **Inference server**       | **UCCI server** (`dfc ucci`) speaking the standard text protocol; a gRPC/HTTP server later for batched, remote inference. | `dongfeng.serve` |
| **Evals**                  | Elo arena vs. baselines/Pikafish, tactical/mate suites, teacher-agreement rate, legal-move rate. | `dongfeng.engines` (eval harness) + `manifest.json` |

Two properties fall out of this framing:

- **Legality is a decoding constraint, not a learned skill.** We never ship an
  illegal move because the mask comes from the authoritative rules backend. The
  model only has to be *good*, not *legal*.
- **Strength is conditioning.** Elo-conditioning lets one model span a range of
  playing strengths, useful for training curricula and for human-matched play.

## 2. Module decomposition

```
dongfeng
├── core/            rules-agnostic vocabulary + the board
│   ├── types.py       Color, GameResult, Move (ICCS 4-char)
│   ├── board.py       Board Protocol, LibBoard (cchess-backed), new_board()
│   ├── fen.py         STARTING_FEN, validate_fen, side_to_move
│   └── notation.py    ICCS parse/format; WXF<->ICCS (M1)
├── protocol/        the universal engine contract
│   ├── engine.py      Engine Protocol; EngineInfo, SearchLimits, ScoredMove, Analysis
│   └── conformance.py run_conformance(make_engine) -> [failure messages]
├── engines/         concrete Engine implementations              (M2+)
│   ├── random_engine.py   legal-random baseline (M0/M1 stand-in)
│   ├── neural_engine.py   TransformerEngine wrapping the model    (M3)
│   └── pikafish_engine.py UCI wrapper around Pikafish (teacher/opponent)
├── data/            corpus + tokenizer                            (M1)
│   ├── tokenizer.py   FEN/ICCS <-> token ids; vocab
│   ├── ingest.py      XQF/CBR/CBL/PGN -> (FEN, move) pairs via cchess
│   └── dataset.py     shard/stream training samples
├── model/           the neural net + training                    (M2+)
│   ├── transformer.py autoregressive policy (+ reserved value head)
│   ├── train.py       BC pretraining loop
│   ├── distill.py     Pikafish distillation, Elo-conditioning     (M4)
│   └── rl.py          self-play + DPO                             (M5)
├── serve/           protocol adapters                            (M3/M6)
│   ├── ucci.py        Engine <-> UCCI/UCI text protocol adapter
│   └── grpc.py        batched remote inference                    (M6)
└── cli.py           `dfc` Typer entrypoint
```

The dependency arrows point *inward*: `core` depends on nothing but `cchess`
(deferred); `protocol` depends only on `core`; everything else depends on those
two contracts and never on each other's internals. This is what lets any engine
be swapped for any other and lets the model layer evolve without touching the
protocol.

## 3. The three-tier interface

Dong Feng deliberately separates the *in-process contract*, the *text protocol*,
and *future network serving*. Each tier wraps the one below it.

### Tier 1 — the `Engine` Protocol (in-process)

`dongfeng.protocol.engine.Engine`. Python `Protocol`, so any object with the right
methods conforms — no base class required. This is what the arena, CLI, and UCCI
adapter call. Positions are FEN, moves are ICCS `Move`. `run_conformance()`
validates any engine factory against it, with legality checked against the real
rules backend.

Why a Protocol and not an ABC: third-party engines, mocks, and future rewrites all
satisfy it structurally; `run_conformance` is the behavioral gate.

### Tier 2 — UCCI + Pikafish-UCI (text protocol)

`dongfeng.serve.ucci` adapts a Tier-1 `Engine` to the **UCCI** (Universal Chinese
Chess Interface) text protocol on stdin/stdout, so any GUI or arena can drive Dong
Feng. The *same* tier, in the other direction, lets us drive **Pikafish** — which
speaks a UCI-flavored dialect — as a teacher/opponent (`engines/pikafish_engine.py`).

Key protocol facts (see `docs/protocol/UCCI.md`, `docs/protocol/pikafish-uci.md`):

- UCCI/Pikafish exchange positions as FEN and moves as coordinate strings.
- **Coordinate gotcha:** ICCS uses ranks `0-9`; some engines/back-ends (notably
  Fairy-Stockfish/pyffish) use ranks `1-10`. Pikafish's own UCI uses the ICCS
  `0-9` convention. The adapter normalizes to ICCS at the boundary.
- Pikafish gives value/policy targets via `UCI_ShowWDL` (per-mille W/D/L) and
  `MultiPV` — the raw material for distillation labels.

### Tier 3 — gRPC + HTTP (future, M6)

`dongfeng.serve.grpc` for batched, remote, multi-tenant inference: several arena
games or a UI backend hitting one GPU-resident model. Wraps the same Tier-1
`Engine`. This tier is a scaling/deployment concern and is intentionally deferred.

## 4. Flagship paradigm: autoregressive transformer + reserved distill head

The chosen flagship model is an **autoregressive transformer policy** over the
move sequence, mirroring the LLM framing end-to-end.

- **Input.** Board state encoded by the tokenizer (piece-placement tokens for the
  90 points, side-to-move, optional move history, optional Elo-conditioning
  token).
- **Backbone.** A standard decoder-style transformer (pre-norm, RoPE or learned
  positional encoding, GQA optional). Size scales across milestones (M2 small →
  M4 scaled).
- **Primary head — policy.** A distribution over the move vocabulary; at decode
  time it is **masked to legal moves** and argmax/sampled. This is the
  next-move-prediction objective for BC pretraining and the policy target for
  distillation and DPO.
- **Reserved head — action-value (distill).** A value/WDL head trained from
  Pikafish's per-mille WDL and centipawn scores. Reserved from day one in the
  architecture even though it is populated at M4, so the interface (`ScoredMove`
  carries `score_cp`, `win_prob`, `policy_prob`) is stable now. This gives
  MCTS-free, single-forward-pass value estimates ("grandmaster without search"
  style) and enables optional shallow search later without a redesign.

Why autoregressive-first (see `docs/adr/0002-autoregressive-flagship.md`):
maximal reuse of the LLM toolchain (tokenizer, transformer, SFT, DPO, KV-cache
decode), a single-forward-pass move for fast play, and a clean path to
value-augmentation without committing to a search engine up front. The `Engine`
Protocol keeps a search-based variant a drop-in if we ever want one.

## 5. Tech stack

| Concern              | Choice                                    | Notes                                                        |
| -------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| Language             | Python 3.11+                              | Full type hints; `pyright` standard mode.                   |
| Packaging / tasks    | `uv` + `hatchling`                        | `uv sync`, `uv run`; wheel packages `src/dongfeng`.         |
| CLI                  | `typer` + `rich`                          | `dfc` entrypoint.                                            |
| Rules backend        | `cchess` (walker8088, PyPI, LGPL-3.0)     | Wrapped behind `core.board`; pseudo-legal + self-check filter. |
| Lint / format        | `ruff` (`E,F,I,UP,B,SIM`, len 100)        | Enforced by editing hooks.                                  |
| Types                | `pyright`                                  | CI gate.                                                     |
| Tests                | `pytest`                                   | Conformance + rules + adapter suites.                       |
| Model / training     | PyTorch (M2+)                             | Transformer, BC/distill/RL loops. Optional extra.           |
| Teacher engine       | Pikafish (GPL-3.0, separate process)      | Distillation labels + opponent; driven over UCI.            |
| Agent tooling        | MCP stdio server (`mcp` SDK, optional)    | Token-efficient queries over `manifest.json`.               |

Optional deps (`torch`, `mcp`) import lazily / are guarded so the core package
installs and imports without them.

## 6. Xiangqi-specific rules notes

The rules backend (`cchess`) enforces all of these; they are documented here so
that anyone touching the board layer, the tokenizer, or eval logic understands the
invariants. Xiangqi is **not** chess with different pieces.

- **Board is 9×10 (90 points).** Pieces sit on intersections, not in cells. Files
  `a-i`, ranks `0-9` (ICCS). FEN placement is written top rank (Black back rank,
  ICCS rank 9) first, bottom rank (Red back rank, ICCS rank 0) last.
- **Palace.** The General and Advisors are confined to the 3×3 palace; the General
  moves one point orthogonally, Advisors one point diagonally, both inside the
  palace only.
- **River.** The board is split by a river between ranks 4 and 5. The Elephant
  (`B`) may not cross it; a Pawn (`P`) gains sideways movement only after crossing.
- **Horse-leg (蹩马腿).** The Horse (`N`) moves one orthogonal + one diagonal, but
  is blocked if the orthogonal "leg" point is occupied. Not a chess knight.
- **Elephant-eye (塞象眼).** The Elephant moves two points diagonally but is blocked
  if the midpoint ("eye") is occupied.
- **Cannon screen (炮).** The Cannon moves like a Chariot but **captures only by
  jumping exactly one intervening piece** (the "screen"/"mount"). It cannot
  capture without a screen.
- **Flying general (对脸/白脸将).** The two Generals may not face each other along an
  open file with no piece between them; a move that would expose this is illegal.
  The board layer treats it exactly like a self-check.
- **No legal moves = LOSS.** A side with no legal move loses — checkmate *and*
  stalemate are losses for the side to move (unlike Western chess, where stalemate
  is a draw). `GameResult` and `Board.result()` reflect this.
- **No promotion.** No piece ever promotes; a coordinate move is always exactly 4
  characters. A 5th (promotion) character is invalid input.
- **Perpetual check / chasing (长将/长捉).** Endless checking or chasing is
  forbidden and is adjudicated against the offender. Pure-Python rules libs handle
  this only partially; for tournament-accurate adjudication we layer an engine
  (Pikafish / Fairy-Stockfish `is_optional_game_end`) on top. Repetition is
  otherwise tracked by hashing positions per ply. This is a known gap called out
  in the roadmap.

See `docs/protocol/xiangqi-fen.md` and `docs/protocol/iccs-notation.md` for the
exact FEN dialect and move-notation encodings, including the two piece-letter
conventions (`K A B N R C P` canonical vs. legacy `K A E R C H P`) and the
`r`/`w`/`b` side-to-move variants.
