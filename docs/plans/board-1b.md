# Plan: Board-state model, scalable to ~1B params (M3.5)

Status: **approved, not yet executed**. Owner: main session. Executors: cheaper
build agents (Sonnet-tier), one work packet (WP) per agent. Reviewer: main session.

## 0. Context & locked decisions

- **Goal**: replace the flat-move flagship with a **board-state** policy+value
  transformer that runs on **Mac M1 (MPS)** and **RTX 5090 (CUDA, rented cloud)**
  from one config-driven codebase, with presets from ~21M (M1 dev) to **~1B**.
- **Scope now**: full 1B architecture **code + smoke-test only**. Real 1B training
  waits for a data engine (self-play / Pikafish distill, M4/M5) — current corpus
  (8.4M positions ≈ 764M board tokens) under-feeds 1B (Chinchilla ~20B tokens).
- **Train small now**: the `m1-dev` preset (~21M) may be trained on the existing
  corpus to validate the pipeline end-to-end; that is a smoke/validation run, not
  the target model.
- **Tokenizer**: existing `BoardTokenizer` (`board-v1`, 91 tokens, vocab 21).
  Move ids: existing `MoveTokenizer` (`move-v1`, vocab 2554) for the policy head.
- **Value target (weak, available now)**: game result from the side-to-move's
  perspective (+1 win / −1 loss / 0 draw, mask when unknown). Real Pikafish
  targets replace these in M4 without changing the architecture.
- Everything below MUST follow repo conventions: `from __future__ import
  annotations`, full type hints, ruff (line 100) + pyright standard clean,
  artifacts under `data/`, `checkpoints/`, `runs/` (git-ignored), manifest is the
  artifacts index, new engines pass `run_conformance`.

## 1. Pinned contracts (all WPs must match these EXACTLY)

### 1.1 Model presets (`BoardTransformerConfig.presets`)

| preset   | d_model | n_layer | n_head | ffn_hidden (SwiGLU) | ~params |
|----------|---------|---------|--------|---------------------|---------|
| `m1-dev` | 384     | 12      | 6      | 1024                | ~21M    |
| `mid`    | 768     | 20      | 12     | 2048                | ~142M   |
| `1b`     | 1536    | 36      | 12     | 4096                | ~1.02B  |

Per-layer params ≈ 4·d² (attn, no bias) + 3·d·h (SwiGLU). Heads read the hidden
state at **index 90** (the side-to-move token).

### 1.2 Architecture (module `src/dongfeng/model/board_transformer.py`)

- Encoder-only, **bidirectional** (`is_causal=False`), pre-norm **RMSNorm**,
  **SwiGLU** FFN, **no biases** anywhere, `F.scaled_dot_product_attention`.
- **2D positional embedding**: for board index `i` in 0..89:
  `pos = col_emb[i % 9] + rank_emb[9 - i // 9]`; index 90 (side token) gets its
  own learned vector. No 1D pos table.
- `gradient_checkpointing: bool` config flag (`torch.utils.checkpoint` per block).
- **Policy head**: `Linear(d, 2554, bias=False)` on hidden[90].
- **Value head**: `RMSNorm → Linear(d, d//4) → SiLU → Linear(d//4, 1) → tanh`
  on hidden[90] → scalar in (−1, 1), from side-to-move's perspective.
- `save/load` mirroring `TransformerPolicy` (dict: `config`, `state_dict`,
  `extra`), plus `num_params()`.
- Dtype/device: helper `resolve_device_dtype() -> tuple[str, torch.dtype]`:
  cuda→bf16 autocast, mps→fp16 autocast for forward / fp32 master weights,
  cpu→fp32. Live in `training/board_loop.py`, reused by engine.

### 1.3 Board shard format (written by `build_board_shards`, dir e.g. `data/board_ds/`)

- `boards_XXXXX.bin` — uint8, shape N×91 flattened (board-v1 ids fit in uint8).
- `moves_XXXXX.bin` — uint16, N (move-v1 id of the move actually played).
- `values_XXXXX.bin` — int8, N: +1 side-to-move won, −1 lost, 0 draw, **127 = mask**.
- `board_meta.json` — `{"schema": "board-ds-v1", "num_samples", "num_games",
  "skipped_games", "tokenizer": "board-v1", "move_tokenizer": "move-v1",
  "shards": [names], "created": iso8601|null}`.
- One "sample" = one ply: board **before** the move + the move + final result.

### 1.4 Run directory + metrics (written by trainer, read by UI)

- `runs/<run_id>/run.json` — `{"id", "kind": "bc-board", "preset", "params",
  "device", "dtype", "data_dir", "started": iso, "finished": iso|null,
  "status": "running"|"done"|"failed", "config": {...trainer config...}}`.
- `runs/<run_id>/metrics.jsonl` — one JSON per line:
  `{"step": int, "split": "train"|"val", "loss": float, "policy_loss": float,
  "value_loss": float|null, "top1": float|null, "lr": float,
  "samples_per_s": float, "elapsed_s": float}`.
- Checkpoint: `runs/<run_id>/ckpt.pt` (best-by-val policy_loss).

### 1.5 Web API (served by `serve/webplay.py`)

- `GET /api/training` → `{"runs": [run.json fields + {"last_train": <last train
  line|null>, "last_val": <last val line|null>}]}`, newest first, from `runs/*/`.
- `GET /api/training?id=<run_id>` → `{"run": {...}, "metrics": [...]}` with
  metrics downsampled to ≤ 500 points per split (uniform stride).
- Runs root overridable via env `DONGFENG_RUNS_DIR` (default `runs`).

### 1.6 CLI (Typer, in `src/dongfeng/cli.py`)

- `dfc data ingest-board PATH --out DIR --id ID [--shard-size N]` — parse games
  → board shards (§1.3) → upsert dataset entry in manifest (`tokenizer:
  "board-v1"`, `notes` includes move_tokenizer).
- `dfc train-board --data DIR --out runs/<id> --id ID --preset m1-dev|mid|1b
  [--steps N --batch N --lr F --warmup N --device auto --seed N --value-weight F
  --grad-checkpoint/--no-grad-checkpoint --smoke]`.
  `--smoke`: build model from preset, print param count, run **2 forward+backward
  steps on synthetic random batches** (no data needed), print peak memory, exit 0.
- `dfc web` gains engine choice `board` (env `DONGFENG_BOARD_CKPT`), same for
  `dfc play/selfplay/eval arena --engine board`.

### 1.7 Loss

`total = policy_CE + value_weight * value_MSE` (default `--value-weight 0.5`);
value_MSE computed only where value target ≠ 127 (masked mean; if the whole batch
is masked, value term = 0). Policy CE over the full 2554 vocab (no train-time
masking); legal masking is inference-only.

## 2. Work packets

Each WP: one agent, must run `uv run ruff check src tests`, `uv run pyright src`,
and its listed tests before reporting. Do NOT touch files owned by another WP
except where listed under "may edit".

### WP1 — Board dataset pipeline  *(agent: sonnet; wave 1)*

- **Input**: `src/dongfeng/data/dataset.py` (`iter_samples`), `base.py` (`Sample`,
  `Game`), tokenizers, §1.3.
- **Output**: `src/dongfeng/data/board_dataset.py` with
  `build_board_shards(games, out_dir, *, shard_size=1_000_000, created=None) ->
  BoardBuildStats` and `load_board_arrays(data_dir) -> (boards uint8 [N,91],
  moves uint16 [N], values int8 [N])` (numpy memmap-friendly); export from
  `data/__init__.py`. Value target: from `Sample.turn` + `Game.result`
  (winner==turn → +1, loser → −1, DRAW → 0, ONGOING → 127).
- **Tests**: `tests/test_board_dataset.py` — build from 2 synthetic games; counts
  match; a random sample's board decodes (BoardTokenizer.decode) to the FEN of a
  replayed board at that ply; value signs correct for a RED_WIN game; meta json
  matches §1.3.
- **Success**: new tests + full suite pass; no torch import in this module.

### WP2 — Model: board transformer  *(agent: sonnet; wave 1)*

- **Input**: §1.1–1.2, existing `model/transformer.py` as style reference.
- **Output**: `src/dongfeng/model/board_transformer.py` (`BoardTransformerConfig`
  with `presets: dict[str, BoardTransformerConfig]` classmethod or module dict,
  `BoardTransformer` with `forward(boards)->(policy_logits[B,2554],
  value[B])`), export from `model/__init__.py`.
- **Tests**: `tests/test_board_model.py` — forward shapes; `num_params()` of each
  preset within ±10% of §1.1 table (assert 0.9e9 < params(`1b`) < 1.15e9 —
  **instantiate 1b on `meta` device** to keep the test light); save/load
  round-trip equality; grad-checkpoint on/off gives same loss (atol 1e-4, small
  preset); backward runs on cpu for `m1-dev`-shaped tiny config.
- **Success**: tests + lint/type clean. Torch only inside this module.

### WP3 — Training loop + metrics writer  *(agent: sonnet; wave 2, needs WP1+WP2)*

- **Input**: WP1 loader, WP2 model, §1.4, §1.7, existing `training/loop.py` style.
- **Output**: `src/dongfeng/training/board_loop.py`:
  `bc_train_board(config: BoardTrainConfig) -> Path` (dataclass config: data_dir,
  out_dir, preset, batch_size, lr, warmup, max_steps, value_weight, device,
  seed, grad_checkpoint, eval_every) — AdamW (fused on cuda when available),
  warmup+cosine, autocast per §1.2 dtype helper, grad clip 1.0, 98/2 split,
  best-by-val save, `run.json` + `metrics.jsonl` per §1.4 (write `status`
  transitions running→done/failed), top1 = argmax(policy)==target on val.
- **Tests**: `tests/test_board_training.py` — 20 steps on synthetic shards (use
  WP1 builder) on cpu: loss decreases vs step 0; metrics.jsonl lines parse and
  match §1.4 keys; run.json status ends "done"; ckpt loads via
  `BoardTransformer.load`.
- **Success**: tests + lint/type clean.

### WP4 — Board engine (Engine protocol)  *(agent: sonnet; wave 2, needs WP2)*

- **Input**: `protocol/engine.py`, `inference/transformer_engine.py` as template.
- **Output**: `src/dongfeng/inference/board_engine.py` — `BoardTransformerEngine
  (checkpoint: str | None, device="cpu")`: random-init `m1-dev`-small fallback
  when no checkpoint; each `bestmove/analyze`: current board FEN →
  BoardTokenizer → forward → **legal mask** over move-v1 ids → argmax or
  temperature/top-k sample (same options as TransformerEngine: Temperature,
  TopK, Seed, Checkpoint); `analyze` fills `policy_prob` AND `win_prob` (value
  head, mapped from tanh to [0,1] via (v+1)/2) on ScoredMoves.
- **Tests**: `tests/test_board_engine.py` — `run_conformance(...) == []`; legal
  bestmove from start; analyze sorted desc by policy_prob; win_prob in [0,1].
- **Success**: tests + lint/type clean.

### WP5 — CLI wiring  *(agent: sonnet; wave 3, needs WP1+WP3+WP4)*

- **Input**: §1.6, `cli.py`.
- **Output** (may edit `cli.py`, `serve/webplay.py` engine factory):
  `dfc data ingest-board`, `dfc train-board` (incl. `--smoke` per §1.6; smoke
  needs no data), engine `board` in `_make_engine` + webplay `_make_engine`
  (+ UI dropdown option), manifest upsert for board datasets & checkpoints
  (register `board-v1` tokenizer if missing).
- **Tests**: `tests/test_cli_board.py` using Typer runner or subprocess on a tmp
  manifest: ingest-board on the sample PGN from `tests/test_ingest.py` fixtures →
  train-board 10 steps m1-dev-tiny overrides → files exist per §1.3/1.4;
  `train-board --smoke --preset m1-dev` exits 0 and prints param count.
- **Success**: tests + full suite + lint/type clean; `dfc data stats` shows the
  board dataset.

### WP6 — Training dashboard in web UI  *(agent: sonnet; wave 2 — independent,
contracts pinned in §1.4/1.5)*

- **Input**: `serve/webplay.py`, §1.4, §1.5.
- **Output**: `/api/training` (list + detail w/ ≤500-pt downsample) in webplay;
  new UI panel/tab "Training": run selector, live line chart (train vs val loss,
  plain `<canvas>`, no external deps, poll 2s while `status=="running"` else
  stop), stat chips (preset, params, step, lr, top1, samples/s, status).
  Keep board play untouched.
- **Tests**: `tests/test_training_api.py` — synthetic `runs/` tree via
  `DONGFENG_RUNS_DIR` env: list endpoint shape, detail downsampling (5k lines →
  ≤500/split), missing id → error json; HTML contains the Training panel markup.
- **Success**: tests + lint/type clean; manual smoke: server starts, `/` still
  plays.

### WP7 — Device matrix, 1B smoke + docs  *(agent: sonnet; wave 3, needs WP3+WP5)*

- **Input**: everything merged.
- **Output**: verify/adjust dtype-device helper across cuda/mps/cpu paths;
  `docs/training-1b.md` — 5090 runbook: VRAM budget table (bf16 weights ~2GB,
  grads ~2GB, AdamW fp32 moments ~8GB → ~12GB + activations; batch/λ/lr
  suggestions: lr 1.5e-4, batch 512–2048, grad-checkpoint on), optional
  bitsandbytes 8-bit Adam (guarded import, cuda-only, `--optim adam8bit` flag on
  train-board), cloud checklist (uv sync --extra model, rsync shards, tmux, and
  how the metrics.jsonl can be rsynced back for the UI).
- **Local success (M1/CI)**: `dfc train-board --smoke --preset 1b` on cpu/mps
  either completes 2 steps or exits gracefully with a clear OOM message
  (documented); `--smoke --preset m1-dev` must pass everywhere.
- **5090 success (deferred, manual)**: documented commands only — not executed
  in this plan.

### WP8 — Final integration review  *(main session / opus; wave 4)*

- Run: full pytest, ruff, pyright; `dfc data ingest-board` on
  `data/vietcotuong/data/selected-games` → real shards; `dfc train-board --preset
  m1-dev --steps 300` on MPS (validation run, ~minutes); watch it in the new UI
  tab; `dfc eval arena --engine board` vs random 10 games (expect ≥ no losses);
  update `ROADMAP.md` (mark M3.5), `CLAUDE.md` (new commands), add
  `docs/adr/0005-board-state-flagship.md` (decision: board-state + value replaces
  flat as flagship; flat kept as baseline/history reference). Commit.

## 3. Dependency graph / execution waves

```
wave 1 (parallel): WP1, WP2, WP6*         (*WP6 builds against pinned §1.4/1.5)
wave 2 (parallel): WP3 (WP1+WP2), WP4 (WP2)
wave 3 (parallel): WP5 (WP1+WP3+WP4), WP7 (WP3+WP5 → serialize after WP5)
wave 4:            WP8 review (main session)
```

## 4. Out of scope / backlog

- Real 1B training (needs data engine), Pikafish distill targets (M4), MCTS,
  mirror augmentation (`cchess.iccs_mirror`, ×2 data), resume-from-checkpoint,
  W&B/etc. (metrics.jsonl is the single source), 32-piece-token variant.

## 5. Risks

- MPS fp16 autocast instability → fall back to fp32 on mps if val loss NaNs
  (trainer must detect NaN and fail the run with status="failed").
- uint8 board ids assume vocab ≤ 255 (board-v1 = 21, safe).
- Two engines in webplay sharing one process: load lazily, keep memory in check.
- Value from game result is noisy — expected; it's a placeholder for M4 targets.
```
