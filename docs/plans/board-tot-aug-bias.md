# Plan — Augmentation, Per-head 2D bias (additive), MCTS/ToT

Three independent work packets extending the board-state flagship (M3.5).
Recommended order by ROI/risk: **WP-AUG → WP-BIAS → WP-MCTS**. They touch
disjoint files and can also run in parallel.

Pinned facts (verified in code — all agents must match):

- **Board seq layout** (`board-v1`, length 91): index `i` in `0..89` →
  `file = i % 9` (a..i = 0..8), `iccs_rank = 9 - i // 9` (FEN stores rank 9
  first). Index `90` = side-to-move token.
- **Token ids** (`BoardTokenizer`, vocab 21): empty=4; red pieces `K A B N R C P`
  = `5..11`; black `k a b n r c p` = `12..18` (black = red + 7); side_red=19,
  side_black=20; specials 0..3.
- **Move ids** (`move-v1`, vocab 2554): id `m ≥ 4` decodes to
  `MoveTokenizer()._moves[m - 4]`, a 4-char ICCS string `f0 r0 f1 r1`
  (`FILES="abcdefghi"`, `RANKS="0123456789"`). Special ids `0..3` are pass-through.
- **Value** target is from the **side-to-move's** perspective
  (`_RESULT_TURN_TO_VALUE`): +1 win / −1 loss / 0 draw / 127 mask.
- **Coordinate transforms** (ICCS `f∈0..8, r∈0..9`):
  - Mirror (left–right, a true Xiangqi symmetry): `(f, r) → (8-f, r)`.
  - Rotate-180 + swap colors (equivalent position for the other side):
    `(f, r) → (8-f, 9-r)` + swap piece colors + flip side token.
  - **Both preserve the value target** (value is mover-relative).

---

## WP-AUG — Mirror + color augmentation in `board_dataset.py`

**Goal:** cheap data multiplication, contained entirely in `data/board_dataset.py`
(+ CLI flags + tests). No model or engine changes. No inference-side change
(we augment both colors instead of canonicalizing, so the model keeps seeing
both side-to-move values — the engine needs no flip logic).

### Precomputed permutations (module-level, built once)

1. `POS_MIRROR: np.ndarray[91]` — `POS_MIRROR[i] = (i//9)*9 + (8 - i%9)` for
   `i<90`; `POS_MIRROR[90]=90`. Applied as `board[POS_MIRROR]` (gather). Token
   **values unchanged** under mirror.
2. `POS_ROT180: np.ndarray[91]` — `POS_ROT180[i] = 89 - i` for `i<90`;
   `[90]=90`. After gather, **swap colors** on the 90 board tokens
   (`5..11 ↔ 12..18`, empty/others unchanged) and **flip the side token**
   (19↔20). Vectorize color-swap with a `COLOR_SWAP: np.ndarray[21]` LUT.
3. `MOVE_MIRROR: np.ndarray[2554]`, `MOVE_ROT180: np.ndarray[2554]` — built once
   from `MoveTokenizer`: for each id `m ≥ 4`, decode to ICCS, transform both
   squares, re-encode, store the new id. Special ids map to themselves. Assert
   every transformed move re-encodes to a valid id (symmetry guarantees it).

### Build changes

- `build_board_shards(..., mirror: bool = True, color_augment: bool = True)`.
- For each collected per-ply sample `(board, move, value)` emit:
  - the original,
  - if `mirror`: `(board[POS_MIRROR], MOVE_MIRROR[move], value)`,
  - if `color_augment`: `(colorswap(board[POS_ROT180]), MOVE_ROT180[move], value)`,
  - if both: also the mirror-of-color-swapped variant (×4 total).
- Keep per-game grouping (augment each game's plies together) so the
  train/val split in `board_loop.py` never leaks a position and its mirror
  across the split — **augment inside the game loop before extending buffers.**
- `board_meta.json`: keep `schema: "board-ds-v1"` (backward compatible), add
  `"augment": {"mirror": bool, "color": bool, "factor": int}` and keep
  `num_samples` = the **post-augmentation** count.

### CLI

- `dfc data ingest-board` gains `--mirror/--no-mirror` (default on) and
  `--color-augment/--no-color-augment` (default on). Plumb into
  `build_board_shards`.

### Tests (`tests/test_board_dataset.py`, extend)

- `POS_MIRROR` and `POS_ROT180` are involutions (`P[P] == identity`).
- Mirroring a known hand-built position gives the expected file-flipped board;
  decode via `BoardTokenizer` round-trips to a legal FEN.
- `MOVE_MIRROR[MOVE_MIRROR[m]] == m`; a specific move (e.g. `h2e2`) mirrors to
  the geometrically correct square.
- Color-swap LUT: `COLOR_SWAP[COLOR_SWAP[t]] == t`; red↔black mapping correct;
  side token flips.
- `build_board_shards` with both flags yields `factor×` the sample count of
  flags-off; value array is unchanged (element-wise) across the 4 variants.
- Every augmented board still parses as a legal FEN and the augmented move is
  legal in it (spot-check a handful via `core.board`).

**Note honestly in the plan output:** color-augment overlaps with signal the
corpus already contains (real games alternate colors), so its marginal value is
smaller than mirror's. Mirror ×2 is the clean, high-value win; color-augment is
optional headroom. Re-ingest + retrain `m1-dev` after landing to confirm val
top1 improves.

---

## WP-BIAS — Additive per-head 2D relative bias (**extra heads**, not replacement)

**Goal:** add `n_bias_head` **new** attention heads that each carry a learnable
2D relative-position bias, on top of the existing `n_head` content heads. The
content heads keep their `head_dim` unchanged (no capacity is taken away).

### Config (`BoardTransformerConfig`)

- New field `n_bias_head: int = 0` (default 0 → **byte-identical** to today).
- `head_dim = d_model // n_head` (based on content heads only — unchanged).
- `total_heads = n_head + n_bias_head`; `inner_dim = total_heads * head_dim`
  (`= d_model + n_bias_head * head_dim`).
- Preset suggestions to ablate: `m1-dev` → 2, `mid` → 4, `1b` → 4. Keep as a
  field so runs can vary it.

### Attention (`_BidirectionalAttention`)

- `qkv = nn.Linear(d_model, 3 * inner_dim, bias=False)`.
- `out_proj = nn.Linear(inner_dim, d_model, bias=False)`.
- `rel_bias = nn.Parameter(torch.zeros(n_bias_head, 17 * 19 + 1))` **(zero-init →
  new heads start as plain extra heads; they diverge only if training finds the
  geometry useful — "do no harm").** Per-layer (each block its own table).
- `rel_index` buffer `[91, 91]` (long), shared, built once and passed in / or a
  module-level cached tensor moved to device: bucket
  `= (Δfile+8)*19 + (Δrank+9)` for board–board pairs; any pair touching index 90
  → bucket `323` (the `+1` slot).
- forward: reshape to `[B, total_heads, 91, head_dim]`; build
  `attn_bias [total_heads, 91, 91]` = `zeros` for the first `n_head` rows,
  `rel_bias[:, rel_index]` for the remaining `n_bias_head`; call
  `F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(0),
  is_causal=False)`. When `n_bias_head == 0`, skip the mask entirely (fast path
  = current behavior).

### Param cost (present these numbers; let the user tune / consider smaller
bias-head `head_dim` if too heavy)

Per layer extra = `4 * d_model * n_bias_head * head_dim + n_bias_head * 324`.

| Preset | d_model | head_dim | n_bias_head | Δ params | new total | Δ% |
|--------|---------|----------|-------------|----------|-----------|----|
| m1-dev | 384 | 64 | 2 | ~2.36M | ~24.6M | +10.6% |
| mid | 768 | 64 | 4 | ~? (compute) | — | ~+? |
| 1b | 1536 | 128 | 4 | ~113M | ~1.14B | +11% |

Cost-control option to document: give bias heads a smaller `head_dim`
(e.g. 32) since a geometric prior needs little capacity.

### Diversity measurement helper

- Add a small util (script under `scripts/` or a method) that, given a trained
  checkpoint + a val batch, reports per layer: (a) mean pairwise cosine
  similarity between heads' attention maps (lower = more diverse), (b)
  `‖rel_bias[h]‖` per bias head (which heads actually *used* the geometry).
  This is the evidence for "heads no longer redundant".

### Tests (`tests/test_board_model.py`, extend)

- `n_bias_head=0` → param count **exactly equals** the current documented number
  (regression guard for backward compat).
- `n_bias_head=k` → param count matches the formula; forward output shapes
  unchanged (`policy [B,2554]`, `value [B]`); output finite.
- `rel_bias` exists with shape `[n_bias_head, 324]`, is zero at init, and
  receives non-None grad after a backward.
- `rel_index` values in `0..323`; involution-free sanity (diagonal bucket =
  `8*19+9 = 161`).
- Instantiate `1b` with `n_bias_head=4` on `meta` device.
- Re-run `dfc train-board --smoke --preset m1-dev` and `--preset 1b`; update the
  measured param counts + smoke numbers in `docs/training-1b.md` and the model
  docstring.

### Ablation (cheap, on `m1-dev`, ~7 min/run, same seed, ~2000 steps)

| Run | Config | Hypothesis |
|-----|--------|-----------|
| A | `n_bias_head=0` | baseline |
| B | `n_bias_head=2` | do-no-harm + gain? |
| C | `n_bias_head=4` | more bias heads help? |
| D | `n_bias_head=2`, bias only in layers 0..5 (taper) | early-layer geometry is where it pays |

Compare val `top1` + the diversity metric. Decide `mid`/`1b` split from results —
**do not commit 1b blind.**

**MPS caveat:** float `attn_mask` in SDPA may drop off the fused MPS kernel
(slower, still correct). Verify `--smoke` passes on MPS after the change.

---

## WP-MCTS — Tree of Thought via PUCT search around the board model

**Goal:** an `Engine`-conformant MCTS wrapper that turns the single-forward-pass
policy into a search (ToT). New file `src/dongfeng/inference/mcts_board.py`;
reuses the model + legal-masking from `BoardTransformerEngine`.

**Honest gating:** the value head is currently trained on an all-masked corpus
(no result labels), so leaf value ≈ noise. Provide `value_mode` and document
that real strength needs M4 distillation labels. Implement now, shine after M4.

### Algorithm (AlphaZero-style PUCT)

- **Node:** stores `N` (visits), `W` (value sum), `Q = W/N`, `P` (prior),
  `children: dict[Move, Node]`, and terminal flag. Board state via
  `core.board` push/pop along the descent path (clone the root once).
- **Select:** from root descend by
  `argmax_a  Q(a) + c_puct · P(a) · √(ΣN) / (1 + N(a))`.
- **Expand:** at an unexpanded leaf, one model forward → policy logits masked to
  legal moves (reuse the `move-v1` id→logit lookup from `board_engine.py`),
  softmax → child priors `P`; leaf value `v` per `value_mode`.
- **Backup:** propagate `v` up the path, **negating each ply** (zero-sum,
  two-player).
- **Terminal:** `board.is_game_over()` / `result()`. **Xiangqi: no legal moves =
  LOSS** for the side to move → leaf value `-1` from that node's perspective
  (never a draw). Repetition/perpetual **not** handled (deferred to M5) — note
  it in the docstring.
- **Move choice:** `bestmove` = max-visit child; `analyze` returns `Analysis`
  with `ScoredMove` sorted by visit count, `win_prob` from root `Q`,
  `score_cp` optional.

### `value_mode` (set_option)

- `"head"` — use the value head (best once M4 labels exist).
- `"rollout"` — play to terminal with the policy (or random) for a leaf estimate
  (works today without a trained value head).
- `"zero"` — priors + terminals only (pure policy-guided tree).

### Engine Protocol surface

- `id`, `new_game`, `set_position(fen)`, `analyze(limits)`, `bestmove(limits)`,
  `set_option(name, value)`, `stop()`.
- `SearchLimits`: `nodes` → `n_simulations`; `movetime_ms` → wall-clock budget
  (loop checks `time.monotonic()`); `stop()` sets a cooperative flag checked in
  the sim loop.
- Options: `c_puct` (default ~2.0), `n_simulations` (default ~200),
  `temperature`, `dirichlet` (root noise for self-play).

### CLI / wiring

- `dfc eval arena --engine-kind board-mcts` (or a `--mcts` flag + `--sims`,
  `--c-puct` on the existing board kind). Optionally `dfc web --engine board-mcts`.

### Tests (`tests/test_mcts_board.py`)

- `run_conformance(factory) == []`.
- Determinism: fixed seed + `temperature=0` + fixed `n_simulations` → identical
  bestmove across two runs.
- Only legal moves are ever returned; terminal (mate) handled as a loss.
- `nodes` limit is honored (root visit count ≈ `n_simulations`).
- Arena: `board-mcts` vs `random` (≥ raw board engine) and vs raw `board`
  (expect ≥ once value labels exist; ~≥ now with `rollout`).

---

## Sequencing & commits

1. **WP-AUG** — land, re-ingest real corpus, retrain `m1-dev`, confirm val top1 ↑.
2. **WP-BIAS** — land config+attention, run the A/B/C/D ablation on `m1-dev`,
   pick `n_bias_head` per preset, update smoke/param docs.
3. **WP-MCTS** — land wrapper + arena wiring; document the M4 value dependency.

Each WP: full `pytest` + `ruff` + `pyright` green before committing; one commit
per WP on the current feature branch (`feat/board-state-flagship` or a new one).
Update `ROADMAP.md` (M3.5 addenda) and `CLAUDE.md` command table as surfaces
land.
