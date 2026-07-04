# Training the 1B Board-State Model — 5090 Cloud Runbook

This guide covers how to launch `dfc train-board --preset 1b` on a cloud GPU
(e.g. RTX 5090 / A100 / H100) and pull the metrics back for the local Training
UI. See `ROADMAP.md` and `docs/plans/board-1b.md` for design context.

---

## 1. Why not train 1B right now

Training 1B parameters to convergence requires roughly 20B tokens
([Chinchilla scaling](https://arxiv.org/abs/2203.02155)). The current corpus
after WP1 ingestion of the full `data/vietcotuong` archive contains approximately
764M board-state samples. Assuming ~1 sample ≈ 1 "effective token", the corpus is
**26× smaller** than the Chinchilla optimum for 1B parameters.

The recommended path is:

1. Finish WP8 + M4 (Pikafish distillation targets, self-play data generation).
2. Accumulate ≥ 20B samples before a full 1B run.
3. Until then, use `--preset mid` (≈ 135M params) for experimentation.

The runbook below documents the steps for when the corpus is ready.
Use `--smoke --preset 1b` (see §6) to verify the architecture at any time without data.

---

## 2. VRAM budget — 1b preset

| Component | Memory | Notes |
|-----------|--------|-------|
| bf16 weights | ~2 GB | 1.024 B params × 2 bytes |
| bf16 gradients | ~2 GB | same size as weights |
| AdamW fp32 moments (m + v) | ~8 GB | 2 × 1.024 B × 4 bytes |
| **AdamW total** | **~12 GB** | before activations |
| Activations (batch 512, grad-ckpt off) | ~8–15 GB | depends on seq/batch |
| Activations (batch 512, **grad-ckpt on**) | ~2–4 GB | recomputes per-block |
| **Recommended minimum VRAM (grad-ckpt on)** | **~16 GB** | A100-40 GB leaves headroom |
| RTX 5090 (32 GB) | fits comfortably | grad-ckpt on, batch 512–2048 |

### With `--optim adam8bit` (bitsandbytes)

`adam8bit` quantises the optimizer moments to 8-bit integers, saving ~6 GB:

| Component | AdamW | adam8bit |
|-----------|-------|----------|
| Optimizer moments | ~8 GB | ~2 GB |
| **Total (grad-ckpt on, batch 512)** | ~16 GB | ~10 GB |

This lets you run larger batches or skip grad-checkpoint on a 16 GB card.

---

## 3. Suggested hyperparameters — 1b preset

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| `--lr` | `1.5e-4` | Lower than mid/m1-dev; large models are sensitive |
| `--batch` | `512` | Per-GPU; increase to 2048 with grad-accumulation (future flag) |
| `--warmup` | `2000` | ~0.5% of total steps |
| `--steps` | `400000` | Tune when corpus ≥ 20B |
| `--value-weight` | `0.5` | Policy CE + 0.5 × value MSE |
| `--grad-checkpoint` | on | Required for ≤ 40 GB VRAM |
| `--optim` | `adam8bit` | Recommended on CUDA; falls back to AdamW if bnb missing |
| `--device` | `auto` or `cuda` | `auto` picks CUDA > MPS > CPU |

For single-GPU runs that cannot fit the target batch size, use a smaller
`--batch` (e.g. 128) and plan to add gradient accumulation in a future WP.
The learning rate should be scaled linearly with effective batch size (e.g. halve
lr if halving batch).

---

## 4. Cloud checklist

### 4.1 Prepare the environment

```bash
# On the cloud VM / container
git clone https://github.com/chuan99nd/dong-feng-chess-bot
cd dong-feng-chess-bot
uv sync --extra model          # installs torch + all model deps
uv pip install bitsandbytes    # optional — enables --optim adam8bit
```

### 4.2 Rsync board shards to the cloud

```bash
# From your local machine
rsync -avz --progress \
  data/vietcotuong-board/ \
  cloud-user@<host>:~/dong-feng-chess-bot/data/vietcotuong-board/
```

### 4.3 Verify architecture before launching (smoke test)

```bash
uv run dfc train-board --smoke --preset 1b
# Expected output (on any device):
#   Smoke test preset='1b'  params=1,023,905,664  device=cuda  dtype=bfloat16
#   step 0: loss=...
#   step 1: loss=...
#   Smoke OK  params=1,023,905,664
```

### 4.4 Launch training in a tmux session

```bash
tmux new -s train1b

uv run dfc train-board \
  --preset 1b \
  --data  data/vietcotuong-board \
  --out   runs/1b-v1 \
  --id    1b-v1 \
  --lr    1.5e-4 \
  --batch 512 \
  --warmup 2000 \
  --steps 400000 \
  --grad-checkpoint \
  --optim adam8bit \
  --device auto

# Detach: Ctrl-b d
```

To resume (future): add `--resume runs/1b-v1/ckpt.pt` once that flag is
implemented (tracked in backlog).

### 4.5 Monitor progress

```bash
# Tail metrics live
tail -f runs/1b-v1/metrics.jsonl | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    if r['split'] == 'val':
        print(f\"step={r['step']:6d}  val_policy={r['policy_loss']:.4f}  top1={r.get('top1',0):.3f}\")
"
```

### 4.6 Rsync metrics back for the local Training UI

The web UI's Training tab polls `runs/<id>/metrics.jsonl` (see
`src/dongfeng/serve/webplay.py`). Point it at the cloud run by setting
`DONGFENG_RUNS_DIR`:

```bash
# Sync metrics back periodically (e.g. in a cron or another tmux pane)
rsync -avz cloud-user@<host>:~/dong-feng-chess-bot/runs/1b-v1/ \
  runs/1b-v1/

# Then start the local web UI pointed at the same runs dir
DONGFENG_RUNS_DIR=$(pwd)/runs uv run dfc web --engine random --no-open
# Open http://127.0.0.1:8000 → Training tab → select run "1b-v1"
```

---

## 5. Device / dtype matrix

| Device | dtype | Autocast | Notes |
|--------|-------|----------|-------|
| CUDA | bf16 | yes | Recommended; stable, efficient |
| MPS (Apple Silicon) | fp16 | yes | May produce NaN if model/LR is aggressive |
| MPS + `DONGFENG_FORCE_FP32=1` | fp32 | no | Escape hatch for fp16 NaN instability |
| CPU | fp32 | no | Smoke/dev only; too slow for real training |

### MPS fp16 NaN escape hatch

If you see val loss become NaN during MPS training, set the environment variable
to force fp32 (disables autocast entirely):

```bash
DONGFENG_FORCE_FP32=1 uv run dfc train-board --preset m1-dev --data ... --out ...
```

The NaN guard in `bc_train_board` marks the run as `status: "failed"` and raises
`ValueError` if loss becomes non-finite, so failed runs are always clearly
identified in `run.json`.

---

## 6. Local smoke test results (2026-07-03, Apple M-series MPS)

Command:

```
uv run dfc train-board --smoke --preset 1b
uv run dfc train-board --smoke --preset m1-dev
```

Results:

| Preset | Params | Device | dtype | Step 0 loss | Step 1 loss | Outcome |
|--------|--------|--------|-------|-------------|-------------|---------|
| `1b` | 1,023,905,664 | mps | float16 | 7.6866 | 8.6577 | **Smoke OK** (2 steps completed) |
| `m1-dev` | 22,277,088 | mps | float16 | 7.6922 | 7.9178 | **Smoke OK** (2 steps completed) |

The 1B model allocated and ran 2 forward+backward passes on MPS without OOM.
Peak MPS memory was not reported (PyTorch does not expose `mps.max_memory_allocated`
in the same way as CUDA). Both presets exit with code 0.

---

## 7. See also

- `docs/plans/board-1b.md` — full milestone plan (§1.2 device/dtype, §5 risks)
- `ROADMAP.md` — M3.5 milestone status
- `src/dongfeng/training/board_loop.py` — `resolve_device_dtype`, `bc_train_board`
- `src/dongfeng/model/board_transformer.py` — `BoardTransformerConfig.presets()`
