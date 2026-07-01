---
name: train-run
description: Launch or resume Dong Feng training — behavior-cloning pretrain, Pikafish distillation, or self-play DPO — and track the run. Use when the user wants to train, fine-tune, or distill a model (milestones M2/M4/M5).
---

# train-run

Use this to train Dong Feng. Three stages map to the LLM recipe (see `DESIGN.md`):
**behavior cloning** (M2), **distillation + Elo-conditioning** (M4), and
**self-play DPO** (M5). Commands print a "planned: Mx" notice until their
milestone lands.

## Commands

```bash
# Behavior-cloning pretrain (next-move prediction) on a tokenized dataset  [M2]
uv run dfc train --dataset <id> --config <path> --out <checkpoint-id>

# Resume from a checkpoint
uv run dfc train --resume <checkpoint-id>

# Distill from Pikafish (policy = MultiPV softmax, value = WDL)             [M4]
uv run dfc distill --dataset <id> --teacher pikafish --out <checkpoint-id>

# Self-play + DPO on preference pairs                                        [M5]
uv run dfc rl dpo --base <checkpoint-id> --out <checkpoint-id>
```

## When to use

- Start a **BC pretrain** from a tokenized corpus (needs the **data-pipeline**
  skill first).
- **Distill** a stronger/controllable policy from Pikafish; see
  `docs/protocol/pikafish-uci.md` for the label recipe (depth ~25, GoodMove within
  100cp, WDL→win-prob, MultiPV→policy).
- Refine with **self-play DPO**.

## Tracking runs — do NOT read `runs/` or `checkpoints/`

Training writes to `runs/` and `checkpoints/` (large, git-ignored) and indexes
them in `manifest.json`. To check status/metrics/metadata:

- `uv run dfc eval last`, `uv run dfc ckpt info <id>`, or
- MCP tools `eval_last` / `checkpoint_info` (`tools/mcp_server.py`).

## Notes

- The architecture is an autoregressive transformer with a **reserved
  action-value head** populated at M4 (ADR-0002).
- After training, evaluate with the **eval-strength** skill and register the model
  as an engine with the **add-model** skill.
