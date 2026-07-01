---
name: checkpoint-inspect
description: Inspect a Dong Feng model checkpoint's metadata (architecture, step, metrics, tokenizer) WITHOUT loading the weights. Use to compare checkpoints or pick one to serve — the token-efficient way.
---

# checkpoint-inspect

Use this to look up a checkpoint's metadata — architecture, training step, parent
run, eval metrics, tokenizer id — **without loading the weights**. This is the
token-efficient way to compare checkpoints or choose one to serve.

## Commands

```bash
# Metadata for one checkpoint (cheap; reads manifest.json, not the weights)
uv run dfc ckpt info <checkpoint-id>

# List known checkpoints
uv run dfc ckpt list
```

## Do NOT read the checkpoint files

Checkpoints live under `checkpoints/` and are large binary blobs (and
git-ignored). **Never `cat`/`Read`/grep them** — one file can cost more tokens
than the whole repo. Everything you need is indexed in `manifest.json`.

Preferred access, cheapest first:

1. MCP tool `checkpoint_info` (`tools/mcp_server.py`) — returns small JSON.
2. `uv run dfc ckpt info <id>` — same data, human-readable.
3. `manifest.json` directly (the `checkpoints` array) if you just want the index.

## What you get

- `id`, `path`, `created`
- architecture / model size, training `step`
- `dataset` and `run` it came from
- headline eval metrics (Elo, top-1 accuracy)
- `tokenizer` id (cross-check with MCP `tokenizer_info`)

## When to use

- Comparing candidate checkpoints before an eval or before serving.
- Picking a checkpoint to wrap as an engine (pair with **add-model**).
- Confirming which dataset/tokenizer a checkpoint was trained on.

## Notes

- Checkpoint metadata is written by training runs (M2+). Until then the
  `checkpoints` array in `manifest.json` is empty and these commands report that.
