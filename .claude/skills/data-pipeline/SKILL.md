---
name: data-pipeline
description: Ingest Xiangqi game archives (XQF/CBR/CBL/PGN) into (FEN, move) training pairs, tokenize them, and report corpus stats. Use when building or inspecting the training corpus (milestone M1).
---

# data-pipeline

Use this to turn raw Xiangqi game archives into training-ready, tokenized data,
and to inspect what's in the corpus. This is a **milestone M1** feature; commands
print a "planned: M1" notice until then.

## Commands

```bash
# Ingest game archives into (FEN, next_move) pairs (via cchess)
uv run dfc data ingest --src <path-or-dir> --out <dataset-id>

# Tokenize an ingested dataset (FEN/ICCS -> token ids)
uv run dfc data tokenize --dataset <id>

# Report corpus stats WITHOUT reading the shards (cheap; from manifest.json)
uv run dfc data stats
```

## Sources & formats

- **XQF** (DPXQ native binary; obfuscated/XOR-keyed — parse via `cchess`, never
  hand-roll).
- **CBR / CBL** (CCBridge single-record / library).
- **PGN-for-Xiangqi** (what the big published corpora used).
- Priority: DPXQ (curated pro games) → optional TianTian/QQ scrapes for volume →
  ChessDB for position evals.

## Filtering (mirror the published recipes)

- Keep **winning-side moves** in decisive games; keep **both sides** in draws.
- Drop very short games and disconnect-terminated games.

## Reading corpus size — do NOT read `data/`

Datasets live under `data/` (large, git-ignored). Get counts/vocab from the
**index**, not the files:

- `uv run dfc data stats`, or
- MCP tools `dataset_stats` / `tokenizer_info` (`tools/mcp_server.py`).

## Notes

- Canonical representation: **FEN positions + ICCS moves** (see the protocol docs).
- WXF↔ICCS conversion (side-relative) is part of this milestone
  (`dongfeng.core.notation`).
- To train on the output, use the **train-run** skill.
