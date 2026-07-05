"""Generate Pikafish value/state labels for board shards (Phase 4).

For each position in a board dataset we ask the teacher (Pikafish) for a
mover-relative score and store it as a float32 ``values_eval_<stem>.bin`` shard
next to ``boards_<stem>.bin`` (schema board-ds-v2). Training then blends this
dense signal with the sparse terminal outcome for a much less noisy value head.

The run is:

* **Resumable** — a ``status.json`` cursor lets an interrupted overnight job
  continue where it stopped (labels already on disk are kept).
* **Deduped** — identical FENs (very common) are evaluated once and cached, so
  the teacher is called far fewer times than there are positions.
* **Monitored** — ``status.json`` carries progress / rate / ETA / cache-hit
  rate, surfaced by the web UI (``/api/labeling``) and printed by the CLI.

The teacher is injected as ``evaluator(fen, depth) -> PikafishEval`` so the
pipeline is testable without a Pikafish binary.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..engines.pikafish_engine import PikafishEval
from ..tokenizer.board_tokenizer import BoardTokenizer

Evaluator = Callable[[str, int], PikafishEval]


def eval_to_value(ev: PikafishEval, cp_scale: float) -> float:
    """Map a Pikafish score to a mover-relative value in [-1, 1].

    Mate → ±1 (sign = side to move). Centipawns → ``tanh(cp / cp_scale)``.
    No score → NaN (training masks it, same as the int8 127 sentinel).
    """
    if ev.mate is not None:
        return 1.0 if ev.mate > 0 else -1.0
    if ev.cp is not None:
        return math.tanh(ev.cp / cp_scale)
    return math.nan


def _status_path(status_dir: Path) -> Path:
    return status_dir / "status.json"


def label_eval(
    data_dir: str | Path,
    *,
    evaluator: Evaluator,
    depth: int = 18,
    cp_scale: float = 300.0,
    status_dir: str | Path,
    flush_every: int = 200,
    on_flush: Callable[[dict[str, Any]], None] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Label every position in *data_dir* with the teacher's value; resumable.

    Writes ``values_eval_<stem>.bin`` (float32, NaN = masked) into ``data_dir``
    and a live ``status.json`` into ``status_dir``. Returns the final status.
    """
    data_dir = Path(data_dir)
    status_dir = Path(status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    tok = BoardTokenizer()

    with open(data_dir / "board_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    stems: list[str] = meta["shards"]
    total = int(meta["num_samples"])
    started = now or datetime.now(UTC).isoformat()

    # Resume: a prior status.json tells us which shard/index to continue from.
    sp = _status_path(status_dir)
    resume_shard, resume_pos, done0 = 0, 0, 0
    if sp.exists():
        try:
            prev = json.loads(sp.read_text())
            resume_shard = int(prev.get("shard_idx", 0))
            resume_pos = int(prev.get("pos_in_shard", 0))
            done0 = int(prev.get("done", 0))
            started = prev.get("started", started)
        except Exception:
            pass

    cache: dict[str, float] = {}
    done = done0
    cache_hits = 0
    t_wall_start = _mono(now)

    def _write_status(status: str, shard_idx: int, pos: int) -> dict[str, Any]:
        elapsed = max(_mono(None) - t_wall_start, 1e-9) if now is None else 0.0
        did = done - done0
        rate = did / elapsed if elapsed > 0 else 0.0
        remaining = max(total - done, 0)
        eta = remaining / rate if rate > 0 else None
        payload = {
            "kind": "label-eval",
            "data_dir": str(data_dir),
            "depth": depth,
            "cp_scale": cp_scale,
            "total": total,
            "done": done,
            "unique_evaluated": len(cache),
            "cache_hits": cache_hits,
            "rate_pos_s": round(rate, 2),
            "eta_s": round(eta) if eta is not None else None,
            "shard_idx": shard_idx,
            "pos_in_shard": pos,
            "status": status,
            "started": started,
            "updated": datetime.now(UTC).isoformat() if now is None else now,
        }
        tmp = sp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(sp)
        if on_flush is not None:
            on_flush(payload)
        return payload

    last: dict[str, Any] = {}
    for si, stem in enumerate(stems):
        if si < resume_shard:
            continue
        boards = np.memmap(data_dir / f"boards_{stem}.bin", dtype=np.uint8, mode="r")
        n = boards.size // 91
        boards = boards[: n * 91].reshape(n, 91)

        out_path = data_dir / f"values_eval_{stem}.bin"
        if out_path.exists():
            arr = np.fromfile(out_path, dtype=np.float32)
            if arr.size != n:
                arr = np.full(n, np.nan, dtype=np.float32)
        else:
            arr = np.full(n, np.nan, dtype=np.float32)

        start = resume_pos if si == resume_shard else 0
        for i in range(start, n):
            fen = tok.decode([int(x) for x in boards[i]])
            if fen in cache:
                arr[i] = cache[fen]
                cache_hits += 1
            else:
                v = eval_to_value(evaluator(fen, depth), cp_scale)
                cache[fen] = v
                arr[i] = v
            done += 1
            if (i + 1) % flush_every == 0:
                arr.tofile(out_path)
                last = _write_status("running", si, i + 1)

        arr.tofile(out_path)
        last = _write_status("running", si, n)
        resume_pos = 0  # subsequent shards start at 0

    last = _write_status("done", len(stems), 0)
    return last


def _mono(now: str | None) -> float:
    """Monotonic clock for rate/ETA; deterministic 0 in tests (``now`` set)."""
    if now is not None:
        return 0.0
    import time  # noqa: PLC0415

    return time.perf_counter()
