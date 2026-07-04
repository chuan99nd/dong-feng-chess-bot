"""Local web UI to play a human against a Dong Feng engine (stdlib only).

A single :class:`GameSession` holds the board, move history, and the chosen engine.
The browser renders the board (SVG) and posts moves to a tiny JSON API:

* ``GET  /``                    -> the board page (HTML/CSS/JS, embedded below)
* ``GET  /api/state``           -> current game state (fen, legal moves, turn, result)
* ``POST /api/new``             -> reset the game (engine, human color, temperature)
* ``POST /api/move``            -> apply a human move, then the engine's reply
* ``POST /api/engine/shutdown`` -> unload the inference model (free memory for training)
* ``POST /api/engine/start``    -> reload the inference model after a shutdown
* ``POST /api/engine/reload``   -> hot-swap the engine to a different checkpoint
* ``GET  /api/checkpoints``     -> list runs with a saved ckpt.pt (for hot-reload)
* ``GET  /api/training``        -> list all training runs (newest first)
* ``GET  /api/training?id=X``   -> detail for run X with downsampled metrics

The engine runs server-side, so the neural :class:`~dongfeng.inference.transformer_engine.TransformerEngine`
(PyTorch) works exactly like the random baseline behind the same
:class:`~dongfeng.protocol.engine.Engine` contract.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..core import STARTING_FEN, new_board
from ..core.types import Color, Move
from ..protocol.engine import Engine, SearchLimits


def _make_engine(name: str, checkpoint: str | None) -> Engine:
    """Build an engine by name (``random`` / ``neural`` / ``board``) for the session."""
    if name == "neural":
        from ..inference.transformer_engine import TransformerEngine  # noqa: PLC0415

        return TransformerEngine(checkpoint or os.environ.get("DONGFENG_CKPT") or None)
    if name == "board":
        from ..inference.board_engine import BoardTransformerEngine  # noqa: PLC0415

        return BoardTransformerEngine(
            checkpoint=checkpoint or os.environ.get("DONGFENG_BOARD_CKPT") or None,
            device="auto",
        )
    from ..engines import RandomEngine  # noqa: PLC0415

    return RandomEngine()


def _free_torch_memory() -> None:
    """Best-effort release of cached GPU/MPS memory after dropping an engine.

    Frees the inference allocator's cache so a concurrent training process can
    reclaim device memory. No-op when torch (or a GPU backend) is absent.
    """
    try:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415
    except Exception:
        return
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class GameSession:
    """A single in-memory game: board + history + engine (human vs engine)."""

    def __init__(self, engine_name: str, checkpoint: str | None) -> None:
        self._checkpoint = checkpoint
        self._lock = threading.Lock()
        self.engine: Engine | None = None
        self.reset(engine_name, "red", 0.0)

    def reset(self, engine_name: str, human: str, temperature: float) -> dict[str, Any]:
        with self._lock:
            self.board = new_board(STARTING_FEN)
            self.history: list[Move] = []
            self.engine_name = (
                engine_name if engine_name in ("random", "neural", "board") else "random"
            )
            self.human = Color.RED if human == "red" else Color.BLACK
            self.temperature = temperature
            self.engine = _make_engine(self.engine_name, self._checkpoint)
            self.engine.new_game()
            self._configure_engine()
            self.last_move: list[str] | None = None
            # If the human plays Black, the engine (Red) opens.
            if self.human is Color.BLACK:
                self._engine_reply()
            return self._state()

    def _configure_engine(self) -> None:
        if self.engine is not None and self.engine_name in ("neural", "board"):
            self.engine.set_option("Temperature", str(self.temperature))

    def shutdown_engine(self) -> dict[str, Any]:
        """Unload the inference engine and free device memory (for training).

        The game state is preserved; moves are refused until the engine is
        started again via :meth:`start_engine`.
        """
        with self._lock:
            if self.engine is not None:
                with contextlib.suppress(Exception):
                    self.engine.stop()
                self.engine = None
                _free_torch_memory()
            return self._state()

    def start_engine(self) -> dict[str, Any]:
        """(Re)load the inference engine after a shutdown, preserving the game."""
        with self._lock:
            if self.engine is None:
                self.engine = _make_engine(self.engine_name, self._checkpoint)
                self.engine.new_game()
                self._configure_engine()
            return self._state()

    def reload_engine(self, checkpoint: str | None) -> dict[str, Any]:
        """Hot-swap the engine to a different checkpoint, preserving the game.

        Drops the current engine (freeing its device memory), then loads
        ``checkpoint`` (or the env default when ``None``) for the same engine
        kind. On failure the previous checkpoint is restored and the engine is
        left unloaded; the error is returned alongside the game state.
        """
        with self._lock:
            if self.engine is not None:
                with contextlib.suppress(Exception):
                    self.engine.stop()
                self.engine = None
                _free_torch_memory()
            prev = self._checkpoint
            self._checkpoint = checkpoint
            try:
                self.engine = _make_engine(self.engine_name, self._checkpoint)
                self.engine.new_game()
                self._configure_engine()
            except Exception as exc:  # keep the server alive on a bad checkpoint
                self._checkpoint = prev
                self.engine = None
                _free_torch_memory()
                return {"error": f"failed to load checkpoint: {exc}", "state": self._state()}
            return self._state()

    def _engine_reply(self) -> str | None:
        if self.engine is None:
            return None
        if self.board.is_game_over():
            return None
        self.engine.set_position(STARTING_FEN, list(self.history))
        move = self.engine.bestmove(SearchLimits(movetime_ms=200))
        self.board.push(move)
        self.history.append(move)
        self.last_move = [move.from_sq, move.to_sq]
        return move.iccs

    def human_move(self, frm: str, to: str) -> dict[str, Any]:
        with self._lock:
            if self.engine is None:
                return {"error": "engine is shut down", "state": self._state()}
            if self.board.turn is not self.human:
                return {"error": "not your turn", "state": self._state()}
            try:
                move = Move(frm, to)
            except ValueError as exc:
                return {"error": str(exc), "state": self._state()}
            if not self.board.is_legal(move):
                return {"error": "illegal move", "state": self._state()}
            self.board.push(move)
            self.history.append(move)
            self.last_move = [frm, to]
            engine_move = self._engine_reply()
            return {"error": None, "engine_move": engine_move, "state": self._state()}

    def undo(self) -> dict[str, Any]:
        """Take back the human's last move (and the engine's reply); back to human's turn."""
        with self._lock:
            # A human move exists only once history is long enough for the human's side.
            min_len = 1 if self.human is Color.RED else 2
            if len(self.history) < min_len:
                return self._state()  # nothing of the human's to take back
            self.board.pop()
            self.history.pop()
            # Keep popping engine plies until it is the human's turn again.
            while self.history and self.board.turn is not self.human:
                self.board.pop()
                self.history.pop()
            last = self.history[-1] if self.history else None
            self.last_move = [last.from_sq, last.to_sq] if last else None
            return self._state()

    def _state(self) -> dict[str, Any]:
        return {
            "fen": self.board.fen(),
            "turn": self.board.turn.value,
            "legal": [[m.from_sq, m.to_sq] for m in self.board.legal_moves()],
            "history": [m.iccs for m in self.history],
            "result": self.board.result().value,
            "in_check": self.board.is_check(),
            "human": self.human.value,
            "engine": self.engine_name,
            "engine_loaded": self.engine is not None,
            "checkpoint": self._checkpoint,
            "temperature": self.temperature,
            "last_move": self.last_move,
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state()


def _runs_root() -> Path:
    """Return the runs root directory, overridable via DONGFENG_RUNS_DIR."""
    return Path(os.environ.get("DONGFENG_RUNS_DIR", "runs"))


def _read_run_json(run_dir: Path) -> dict[str, Any] | None:
    """Read and parse runs/<id>/run.json; return None on any error."""
    try:
        with open(run_dir / "run.json") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return None


def _read_last_metrics(metrics_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read metrics.jsonl and return (last_train, last_val) or (None, None) on error."""
    last_train: dict[str, Any] | None = None
    last_val: dict[str, Any] | None = None
    try:
        with open(metrics_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                split = row.get("split")
                if split == "train":
                    last_train = row
                elif split == "val":
                    last_val = row
    except Exception:
        pass
    return last_train, last_val


def _list_runs() -> list[dict[str, Any]]:
    """List all runs from DONGFENG_RUNS_DIR, newest first (by started field / mtime)."""
    root = _runs_root()
    results: list[dict[str, Any]] = []
    if not root.is_dir():
        return results
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        data = _read_run_json(run_dir)
        if data is None:
            continue
        metrics_path = run_dir / "metrics.jsonl"
        last_train, last_val = _read_last_metrics(metrics_path)
        data["last_train"] = last_train
        data["last_val"] = last_val
        results.append(data)

    # Sort by "started" desc; fall back to mtime of run.json for stability.
    def _sort_key(r: dict[str, Any]) -> str:
        started = r.get("started") or ""
        if not started:
            rj = _runs_root() / str(r.get("id", "")) / "run.json"
            try:
                started = str(rj.stat().st_mtime)
            except Exception:
                started = ""
        return started

    results.sort(key=_sort_key, reverse=True)
    return results


def _list_models() -> list[dict[str, Any]]:
    """Group runs by architecture hash — one entry per distinct model design.

    Each model aggregates its runs (newest first) and surfaces the best
    validation top-1 achieved across them, so the web UI can offer a
    "choose a model, then a run" drill-down keyed by a stable arch hash.
    """
    runs = _list_runs()
    models: dict[str, dict[str, Any]] = {}
    for r in runs:
        h = r.get("arch_hash") or "unknown"
        m = models.get(h)
        if m is None:
            arch = r.get("arch") or {}
            m = {
                "arch_hash": h,
                "preset": r.get("preset"),
                "params": r.get("params"),
                "arch": arch,
                "arch_summary": (
                    f"d{arch.get('d_model')}·L{arch.get('n_layer')}·"
                    f"h{arch.get('n_head')}"
                    + (f"+{arch.get('n_bias_head')}b" if arch.get("n_bias_head") else "")
                )
                if arch
                else None,
                "runs": [],
                "best_top1": None,
                "best_run_id": None,
                "latest_started": r.get("started"),
            }
            models[h] = m
        lv = r.get("last_val") or {}
        top1 = lv.get("top1")
        m["runs"].append(
            {
                "id": r.get("id"),
                "status": r.get("status"),
                "started": r.get("started"),
                "step": (r.get("last_train") or {}).get("step"),
                "top1": top1,
            }
        )
        if top1 is not None and (m["best_top1"] is None or top1 > m["best_top1"]):
            m["best_top1"] = top1
            m["best_run_id"] = r.get("id")

    result = list(models.values())
    result.sort(key=lambda m: m.get("latest_started") or "", reverse=True)
    return result


def _list_checkpoints() -> list[dict[str, Any]]:
    """List runs that have a saved ``ckpt.pt`` (newest first) for hot-reload.

    Each entry carries just enough to label the checkpoint in the UI; the
    ``path`` is what :meth:`GameSession.reload_engine` loads.
    """
    out: list[dict[str, Any]] = []
    for r in _list_runs():
        run_id = str(r.get("id") or "")
        if not run_id:
            continue
        path = _runs_root() / run_id / "ckpt.pt"
        if not path.is_file():
            continue
        lv = r.get("last_val") or {}
        out.append(
            {
                "id": run_id,
                "path": str(path),
                "preset": r.get("preset"),
                "params": r.get("params"),
                "arch_hash": r.get("arch_hash"),
                "status": r.get("status"),
                "top1": lv.get("top1"),
                "step": (r.get("last_train") or {}).get("step"),
            }
        )
    return out


def _downsample(rows: list[dict[str, Any]], max_pts: int = 500) -> list[dict[str, Any]]:
    """Uniformly downsample a list to at most max_pts entries."""
    n = len(rows)
    if n <= max_pts:
        return rows
    stride = n / max_pts
    return [rows[int(i * stride)] for i in range(max_pts)]


def _get_run_detail(run_id: str) -> tuple[dict[str, Any], int]:
    """Return (response_dict, http_status) for GET /api/training?id=<run_id>."""
    root = _runs_root()
    run_dir = root / run_id
    if not run_dir.is_dir():
        return {"error": f"run not found: {run_id}"}, 404
    data = _read_run_json(run_dir)
    if data is None:
        return {"error": f"run.json missing or invalid for: {run_id}"}, 404
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    metrics_path = run_dir / "metrics.jsonl"
    try:
        with open(metrics_path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row: dict[str, Any] = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                split = row.get("split")
                if split == "train":
                    train_rows.append(row)
                elif split == "val":
                    val_rows.append(row)
    except Exception:
        pass
    metrics = _downsample(train_rows) + _downsample(val_rows)
    return {"run": data, "metrics": metrics}, 200


def _read_profile(run_id: str) -> dict[str, Any]:
    """Return runs/<id>/profile.json (PyTorch profiler op breakdown), or {} if absent."""
    if not run_id:
        return {}
    path = _runs_root() / run_id / "profile.json"
    try:
        with open(path) as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return {}


def _get_system_info() -> dict[str, Any]:
    """Return GPU stats from nvidia-smi; gracefully returns empty dict if unavailable."""
    info: dict[str, Any] = {}
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
                    "memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 5:
            info["gpu_util"] = int(parts[0])
            info["gpu_mem_bw"] = int(parts[1])  # memory-controller BW util % (dmon "mem")
            info["gpu_mem_used"] = int(parts[2])
            info["gpu_mem_total"] = int(parts[3])
            info["gpu_temp"] = int(parts[4])
        if len(parts) >= 6:
            with contextlib.suppress(ValueError):
                info["gpu_power"] = round(float(parts[5]))
    except Exception:
        pass
    return info


def _make_handler(session: GameSession) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - matches base
            pass  # quiet: suppress default request logging

        def _send_json(self, obj: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                body = _HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._send_json(session.state())
            elif self.path == "/api/training" or self.path.startswith("/api/training?"):
                parsed = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(parsed.query)
                run_id_list = qs.get("id")
                if run_id_list:
                    resp, status = _get_run_detail(run_id_list[0])
                    self._send_json(resp, status)
                else:
                    self._send_json({"runs": _list_runs()})
            elif self.path == "/api/models":
                self._send_json({"models": _list_models()})
            elif self.path == "/api/checkpoints":
                self._send_json({"checkpoints": _list_checkpoints()})
            elif self.path == "/api/system":
                self._send_json(_get_system_info())
            elif self.path.startswith("/api/profile"):
                parsed = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(parsed.query)
                rid = (qs.get("id") or [""])[0]
                self._send_json(_read_profile(rid))
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            data = self._read_json()
            if self.path == "/api/new":
                self._send_json(
                    session.reset(
                        str(data.get("engine", "neural")),
                        str(data.get("human", "red")),
                        float(data.get("temperature", 0.0)),
                    )
                )
            elif self.path == "/api/move":
                self._send_json(session.human_move(str(data.get("from")), str(data.get("to"))))
            elif self.path == "/api/undo":
                self._send_json(session.undo())
            elif self.path == "/api/engine/shutdown":
                self._send_json(session.shutdown_engine())
            elif self.path == "/api/engine/start":
                self._send_json(session.start_engine())
            elif self.path == "/api/engine/reload":
                raw = data.get("checkpoint")
                ckpt = str(raw) if raw else None
                # Only load a checkpoint the server discovered (localhost-dev safety).
                if ckpt is not None and ckpt not in {c["path"] for c in _list_checkpoints()}:
                    self._send_json({"error": "unknown checkpoint", "state": session.state()}, 400)
                else:
                    self._send_json(session.reload_engine(ckpt))
            else:
                self.send_error(404)

    return Handler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    engine: str = "neural",
    checkpoint: str | None = None,
    open_browser: bool = True,
) -> None:
    """Start the web-play server (blocking) and optionally open a browser."""
    session = GameSession(engine, checkpoint)
    httpd = ThreadingHTTPServer((host, port), _make_handler(session))
    url = f"http://{host}:{port}/"
    print(f"Dong Feng web UI on {url}  (engine: {engine})  — Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…", flush=True)
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- #
# Embedded single-page board UI (no external assets; strict-CSP friendly).
# --------------------------------------------------------------------------- #
_HTML = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Dong Feng — Cờ tướng</title>
<style>
  :root { --wood:#e8c48c; --line:#5b3a1a; --red:#c0392b; --black:#1c1c1c; --cream:#f4e4c1; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
         background:#2b2b2b; color:#eee; display:flex; flex-direction:column; align-items:center; padding:16px; }
  .tabs { display:flex; gap:4px; margin-bottom:12px; }
  .tab-btn { background:#444; border:none; color:#ccc; padding:8px 20px; border-radius:6px 6px 0 0; cursor:pointer; font-size:14px; font-weight:600; }
  .tab-btn.active { background:#c0392b; color:#fff; }
  .tab-content { display:none; }
  .tab-content.active { display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start; }
  h1 { font-size:20px; margin:0 0 10px; }
  #board { background:var(--wood); border-radius:6px; box-shadow:0 6px 24px rgba(0,0,0,.5); touch-action:manipulation; }
  .panel { min-width:230px; max-width:280px; }
  .card { background:#333; border-radius:8px; padding:12px 14px; margin-bottom:12px; }
  label { display:block; font-size:13px; margin:8px 0 4px; color:#bbb; }
  select, button, input { font-size:14px; padding:6px 8px; border-radius:6px; border:1px solid #555;
          background:#222; color:#eee; width:100%; }
  button { background:var(--red); border:none; cursor:pointer; font-weight:600; }
  button:hover { filter:brightness(1.1); }
  button.secondary { background:#444; }
  #status { font-size:15px; font-weight:600; min-height:22px; }
  #result { font-size:16px; color:#ffd36b; min-height:20px; margin-top:6px; }
  #moves { font-family: ui-monospace, monospace; font-size:12px; color:#cfcfcf; max-height:220px;
           overflow:auto; white-space:pre-wrap; line-height:1.5; }
  .row { display:flex; gap:8px; }
  .muted { color:#999; font-size:12px; }
  /* Training panel styles */
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
  .chip { background:#222; border-radius:20px; padding:4px 12px; font-size:12px; color:#ccc; border:1px solid #555; }
  .chip span { color:#fff; font-weight:600; }
  .legend { display:flex; gap:16px; margin-top:6px; font-size:12px; }
  .legend-item { display:flex; align-items:center; gap:6px; }
  .legend-dot { width:14px; height:4px; border-radius:2px; }
  .status-badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700; }
  .status-running { background:#2255aa; color:#adf; }
  .status-done { background:#225533; color:#afd; }
  .status-failed { background:#552222; color:#faa; }
  .progress-bar-wrap { background:#111; border-radius:6px; height:16px; overflow:hidden; margin-top:10px; }
  .progress-bar-fill { height:100%; background:linear-gradient(90deg,#1a4a9a,#4da6ff); border-radius:6px; transition:width .6s ease; }
  .progress-label { font-size:11px; color:#888; margin-top:4px; }
  .gpu-bar { display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }
  .gpu-chip { background:#1a2a1a; border:1px solid #3a5a3a; border-radius:20px; padding:4px 12px; font-size:12px; color:#8fbc8f; }
  .gpu-chip span { color:#afd; font-weight:700; }
  .charts-row { display:flex; gap:16px; flex-wrap:wrap; width:100%; max-width:960px; }
  .charts-row .card { flex:1; min-width:260px; }
  #training-chart, #training-chart2 { display:block; width:100%; height:240px; background:#1a1a1a; border-radius:6px; }
  /* Richer monitor: responsive grid of insight charts */
  .chart-grid { display:grid; grid-template-columns:repeat(2,minmax(280px,1fr)); gap:16px; width:100%; max-width:960px; }
  @media (max-width:720px){ .chart-grid { grid-template-columns:1fr; } }
  .chart-grid canvas { display:block; width:100%; height:200px; background:#1a1a1a; border-radius:6px; }
  .arch-hash { font-family: ui-monospace, monospace; font-size:11px; color:#7fd0ff; background:#12233a; border:1px solid #2a4a6a; border-radius:6px; padding:2px 8px; }
  .insight-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:8px; margin-top:8px; }
  .insight { background:#1c1c1c; border:1px solid #444; border-radius:8px; padding:8px 10px; }
  .insight .k { font-size:10px; color:#888; text-transform:uppercase; letter-spacing:.5px; }
  .insight .v { font-size:17px; font-weight:700; color:#fff; margin-top:2px; }
  .insight .sub { font-size:10px; color:#7a9; margin-top:2px; }
  .insight.warn .v { color:#ffb347; }
  .insight.good .v { color:#7fdf9f; }
</style>
</head>
<body>
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('play')">Cờ tướng</button>
  <button class="tab-btn" onclick="switchTab('training')" id="tab-training-btn">Training</button>
</div>
<div id="tab-play" class="tab-content active">
  <div>
    <h1>东风 · Dong Feng — Cờ tướng</h1>
    <svg id="board" width="540" height="600" viewBox="0 0 540 600"></svg>
  </div>
  <div class="panel">
    <div class="card">
      <div id="status">Đang tải…</div>
      <div id="result"></div>
    </div>
    <div class="card">
      <label>Đối thủ (engine)</label>
      <select id="engine">
        <option value="neural">Neural (model đã train)</option>
        <option value="board">Board transformer</option>
        <option value="random">Random (baseline)</option>
      </select>
      <label>Bạn cầm quân</label>
      <select id="human">
        <option value="red">Đỏ (đi trước)</option>
        <option value="black">Đen</option>
      </select>
      <label>Độ ngẫu nhiên (temperature): <span id="tval">0.0</span></label>
      <input type="range" id="temp" min="0" max="1.5" step="0.1" value="0"/>
      <div class="row" style="margin-top:10px;">
        <button id="new">Ván mới</button>
        <button id="undo" class="secondary">Đi lại</button>
      </div>
      <div class="row" style="margin-top:10px;">
        <button id="engine-toggle" class="secondary">Tắt engine (nhường RAM cho training)</button>
      </div>
      <label style="margin-top:10px;">Checkpoint (hot-reload)</label>
      <select id="ckpt-select">
        <option value="">-- mặc định (env) --</option>
      </select>
      <div class="row" style="margin-top:8px;">
        <button id="ckpt-reload" class="secondary">Nạp checkpoint này</button>
      </div>
      <div class="muted" id="ckpt-current" style="margin-top:6px;"></div>
    </div>
    <div class="card">
      <label>Nước đi (ICCS)</label>
      <div id="moves"></div>
    </div>
    <div class="muted">Bấm quân của bạn rồi bấm ô đích. Chấm xanh = nước hợp lệ.</div>
  </div>
</div>
<div id="tab-training" class="tab-content">
  <div style="width:100%;max-width:960px;">
    <h1>Training Dashboard</h1>
    <div class="card">
      <label>Model (theo arch-hash)</label>
      <select id="training-model-select" onchange="selectModel(this.value)">
        <option value="">-- chọn model --</option>
      </select>
      <div id="training-model-meta" class="muted" style="margin-top:6px;"></div>
      <label style="margin-top:10px;">Training run</label>
      <select id="training-run-select" onchange="selectRun(this.value)">
        <option value="">-- chọn run --</option>
      </select>
    </div>
    <div class="card" id="training-stats-card" style="display:none;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;">
        <strong id="training-run-id" style="font-size:15px;"></strong>
        <span id="training-status-badge" class="status-badge"></span>
        <span id="training-arch-hash" class="arch-hash"></span>
        <span id="training-eta" style="font-size:12px;color:#aaa;margin-left:auto;"></span>
      </div>
      <div class="chips" id="training-chips"></div>
      <div class="progress-bar-wrap" id="training-progress-wrap" style="display:none;">
        <div class="progress-bar-fill" id="training-progress-fill" style="width:0%"></div>
      </div>
      <div class="progress-label" id="training-progress-label"></div>
      <div class="insight-grid" id="training-insights"></div>
    </div>
    <div class="card" id="training-gpu-card" style="display:none;">
      <label style="margin-bottom:4px;">GPU</label>
      <div class="gpu-bar" id="training-gpu-bar"></div>
    </div>
    <div class="card" id="training-profile-card" style="display:none;">
      <label style="margin-bottom:4px;">PyTorch Profiler — op breakdown (FLOPS &amp; device time)</label>
      <div class="chips" id="training-profile-summary"></div>
      <div style="overflow-x:auto;margin-top:8px;">
        <table id="training-profile-table" style="width:100%;border-collapse:collapse;font-size:12px;">
        </table>
      </div>
      <div class="muted" id="training-profile-note" style="margin-top:6px;"></div>
    </div>
    <div class="chart-grid" id="training-chart-card" style="display:none;">
      <div class="card">
        <label>Policy Loss (log)</label>
        <canvas id="training-chart"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#4da6ff;"></div>Train</div>
          <div class="legend-item"><div class="legend-dot" style="background:#ff7043;"></div>Val</div>
        </div>
      </div>
      <div class="card">
        <label>Val Accuracy</label>
        <canvas id="training-chart2"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#4caf50;"></div>Top-1</div>
          <div class="legend-item"><div class="legend-dot" style="background:#c58af9;"></div>Top-5</div>
        </div>
      </div>
      <div class="card">
        <label>Learning Rate</label>
        <canvas id="training-chart-lr"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#ffd36b;"></div>LR</div>
        </div>
      </div>
      <div class="card">
        <label>Gradient Norm</label>
        <canvas id="training-chart-grad"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#ff6b9d;"></div>‖grad‖ (clip=1.0)</div>
        </div>
      </div>
      <div class="card">
        <label>Throughput (tokens/s)</label>
        <canvas id="training-chart-tps"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#4dd0e1;"></div>tok/s</div>
        </div>
      </div>
      <div class="card">
        <label>Generalization gap (val − train loss)</label>
        <canvas id="training-chart-gap"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#ffa726;"></div>gap</div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
// ---- Tab switching ----
function switchTab(name){
  document.querySelectorAll(".tab-content").forEach(el=>el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el=>el.classList.remove("active"));
  document.getElementById("tab-"+name).classList.add("active");
  event.target.classList.add("active");
  if(name==="training") loadModels();
}

// ---- Board play ----
const FILES = "abcdefghi";
const RED = {K:"帅",A:"仕",B:"相",N:"马",R:"车",C:"炮",P:"兵"};
const BLACK = {k:"将",a:"士",b:"象",n:"马",r:"车",c:"炮",p:"卒"};
const CELL=54, M=36;
const svg = document.getElementById("board");
let state=null, sel=null, busy=false;

function sq(col,rank){ return FILES[col]+rank; }
function px(col,rank){ return [M+col*CELL, M+(9-rank)*CELL]; }

function parseFen(fen){
  const rows = fen.split(" ")[0].split("/"); // row0 = rank9 (top)
  const occ = {}; // sq -> letter
  for(let i=0;i<10;i++){ const rank=9-i; let col=0;
    for(const ch of rows[i]){
      if(/\d/.test(ch)){ col+=parseInt(ch); }
      else { occ[sq(col,rank)]=ch; col++; }
    }
  }
  return occ;
}

function el(tag,attrs,text){ const e=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const k in attrs) e.setAttribute(k,attrs[k]); if(text!=null) e.textContent=text; return e; }

function render(){
  svg.innerHTML="";
  // grid lines
  for(let r=0;r<10;r++){ const [x0,y]=px(0,r); const [x1]=px(8,r);
    svg.appendChild(el("line",{x1:x0,y1:y,x2:x1,y2:y,stroke:"var(--line)","stroke-width":1.4})); }
  for(let c=0;c<9;c++){
    if(c===0||c===8){ const [x,y0]=px(c,9); const [,y1]=px(c,0);
      svg.appendChild(el("line",{x1:x,y1:y0,x2:x,y2:y1,stroke:"var(--line)","stroke-width":1.4})); }
    else { // stop at the river (between rank5 and rank4)
      let a=px(c,9), b=px(c,5); svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"var(--line)","stroke-width":1.4}));
      let d=px(c,4), e2=px(c,0); svg.appendChild(el("line",{x1:d[0],y1:d[1],x2:e2[0],y2:e2[1],stroke:"var(--line)","stroke-width":1.4})); }
  }
  // palaces (X)
  const palace=(r0,r1)=>{ let a=px(3,r0),b=px(5,r1); svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:"var(--line)","stroke-width":1.2}));
    let c=px(5,r0),d=px(3,r1); svg.appendChild(el("line",{x1:c[0],y1:c[1],x2:d[0],y2:d[1],stroke:"var(--line)","stroke-width":1.2})); };
  palace(9,7); palace(2,0);
  // river text
  const ry=M+4.5*CELL;
  svg.appendChild(el("text",{x:M+1.6*CELL,y:ry+6,"font-size":20,fill:"#7a4a1a","letter-spacing":6},"楚 河"));
  svg.appendChild(el("text",{x:M+5.4*CELL,y:ry+6,"font-size":20,fill:"#7a4a1a","letter-spacing":6},"漢 界"));

  if(!state) return;
  const occ=parseFen(state.fen);
  const last=state.last_move;
  if(last){ const [lx,ly]=px(FILES.indexOf(last[1][0]),parseInt(last[1][1]));
    svg.appendChild(el("circle",{cx:lx,cy:ly,r:CELL*0.46,fill:"none",stroke:"#3aa0ff","stroke-width":2})); }
  // legal dest dots for selected
  if(sel){ const dests=(state.legal||[]).filter(m=>m[0]===sel).map(m=>m[1]);
    for(const d of dests){ const [x,y]=px(FILES.indexOf(d[0]),parseInt(d[1]));
      const hit=occ[d]; svg.appendChild(el("circle",{cx:x,cy:y,r:hit?CELL*0.44:7,
        fill:hit?"none":"#2ecc71",stroke:hit?"#2ecc71":"none","stroke-width":3,opacity:0.85})); } }
  // pieces
  for(const s in occ){ const ch=occ[s]; const isRed=ch===ch.toUpperCase();
    const [x,y]=px(FILES.indexOf(s[0]),parseInt(s[1]));
    svg.appendChild(el("circle",{cx:x,cy:y,r:CELL*0.42,fill:"var(--cream)",
      stroke:isRed?"var(--red)":"var(--black)","stroke-width":s===sel?4:2}));
    svg.appendChild(el("text",{x:x,y:y+7,"font-size":24,"text-anchor":"middle",
      fill:isRed?"var(--red)":"var(--black)","font-weight":700},(isRed?RED:BLACK)[ch])); }
}

function nearest(evt){ const r=svg.getBoundingClientRect();
  const sx=(evt.clientX-r.left)*(540/r.width), sy=(evt.clientY-r.top)*(600/r.height);
  const col=Math.round((sx-M)/CELL), row=Math.round((sy-M)/CELL);
  if(col<0||col>8||row<0||row>9) return null; return sq(col,9-row); }

svg.addEventListener("click",async(evt)=>{
  if(busy||!state||state.result!=="ongoing"||state.turn!==state.human) return;
  const s=nearest(evt); if(!s) return;
  const occ=parseFen(state.fen); const ch=occ[s];
  const mine=ch && ((state.human==="red")===(ch===ch.toUpperCase()));
  if(sel && (state.legal||[]).some(m=>m[0]===sel&&m[1]===s)){ await move(sel,s); sel=null; }
  else if(mine){ sel=s; render(); }
  else { sel=null; render(); }
});

async function move(frm,to){ busy=true; setStatus("Máy đang nghĩ…");
  const r=await fetch("/api/move",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({from:frm,to:to})});
  const j=await r.json(); busy=false;
  if(j.error){ setStatus("⚠ "+j.error); return; }
  state=j.state; sel=null; render(); update(); }

async function refresh(){ state=await(await fetch("/api/state")).json(); render(); update(); loadCheckpoints(); }
async function newGame(){ busy=true; sel=null; setStatus("Đang tạo ván…");
  const body={engine:document.getElementById("engine").value,human:document.getElementById("human").value,
    temperature:parseFloat(document.getElementById("temp").value)};
  state=await(await fetch("/api/new",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)})).json(); busy=false; render(); update(); }

function setStatus(t){ document.getElementById("status").textContent=t; }
function update(){
  const names={red:"Đỏ",black:"Đen"};
  let s = state.result==="ongoing"
    ? "Lượt: "+names[state.turn]+(state.turn===state.human?" (bạn)":" (máy)")+(state.in_check?"  — CHIẾU!":"")
    : "Kết thúc";
  setStatus(s);
  const res={ongoing:"",red_win:"🔴 Đỏ thắng!",black_win:"⚫ Đen thắng!",draw:"Hòa"};
  document.getElementById("result").textContent=res[state.result]||"";
  let out=""; state.history.forEach((m,i)=>{ if(i%2===0) out+=(i/2+1)+". "; out+=m+(i%2===0?"  ":"\n"); });
  document.getElementById("moves").textContent=out;
  const et=document.getElementById("engine-toggle");
  if(state.engine_loaded===false){ et.textContent="Bật engine"; et.classList.add("active"); }
  else { et.textContent="Tắt engine (nhường RAM cho training)"; et.classList.remove("active"); }
  const cc=document.getElementById("ckpt-current");
  if(cc){ cc.textContent = "Đang dùng: " + (state.checkpoint
    ? state.checkpoint.split("/").slice(-2).join("/") : "mặc định (env)"); }
}
async function undo(){ if(busy||!state) return; busy=true; sel=null; setStatus("Đang đi lại…");
  state=await(await fetch("/api/undo",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"})).json();
  busy=false; render(); update(); }
async function toggleEngine(){ if(busy||!state) return; busy=true;
  const path=state.engine_loaded===false?"/api/engine/start":"/api/engine/shutdown";
  setStatus(state.engine_loaded===false?"Đang bật engine…":"Đang tắt engine…");
  state=await(await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"})).json();
  busy=false; render(); update(); loadCheckpoints(); }

async function loadCheckpoints(){
  try{
    const data=await(await fetch("/api/checkpoints")).json();
    const sel=document.getElementById("ckpt-select");
    const cur=state?state.checkpoint:null;
    sel.innerHTML='<option value="">-- mặc định (env) --</option>';
    for(const c of (data.checkpoints||[])){
      const opt=document.createElement("option"); opt.value=c.path;
      const p=c.params!=null?" · "+(c.params/1e6).toFixed(1)+"M":"";
      const t1=c.top1!=null?" · top1="+Number(c.top1).toFixed(3):"";
      opt.textContent=`${c.id} [${c.status||"?"}]${p}${t1}`;
      if(c.path===cur) opt.selected=true;
      sel.appendChild(opt);
    }
  }catch(e){ console.error("checkpoints error",e); }
}
async function reloadCkpt(){ if(busy||!state) return; busy=true;
  const ckpt=document.getElementById("ckpt-select").value||null;
  setStatus("Đang nạp checkpoint…");
  const j=await(await fetch("/api/engine/reload",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify({checkpoint:ckpt})})).json();
  busy=false;
  if(j.error){ setStatus("⚠ "+j.error); if(j.state){ state=j.state; render(); update(); } return; }
  state=j; render(); update(); }

document.getElementById("new").addEventListener("click",newGame);
document.getElementById("undo").addEventListener("click",undo);
document.getElementById("engine-toggle").addEventListener("click",toggleEngine);
document.getElementById("ckpt-reload").addEventListener("click",reloadCkpt);
document.getElementById("temp").addEventListener("input",e=>document.getElementById("tval").textContent=e.target.value);
refresh();

// ---- Training dashboard ----
let _trainingPollTimer = null;
let _systemPollTimer = null;
let _currentRunId = null;
let _selectedArch = "";   // "" = all models
let _modelsCache = [];

async function loadModels(){
  try {
    const resp = await fetch("/api/models");
    const data = await resp.json();
    _modelsCache = data.models || [];
    const sel = document.getElementById("training-model-select");
    const cur = sel.value;
    sel.innerHTML = '<option value="">-- tất cả model --</option>';
    for(const m of _modelsCache){
      const opt = document.createElement("option");
      opt.value = m.arch_hash||"";
      const p = m.params != null ? " · "+(m.params/1e6).toFixed(1)+"M" : "";
      const best = m.best_top1 != null ? " · best top1="+Number(m.best_top1).toFixed(3) : "";
      opt.textContent = `${m.arch_summary||m.preset||"?"} [${(m.arch_hash||"").slice(0,8)}]${p} · ${m.runs.length} run${best}`;
      sel.appendChild(opt);
    }
    if(cur){ sel.value = cur; }
    updateModelMeta();
    loadTrainingList();
  } catch(e){ console.error("models error",e); }
}

function selectModel(hash){
  _selectedArch = hash || "";
  updateModelMeta();
  // Reset run selection when the model filter changes.
  const rsel = document.getElementById("training-run-select");
  rsel.value = "";
  selectRun("");
  loadTrainingList();
}

function updateModelMeta(){
  const el = document.getElementById("training-model-meta");
  if(!_selectedArch){ el.textContent = _modelsCache.length ? `${_modelsCache.length} model design(s)` : ""; return; }
  const m = _modelsCache.find(x=>x.arch_hash===_selectedArch);
  if(!m){ el.textContent=""; return; }
  const a = m.arch||{};
  el.innerHTML = `arch-hash <span class="arch-hash">${m.arch_hash}</span> · `
    + `d_model=${a.d_model} · layers=${a.n_layer} · heads=${a.n_head}`
    + (a.n_bias_head?` (+${a.n_bias_head} bias)`:"")
    + ` · ffn=${a.ffn_hidden} · ${m.params!=null?(m.params/1e6).toFixed(2)+"M params":""}`;
}

async function loadTrainingList(){
  try {
    const resp = await fetch("/api/training");
    const data = await resp.json();
    const sel = document.getElementById("training-run-select");
    const cur = sel.value;
    sel.innerHTML = '<option value="">-- chọn run --</option>';
    let runs = data.runs||[];
    if(_selectedArch){ runs = runs.filter(r=>(r.arch_hash||"")===_selectedArch); }
    for(const run of runs){
      const opt = document.createElement("option");
      opt.value = run.id||"";
      const st = run.status||"";
      const lt = run.last_train;
      const lv = run.last_val;
      const step = lt ? lt.step : "?";
      const top1 = lv && lv.top1 != null ? " top1="+Number(lv.top1).toFixed(3) : "";
      opt.textContent = `${run.id||"?"} [${st}] step=${step}${top1}`;
      sel.appendChild(opt);
    }
    if(cur){ sel.value = cur; }
  } catch(e){ console.error("training list error",e); }
}

function selectRun(runId){
  if(_trainingPollTimer){ clearInterval(_trainingPollTimer); _trainingPollTimer=null; }
  if(_systemPollTimer){ clearInterval(_systemPollTimer); _systemPollTimer=null; }
  _currentRunId = runId;
  if(!runId){
    ["training-stats-card","training-gpu-card","training-chart-card","training-profile-card"].forEach(id=>document.getElementById(id).style.display="none");
    return;
  }
  fetchRunDetail(runId);
  loadSystemInfo();
}

async function loadProfile(runId){
  try{
    const data = await (await fetch("/api/profile?id="+encodeURIComponent(runId))).json();
    const card = document.getElementById("training-profile-card");
    if(!data || (!data.ops && !data.error)){ card.style.display="none"; return; }
    card.style.display="";
    if(data.error){
      document.getElementById("training-profile-summary").innerHTML = "";
      document.getElementById("training-profile-table").innerHTML = "";
      document.getElementById("training-profile-note").textContent = "Profiler error: "+data.error;
      return;
    }
    const PEAK = 48; // RTX 5060 Ti bf16 dense ≈ 48 — same constant as the perf cards
    const mfu = data.measured_tflops!=null ? (data.measured_tflops/PEAK*100) : null;
    document.getElementById("training-profile-summary").innerHTML = [
      ["ms/step", data.ms_per_step!=null?data.ms_per_step+" ms":"—"],
      ["GFLOP/step", data.gflops_per_step!=null?data.gflops_per_step:"—"],
      ["measured", data.measured_tflops!=null?data.measured_tflops+" TFLOP/s":"—"],
      ["MFU", mfu!=null?mfu.toFixed(1)+"%":"—"],
      ["device", data.device||"—"],
    ].map(([k,v])=>`<div class="chip">${k}: <span>${v}</span></div>`).join("");
    const rows = (data.ops||[]).slice(0,15);
    const head = "<tr style='text-align:left;color:#888;border-bottom:1px solid #444;'>"
      + "<th style='padding:4px 6px;'>op</th><th style='padding:4px 6px;'>device %</th>"
      + "<th style='padding:4px 6px;'>GFLOPs</th><th style='padding:4px 6px;'>calls</th></tr>";
    const body = rows.map(r=>{
      const bar = `<div style="background:#2a4a6a;height:10px;width:${Math.min(100,r.device_pct)}%;border-radius:2px;"></div>`;
      return `<tr style="border-bottom:1px solid #2a2a2a;">`
        + `<td style="padding:4px 6px;font-family:ui-monospace,monospace;color:#cfcfcf;">${r.name}</td>`
        + `<td style="padding:4px 6px;">${r.device_pct}% ${bar}</td>`
        + `<td style="padding:4px 6px;color:#7fdf9f;">${r.gflops}</td>`
        + `<td style="padding:4px 6px;color:#999;">${r.count}</td></tr>`;
    }).join("");
    document.getElementById("training-profile-table").innerHTML = head + body;
    document.getElementById("training-profile-note").textContent =
      "FLOPs = fwd+bwd aten ops (mm/bmm) that torch recognises; flash-attention time shows but its FLOPs aren't counted. Sorted by device time.";
  }catch(e){ console.error("profile error",e); }
}

async function fetchRunDetail(runId){
  try {
    const resp = await fetch("/api/training?id="+encodeURIComponent(runId));
    const data = await resp.json();
    if(data.error){ console.error("run detail error",data.error); return; }
    renderTrainingDetail(data.run, data.metrics||[]);
    loadProfile(runId);
    if(data.run.status==="running"){
      if(!_trainingPollTimer){
        _trainingPollTimer = setInterval(()=>{
          if(_currentRunId===runId){ fetchRunDetail(runId); loadTrainingList(); }
          else clearInterval(_trainingPollTimer);
        }, 2000);
      }
      if(!_systemPollTimer){ _systemPollTimer = setInterval(loadSystemInfo, 5000); }
    } else {
      if(_trainingPollTimer){ clearInterval(_trainingPollTimer); _trainingPollTimer=null; }
      if(_systemPollTimer){ clearInterval(_systemPollTimer); _systemPollTimer=null; }
    }
  } catch(e){ console.error("fetchRunDetail error",e); }
}

async function loadSystemInfo(){
  try {
    const resp = await fetch("/api/system");
    const data = await resp.json();
    const card = document.getElementById("training-gpu-card");
    const bar  = document.getElementById("training-gpu-bar");
    if(data.gpu_util != null){
      card.style.display="";
      const mem = data.gpu_mem_used != null
        ? `${data.gpu_mem_used} / ${data.gpu_mem_total} MiB`
        : "—";
      bar.innerHTML = [
        ["SM util",  data.gpu_util+"%"],
        ["Mem BW",   data.gpu_mem_bw != null ? data.gpu_mem_bw+"%" : "—"],
        ["VRAM",     mem],
        ["Power",    data.gpu_power != null ? data.gpu_power+" W" : "—"],
        ["Temp",     data.gpu_temp != null ? data.gpu_temp+"°C" : "—"],
      ].map(([k,v])=>`<div class="gpu-chip">${k}: <span>${v}</span></div>`).join("");
    } else { card.style.display="none"; }
  } catch(e){ /* no gpu */ }
}

function renderTrainingDetail(run, metrics){
  document.getElementById("training-stats-card").style.display="";
  document.getElementById("training-chart-card").style.display="";
  document.getElementById("training-run-id").textContent = run.id||"";
  const badge = document.getElementById("training-status-badge");
  const st = run.status||"";
  badge.textContent = st; badge.className = "status-badge status-"+st;
  document.getElementById("training-arch-hash").textContent = run.arch_hash ? ("arch "+run.arch_hash) : "";

  const trainM = metrics.filter(m=>m.split==="train");
  const valM   = metrics.filter(m=>m.split==="val");
  const last   = trainM.length ? trainM[trainM.length-1] : null;
  const lastV  = valM.length   ? valM[valM.length-1]   : null;
  const fmt = (v,d=4)=> v==null?"—":Number(v).toFixed(d);

  // ETA
  const cfg = run.config || {};
  const maxSteps = cfg.max_steps ? Number(cfg.max_steps) : 0;
  const curStep  = last ? Number(last.step)+1 : 0;
  let etaStr = "", etaSec = 0;
  if(st==="running" && last && maxSteps && last.elapsed_s && last.step > 0){
    etaSec = Math.max(0, (maxSteps-curStep) * last.elapsed_s / curStep);
    const h = Math.floor(etaSec/3600), m = Math.floor((etaSec%3600)/60);
    etaStr = `ETA ~${h}h ${m}m`;
  }
  document.getElementById("training-eta").textContent = etaStr;

  // Progress bar
  if(maxSteps > 0){
    const pct = Math.min(100, curStep/maxSteps*100);
    document.getElementById("training-progress-wrap").style.display="";
    document.getElementById("training-progress-fill").style.width = pct.toFixed(1)+"%";
    document.getElementById("training-progress-label").textContent =
      `${curStep.toLocaleString()} / ${maxSteps.toLocaleString()} steps (${pct.toFixed(1)}%)`;
  } else {
    document.getElementById("training-progress-wrap").style.display="none";
    document.getElementById("training-progress-label").textContent="";
  }

  // Chips
  document.getElementById("training-chips").innerHTML = [
    ["preset",     run.preset||"—"],
    ["params",     run.params!=null ? (Number(run.params)/1e6).toFixed(2)+"M" : "—"],
    ["device",     cfg.device||"—"],
    ["dtype",      run.dtype||"—"],
    ["batch",      cfg.batch_size||"—"],
    ["optim",      cfg.optim||"—"],
  ].map(([k,v])=>`<div class="chip">${k}: <span>${v}</span></div>`).join("");

  // ---- Data-scientist insight cards ----
  const trainLoss = last ? (last.policy_loss??last.loss) : null;
  const valLoss   = lastV ? (lastV.policy_loss??lastV.loss) : null;
  const gap = (trainLoss!=null && valLoss!=null) ? (valLoss - trainLoss) : null;
  const bestTop1 = valM.reduce((b,r)=> (r.top1!=null && r.top1>b ? r.top1 : b), -1);
  // Recent val-loss slope (improving vs plateau) over the last few evals.
  let trend = "—", trendCls="";
  if(valM.length>=3){
    const a=valM[valM.length-3], b=lastV;
    const d=(b.policy_loss??b.loss)-(a.policy_loss??a.loss);
    if(d < -1e-3){ trend="↓ improving"; trendCls="good"; }
    else if(d > 1e-3){ trend="↑ rising"; trendCls="warn"; }
    else { trend="→ plateau"; trendCls="warn"; }
  }
  const tps = last ? last.tokens_per_s : null;
  const gradN = last ? last.grad_norm : null;
  const elapsedH = last && last.elapsed_s ? (last.elapsed_s/3600) : null;
  // ---- Perf metrics (derived) — compare these before/after an optimization ----
  // Board model = fixed 91-token sequence. Training FLOPs ≈ 6·N·tokens (fwd+bwd).
  // Set GPU_PEAK_TFLOPS to your GPU's bf16 dense peak (RTX 5090 ≈ 209) to show MFU%.
  const SEQ_LEN = 91;
  const GPU_PEAK_TFLOPS = 48;   // RTX 5060 Ti bf16 dense ≈ 48. Verify/adjust via verify-gpu.yml.
  const bs = (run.config && run.config.batch_size) ? Number(run.config.batch_size) : null;
  const nParams = run.params != null ? Number(run.params) : null;
  const samplesPerS = tps!=null ? tps/SEQ_LEN : null;
  const stepsPerS   = (samplesPerS!=null && bs) ? samplesPerS/bs : null;
  const msPerStep   = stepsPerS ? 1000/stepsPerS : null;
  const effTflops   = (tps!=null && nParams!=null) ? 6*nParams*tps/1e12 : null;
  const mfu         = (effTflops!=null && GPU_PEAK_TFLOPS) ? effTflops/GPU_PEAK_TFLOPS*100 : null;
  const ic = (k,v,cls,sub)=>`<div class="insight ${cls||""}"><div class="k">${k}</div><div class="v">${v}</div>${sub?`<div class="sub">${sub}</div>`:""}</div>`;
  document.getElementById("training-insights").innerHTML = [
    ic("train loss", fmt(trainLoss,4)),
    ic("val loss",   fmt(valLoss,4)),
    ic("gen. gap",   gap==null?"—":fmt(gap,4), gap!=null&&gap>0.3?"warn":(gap!=null?"good":""), "val − train"),
    ic("val top-1",  lastV&&lastV.top1!=null?(lastV.top1*100).toFixed(2)+"%":"—", "good"),
    ic("val top-5",  lastV&&lastV.top5!=null?(lastV.top5*100).toFixed(2)+"%":"—"),
    ic("best top-1", bestTop1>=0?(bestTop1*100).toFixed(2)+"%":"—", "good"),
    ic("val trend",  trend, trendCls),
    ic("grad norm",  gradN==null?"—":fmt(gradN,3), gradN!=null&&gradN>=0.99?"warn":"", gradN!=null&&gradN>=0.99?"clipping":""),
    ic("throughput", tps==null?"—":Math.round(tps).toLocaleString()+" tok/s"),
    ic("samples/s",  samplesPerS==null?"—":Math.round(samplesPerS).toLocaleString(), "good"),
    ic("ms/step",    msPerStep==null?"—":msPerStep.toFixed(0)+" ms", "", bs?("batch "+bs):""),
    ic("eff. TFLOP/s", effTflops==null?"—":effTflops.toFixed(1), "good", "6·N·tok/s"),
    ic("MFU",        mfu==null?"—":mfu.toFixed(1)+"%",
                     mfu==null?"":(mfu>=40?"good":(mfu<25?"warn":"")),
                     "vs "+GPU_PEAK_TFLOPS+" TFLOP/s peak"),
    ic("lr",         last?fmt(last.lr,6):"—"),
    ic("elapsed",    elapsedH==null?"—":elapsedH.toFixed(2)+" h"),
    ic("remaining",  etaSec?((etaSec/3600).toFixed(2)+" h"):"—"),
  ].join("");

  drawLossChart(trainM, valM);
  drawAccuracyChart(valM);
  drawSeriesChart("training-chart-lr", trainM, r=>r.lr, "#ffd36b", v=>v.toExponential(1));
  drawSeriesChart("training-chart-grad", trainM, r=>r.grad_norm, "#ff6b9d", v=>v.toFixed(2));
  drawSeriesChart("training-chart-tps", trainM, r=>r.tokens_per_s, "#4dd0e1", v=>(v/1000).toFixed(0)+"k");
  drawGapChart(trainM, valM);
}

function _chartCtx(id){
  const canvas = document.getElementById(id);
  const dpr = window.devicePixelRatio||1;
  const W = canvas.offsetWidth||800, H = canvas.offsetHeight||200;
  canvas.width = W*dpr; canvas.height = H*dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr,dpr); ctx.clearRect(0,0,W,H);
  return {ctx,W,H};
}

function _empty(ctx,W,H,msg){ ctx.fillStyle="#666"; ctx.font="13px system-ui"; ctx.textAlign="center"; ctx.fillText(msg,W/2,H/2); }

function _drawGrid(ctx, PAD, W, H, xArr, yArr, yFmt){
  const cw=W-PAD.l-PAD.r, ch=H-PAD.t-PAD.b;
  const minX=Math.min(...xArr), maxX=Math.max(...xArr)||1;
  let minY=Math.min(...yArr), maxY=Math.max(...yArr)||1;
  if(maxY-minY < 1e-6){ minY-=0.001; maxY+=0.001; }
  ctx.strokeStyle="#555"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(PAD.l,PAD.t); ctx.lineTo(PAD.l,PAD.t+ch); ctx.lineTo(PAD.l+cw,PAD.t+ch); ctx.stroke();
  ctx.fillStyle="#888"; ctx.font="10px system-ui"; ctx.textAlign="right";
  for(let i=0;i<=4;i++){
    const v=minY+(maxY-minY)*i/4; const y=PAD.t+(1-i/4)*ch;
    ctx.fillText(yFmt?yFmt(v):v.toFixed(3), PAD.l-4, y+4);
    ctx.strokeStyle="#2a2a2a"; ctx.lineWidth=0.5; ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(PAD.l+cw,y); ctx.stroke();
  }
  ctx.textAlign="center"; ctx.fillStyle="#888";
  for(let i=0;i<=4;i++){
    const s=Math.round(minX+(maxX-minX)*i/4); ctx.fillText(s, PAD.l+cw*i/4, PAD.t+ch+20);
  }
  const xp=s=>PAD.l+(maxX===minX?cw/2:(s-minX)/(maxX-minX)*cw);
  const yp=v=>PAD.t+(1-(v-minY)/(maxY-minY))*ch;
  return {xp,yp};
}

// Generic single-series line chart keyed on step.
function drawSeriesChart(id, rows, getY, color, yFmt){
  const {ctx,W,H} = _chartCtx(id);
  const PAD={l:52,r:12,t:14,b:34};
  const pts = rows.filter(r=>{const y=getY(r); return y!=null&&isFinite(y);});
  if(!pts.length){ _empty(ctx,W,H,"No data yet."); return; }
  const {xp,yp}=_drawGrid(ctx,PAD,W,H,pts.map(r=>r.step),pts.map(getY),yFmt);
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  pts.forEach((r,i)=>{ const x=xp(r.step),y=yp(getY(r)); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.stroke();
}

function drawLossChart(trainRows, valRows){
  const {ctx,W,H} = _chartCtx("training-chart");
  const PAD={l:52,r:12,t:14,b:34};
  const getY=r=>{ const v=r.policy_loss??r.loss; return (v!=null&&v>0)?Math.log10(v):null; };
  const all=[...trainRows,...valRows].filter(r=>getY(r)!=null);
  if(!all.length){ _empty(ctx,W,H,"No metrics yet."); return; }
  const {xp,yp}=_drawGrid(ctx,PAD,W,H,all.map(r=>r.step),all.map(getY),v=>Math.pow(10,v).toFixed(3));
  const drawLine=(rows,color)=>{
    const pts=rows.filter(r=>getY(r)!=null);
    if(!pts.length)return;
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
    pts.forEach((r,i)=>{ const x=xp(r.step),y=yp(getY(r)); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.stroke();
  };
  drawLine(trainRows,"#4da6ff"); drawLine(valRows,"#ff7043");
}

function drawAccuracyChart(valRows){
  const {ctx,W,H} = _chartCtx("training-chart2");
  const PAD={l:52,r:12,t:14,b:34};
  const rows=valRows.filter(r=>r.top1!=null&&isFinite(r.top1));
  if(!rows.length){ _empty(ctx,W,H,"No val metrics yet."); return; }
  const ys=[...rows.map(r=>r.top1), ...rows.filter(r=>r.top5!=null).map(r=>r.top5)];
  const {xp,yp}=_drawGrid(ctx,PAD,W,H,rows.map(r=>r.step),ys,v=>(v*100).toFixed(1)+"%");
  const line=(getY,color)=>{
    const pts=rows.filter(r=>getY(r)!=null&&isFinite(getY(r)));
    if(!pts.length)return;
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
    pts.forEach((r,i)=>{ const x=xp(r.step),y=yp(getY(r)); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.stroke();
  };
  line(r=>r.top5,"#c58af9");
  line(r=>r.top1,"#4caf50");
  const lr=rows[rows.length-1];
  ctx.fillStyle="#4caf50"; ctx.beginPath(); ctx.arc(xp(lr.step),yp(lr.top1),4,0,Math.PI*2); ctx.fill();
  ctx.fillStyle="#afd"; ctx.font="11px system-ui"; ctx.textAlign="left";
  ctx.fillText((lr.top1*100).toFixed(1)+"%", xp(lr.step)+8, yp(lr.top1)+4);
}

// Generalization gap = val loss − interpolated train loss at each val step.
function drawGapChart(trainRows, valRows){
  const {ctx,W,H} = _chartCtx("training-chart-gap");
  const PAD={l:52,r:12,t:14,b:34};
  const tl=trainRows.filter(r=>(r.policy_loss??r.loss)!=null);
  const vs=valRows.filter(r=>(r.policy_loss??r.loss)!=null);
  if(!tl.length||!vs.length){ _empty(ctx,W,H,"No data yet."); return; }
  const trainAt=(step)=>{ // nearest train loss by step
    let best=tl[0], bd=Math.abs(tl[0].step-step);
    for(const r of tl){ const d=Math.abs(r.step-step); if(d<bd){bd=d;best=r;} }
    return best.policy_loss??best.loss;
  };
  const pts=vs.map(r=>({step:r.step, gap:(r.policy_loss??r.loss)-trainAt(r.step)}));
  const {xp,yp}=_drawGrid(ctx,PAD,W,H,pts.map(p=>p.step),pts.map(p=>p.gap),v=>v.toFixed(3));
  // zero line
  ctx.strokeStyle="#444"; ctx.lineWidth=1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(xp(pts[0].step),yp(0)); ctx.lineTo(xp(pts[pts.length-1].step),yp(0)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle="#ffa726"; ctx.lineWidth=2; ctx.beginPath();
  pts.forEach((p,i)=>{ const x=xp(p.step),y=yp(p.gap); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.stroke();
}
</script>
</body>
</html>
"""
