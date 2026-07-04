"""Prometheus exporter for Dong Feng training runs (stdlib only).

Scrapes ``runs/<id>/run.json`` + ``runs/<id>/metrics.jsonl`` (the artifacts written
by :mod:`dongfeng.training.board_loop`) and re-publishes the latest values in the
Prometheus text exposition format on ``GET /metrics``. A dedicated exporter keeps
training telemetry decoupled from the play server, so Prometheus can scrape it even
when the web UI's inference engine is shut down to free GPU memory.

Run it with ``dfc metrics-export`` (see :func:`serve_metrics`), or embed the
collector via :func:`render_exposition`.

Exposed series (all gauges), labelled by ``run`` (the run id):

* ``dongfeng_train_loss{run,split}``            -- total loss (train + val)
* ``dongfeng_train_policy_loss{run,split}``     -- policy cross-entropy
* ``dongfeng_train_value_loss{run,split}``      -- value MSE (train only; may be absent)
* ``dongfeng_train_top1{run}`` / ``_top5{run}`` -- validation move accuracy
* ``dongfeng_train_lr{run}``                    -- current learning rate
* ``dongfeng_train_grad_norm{run}``             -- gradient norm
* ``dongfeng_train_samples_per_s{run}``         -- throughput (samples/s)
* ``dongfeng_train_tokens_per_s{run}``          -- throughput (tokens/s)
* ``dongfeng_train_step{run}``                  -- latest optimizer step
* ``dongfeng_train_elapsed_seconds{run}``       -- wall-clock elapsed
* ``dongfeng_run_params{run}``                  -- parameter count
* ``dongfeng_run_status{run,preset,arch_hash,device,status}`` -- 1 per run (info gauge)
* ``dongfeng_runs_total``                       -- number of runs discovered
* ``dongfeng_up``                               -- 1 (exporter liveness)
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_MetricRow = dict[str, Any]


def _runs_root() -> Path:
    """Return the runs root directory, overridable via ``DONGFENG_RUNS_DIR``."""
    return Path(os.environ.get("DONGFENG_RUNS_DIR", "runs"))


def _read_run_json(run_dir: Path) -> dict[str, Any] | None:
    try:
        with open(run_dir / "run.json", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return None


def _read_last_metrics(metrics_path: Path) -> tuple[_MetricRow | None, _MetricRow | None]:
    """Return the last ``train`` row and last ``val`` row from ``metrics.jsonl``."""
    last_train: _MetricRow | None = None
    last_val: _MetricRow | None = None
    try:
        with open(metrics_path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row: _MetricRow = json.loads(raw)
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


def _esc_label(value: str) -> str:
    """Escape a Prometheus label value (backslash, double-quote, newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_esc_label(str(v))}"' for k, v in labels.items())
    return "{" + inner + "}"


class _Registry:
    """Accumulates samples and renders them in the Prometheus text format."""

    def __init__(self) -> None:
        self._help: dict[str, str] = {}
        self._samples: dict[str, list[tuple[dict[str, str], float]]] = {}

    def gauge(self, name: str, help_text: str) -> None:
        self._help[name] = help_text
        self._samples.setdefault(name, [])

    def add(self, name: str, value: float | int | None, labels: dict[str, str]) -> None:
        if value is None:
            return
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return
        self._samples.setdefault(name, []).append((labels, fval))

    def render(self) -> str:
        lines: list[str] = []
        for name in self._samples:
            lines.append(f"# HELP {name} {self._help.get(name, name)}")
            lines.append(f"# TYPE {name} gauge")
            for labels, value in self._samples[name]:
                lines.append(f"{name}{_fmt_labels(labels)} {value!r}")
        return "\n".join(lines) + "\n"


def collect(runs_dir: Path | None = None) -> _Registry:
    """Build a :class:`_Registry` snapshot from the runs directory."""
    root = runs_dir or _runs_root()
    reg = _Registry()
    reg.gauge("dongfeng_up", "Exporter liveness (always 1).")
    reg.gauge("dongfeng_runs_total", "Number of training runs discovered.")
    reg.gauge("dongfeng_train_loss", "Total loss per run and split.")
    reg.gauge("dongfeng_train_policy_loss", "Policy cross-entropy per run and split.")
    reg.gauge("dongfeng_train_value_loss", "Value MSE per run and split.")
    reg.gauge("dongfeng_train_top1", "Validation top-1 move accuracy per run.")
    reg.gauge("dongfeng_train_top5", "Validation top-5 move accuracy per run.")
    reg.gauge("dongfeng_train_lr", "Current learning rate per run.")
    reg.gauge("dongfeng_train_grad_norm", "Gradient norm per run.")
    reg.gauge("dongfeng_train_samples_per_s", "Training throughput in samples/s.")
    reg.gauge("dongfeng_train_tokens_per_s", "Training throughput in tokens/s.")
    reg.gauge("dongfeng_train_step", "Latest optimizer step per run.")
    reg.gauge("dongfeng_train_elapsed_seconds", "Wall-clock elapsed seconds per run.")
    reg.gauge("dongfeng_run_params", "Model parameter count per run.")
    reg.gauge("dongfeng_run_status", "Info gauge (=1) with run metadata as labels.")

    reg.add("dongfeng_up", 1, {})

    n_runs = 0
    if root.is_dir():
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            meta = _read_run_json(run_dir)
            if meta is None:
                continue
            n_runs += 1
            run_id = str(meta.get("id") or run_dir.name)
            last_train, last_val = _read_last_metrics(run_dir / "metrics.jsonl")

            reg.add("dongfeng_run_params", meta.get("params"), {"run": run_id})
            reg.add(
                "dongfeng_run_status",
                1,
                {
                    "run": run_id,
                    "preset": str(meta.get("preset") or ""),
                    "arch_hash": str(meta.get("arch_hash") or ""),
                    "device": str(meta.get("device") or ""),
                    "status": str(meta.get("status") or ""),
                },
            )

            if last_train is not None:
                lt = {"run": run_id, "split": "train"}
                reg.add("dongfeng_train_loss", last_train.get("loss"), lt)
                reg.add("dongfeng_train_policy_loss", last_train.get("policy_loss"), lt)
                reg.add("dongfeng_train_value_loss", last_train.get("value_loss"), lt)
                reg.add("dongfeng_train_lr", last_train.get("lr"), {"run": run_id})
                reg.add("dongfeng_train_grad_norm", last_train.get("grad_norm"), {"run": run_id})
                reg.add(
                    "dongfeng_train_samples_per_s",
                    last_train.get("samples_per_s"),
                    {"run": run_id},
                )
                reg.add(
                    "dongfeng_train_tokens_per_s",
                    last_train.get("tokens_per_s"),
                    {"run": run_id},
                )
                reg.add("dongfeng_train_step", last_train.get("step"), {"run": run_id})
                reg.add(
                    "dongfeng_train_elapsed_seconds",
                    last_train.get("elapsed_s"),
                    {"run": run_id},
                )

            if last_val is not None:
                lv = {"run": run_id, "split": "val"}
                reg.add("dongfeng_train_loss", last_val.get("loss"), lv)
                reg.add("dongfeng_train_policy_loss", last_val.get("policy_loss"), lv)
                reg.add("dongfeng_train_top1", last_val.get("top1"), {"run": run_id})
                reg.add("dongfeng_train_top5", last_val.get("top5"), {"run": run_id})

    reg.add("dongfeng_runs_total", n_runs, {})
    return reg


def render_exposition(runs_dir: Path | None = None) -> str:
    """Return the full Prometheus exposition text for the current runs snapshot."""
    return collect(runs_dir).render()


def _make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # quiet

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/metrics", "/"):
                body = render_exposition().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_error(404)

    return Handler


def serve_metrics(host: str = "0.0.0.0", port: int = 9105) -> None:  # noqa: S104
    """Start the exporter HTTP server (blocking) on ``host:port/metrics``.

    Binds to all interfaces by default so Prometheus (which may run in a container
    or on another host) can scrape it; restrict with ``--host 127.0.0.1`` when the
    scraper is co-located behind loopback.
    """
    httpd = ThreadingHTTPServer((host, port), _make_handler())
    print(
        f"Dong Feng metrics exporter on http://{host}:{port}/metrics — Ctrl+C to stop",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…", flush=True)
    finally:
        httpd.server_close()
