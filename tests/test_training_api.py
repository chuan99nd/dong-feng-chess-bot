"""Tests for the /api/training endpoints in serve/webplay.py (WP6)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from dongfeng.serve.webplay import GameSession, _make_handler

# ---------------------------------------------------------------------------
# Helpers to build a synthetic runs/ tree
# ---------------------------------------------------------------------------


def _make_run(
    runs_root: Path,
    run_id: str,
    *,
    status: str = "done",
    preset: str = "m1-dev",
    params: int = 21_000_000,
    num_train: int = 10,
    num_val: int = 5,
) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = {
        "id": run_id,
        "kind": "bc-board",
        "preset": preset,
        "params": params,
        "device": "cpu",
        "dtype": "float32",
        "data_dir": "data/board_ds",
        "started": "2024-01-01T00:00:00",
        "finished": "2024-01-01T01:00:00" if status == "done" else None,
        "status": status,
        "config": {"batch_size": 32, "lr": 1e-4},
    }
    (run_dir / "run.json").write_text(json.dumps(run_json))
    # Write metrics.jsonl
    lines: list[str] = []
    for i in range(num_train):
        row = {
            "step": i + 1,
            "split": "train",
            "loss": 3.0 - i * 0.1,
            "policy_loss": 2.8 - i * 0.1,
            "value_loss": 0.5,
            "top1": 0.1 + i * 0.01,
            "lr": 1e-4,
            "samples_per_s": 200.0,
            "elapsed_s": float(i * 10),
        }
        lines.append(json.dumps(row))
    for j in range(num_val):
        row = {
            "step": (j + 1) * 2,
            "split": "val",
            "loss": 2.9 - j * 0.1,
            "policy_loss": 2.7 - j * 0.1,
            "value_loss": 0.4,
            "top1": 0.12 + j * 0.01,
            "lr": 1e-4,
            "samples_per_s": 0.0,
            "elapsed_s": float(j * 20),
        }
        lines.append(json.dumps(row))
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")


def _make_large_run(  # noqa: E501
    runs_root: Path, run_id: str, num_train: int = 5000, num_val: int = 1000
) -> None:
    """A run with many metric lines to test downsampling."""
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = {
        "id": run_id,
        "kind": "bc-board",
        "preset": "mid",
        "params": 142_000_000,
        "device": "cuda",
        "dtype": "bfloat16",
        "data_dir": "data/board_ds",
        "started": "2024-02-01T00:00:00",
        "finished": None,
        "status": "running",
        "config": {},
    }
    (run_dir / "run.json").write_text(json.dumps(run_json))
    with open(run_dir / "metrics.jsonl", "w") as f:
        for i in range(num_train):
            row = {
                "step": i,
                "split": "train",
                "loss": 3.0 - i * 0.0001,
                "policy_loss": 2.8,
                "value_loss": 0.5,
                "top1": 0.1,
                "lr": 1e-4,
                "samples_per_s": 500.0,
                "elapsed_s": float(i),
            }
            f.write(json.dumps(row) + "\n")
        for j in range(num_val):
            row = {
                "step": j * 5,
                "split": "val",
                "loss": 2.9 - j * 0.0002,
                "policy_loss": 2.7,
                "value_loss": 0.4,
                "top1": 0.12,
                "lr": 1e-4,
                "samples_per_s": 0.0,
                "elapsed_s": float(j * 50),
            }
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Fixture: ephemeral HTTP server + synthetic runs/ tree
# ---------------------------------------------------------------------------


class _Server:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root
        session = GameSession("random", None)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(session))
        self.port: int = self._httpd.server_address[1]  # type: ignore[index]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def get(self, path: str) -> tuple[int, dict]:  # type: ignore[type-arg]
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get_html(self) -> tuple[int, str]:
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as resp:
            return resp.status, resp.read().decode()

    def close(self) -> None:
        self._httpd.shutdown()


@pytest.fixture()
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[type-arg]
    """Start an ephemeral server pointing at a synthetic runs/ tree."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("DONGFENG_RUNS_DIR", str(runs_root))
    _make_run(runs_root, "run-001", status="done", num_train=10, num_val=5)
    _make_run(runs_root, "run-002", status="running", preset="mid", num_train=8, num_val=3)
    srv = _Server(runs_root)
    yield srv, runs_root
    srv.close()


@pytest.fixture()
def server_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[type-arg]
    """Server with a large run for downsampling tests."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("DONGFENG_RUNS_DIR", str(runs_root))
    _make_large_run(runs_root, "big-run", num_train=5000, num_val=1000)
    srv = _Server(runs_root)
    yield srv, runs_root
    srv.close()


# ---------------------------------------------------------------------------
# Tests: list endpoint
# ---------------------------------------------------------------------------


def test_list_returns_runs(server):  # type: ignore[no-untyped-def]
    srv, _ = server
    status, data = srv.get("/api/training")
    assert status == 200
    runs = data["runs"]
    assert len(runs) == 2
    ids = {r["id"] for r in runs}
    assert ids == {"run-001", "run-002"}


def test_list_has_last_train_val(server):  # type: ignore[no-untyped-def]
    srv, _ = server
    status, data = srv.get("/api/training")
    assert status == 200
    by_id = {r["id"]: r for r in data["runs"]}
    r1 = by_id["run-001"]
    assert r1["last_train"] is not None
    assert r1["last_train"]["split"] == "train"
    assert r1["last_val"] is not None
    assert r1["last_val"]["split"] == "val"


def test_list_newest_first(server):  # type: ignore[no-untyped-def]
    """Runs sorted by 'started' descending — run-002 has a later started value."""
    srv, runs_root = server
    # Update run-002 started to be clearly later
    run2_path = runs_root / "run-002" / "run.json"
    d = json.loads(run2_path.read_text())
    d["started"] = "2024-06-01T00:00:00"
    run2_path.write_text(json.dumps(d))
    status, data = srv.get("/api/training")
    assert status == 200
    runs = data["runs"]
    assert runs[0]["id"] == "run-002"


def test_list_empty_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty runs dir returns empty list without error."""
    runs_root = tmp_path / "runs_empty"
    runs_root.mkdir()
    monkeypatch.setenv("DONGFENG_RUNS_DIR", str(runs_root))
    session = GameSession("random", None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(session))
    port: int = httpd.server_address[1]  # type: ignore[index]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/api/training"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
        assert data["runs"] == []
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Tests: detail endpoint
# ---------------------------------------------------------------------------


def test_detail_returns_run_and_metrics(server):  # type: ignore[no-untyped-def]
    srv, _ = server
    status, data = srv.get("/api/training?id=run-001")
    assert status == 200
    assert "run" in data
    assert "metrics" in data
    assert data["run"]["id"] == "run-001"
    assert isinstance(data["metrics"], list)


def test_detail_metrics_have_both_splits(server):  # type: ignore[no-untyped-def]
    srv, _ = server
    _, data = srv.get("/api/training?id=run-001")
    splits = {m["split"] for m in data["metrics"]}
    assert "train" in splits
    assert "val" in splits


def test_detail_unknown_id_returns_error(server):  # type: ignore[no-untyped-def]
    srv, _ = server
    status, data = srv.get("/api/training?id=does-not-exist")
    assert status != 200
    assert "error" in data


# ---------------------------------------------------------------------------
# Tests: downsampling
# ---------------------------------------------------------------------------


def test_detail_downsampling_train(server_large):  # type: ignore[no-untyped-def]
    """5000 train lines → at most 500 train metrics in response."""
    srv, _ = server_large
    status, data = srv.get("/api/training?id=big-run")
    assert status == 200
    train_metrics = [m for m in data["metrics"] if m["split"] == "train"]
    assert len(train_metrics) <= 500


def test_detail_downsampling_val(server_large):  # type: ignore[no-untyped-def]
    """1000 val lines → at most 500 val metrics in response."""
    srv, _ = server_large
    status, data = srv.get("/api/training?id=big-run")
    assert status == 200
    val_metrics = [m for m in data["metrics"] if m["split"] == "val"]
    assert len(val_metrics) <= 500


def test_detail_small_run_not_downsampled(server):  # type: ignore[no-untyped-def]
    """Small run with 10 train + 5 val lines → all returned, no downsampling loss."""
    srv, _ = server
    _, data = srv.get("/api/training?id=run-001")
    train_metrics = [m for m in data["metrics"] if m["split"] == "train"]
    val_metrics = [m for m in data["metrics"] if m["split"] == "val"]
    assert len(train_metrics) == 10
    assert len(val_metrics) == 5


# ---------------------------------------------------------------------------
# Tests: HTML contains Training panel markup
# ---------------------------------------------------------------------------


def test_html_contains_training_panel(server):  # type: ignore[no-untyped-def]
    srv, _ = server
    status, html = srv.get_html()
    assert status == 200
    # The Training tab button and panel must be present
    assert "tab-training" in html
    assert "training-run-select" in html
    assert "training-chart" in html
    assert "Training" in html


def test_html_board_play_still_works(server):  # type: ignore[no-untyped-def]
    """Ensure the board play tab markup is still present after adding Training."""
    srv, _ = server
    _, html = srv.get_html()
    assert "tab-play" in html
    assert 'id="board"' in html
    assert "/api/move" in html


# ---------------------------------------------------------------------------
# Tests: runs dir nonexistent is handled gracefully
# ---------------------------------------------------------------------------


def test_list_runs_dir_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DONGFENG_RUNS_DIR pointing at a nonexistent dir returns empty list."""
    monkeypatch.setenv("DONGFENG_RUNS_DIR", str(tmp_path / "no_such_dir"))
    session = GameSession("random", None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(session))
    port: int = httpd.server_address[1]  # type: ignore[index]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/api/training"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())
        assert data["runs"] == []
    finally:
        httpd.shutdown()
