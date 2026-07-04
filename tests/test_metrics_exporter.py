"""Tests for the Prometheus training-metrics exporter."""

from __future__ import annotations

import json
from pathlib import Path

from dongfeng.serve.metrics_exporter import collect, render_exposition


def _write_run(runs_dir: Path, run_id: str, rows: list[dict]) -> None:
    d = runs_dir / run_id
    d.mkdir(parents=True)
    meta = {
        "id": run_id,
        "kind": "bc-board",
        "preset": "mid",
        "arch_hash": "abc123",
        "params": 42_000_000,
        "device": "cuda",
        "status": "running",
    }
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with open(d / "metrics.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_render_empty_runs_dir(tmp_path: Path) -> None:
    text = render_exposition(tmp_path)
    assert "dongfeng_up 1.0" in text
    assert "dongfeng_runs_total 0.0" in text
    # Well-formed exposition: every metric has HELP + TYPE.
    assert "# TYPE dongfeng_up gauge" in text


def test_collect_reports_latest_train_and_val(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "board-5090-run1",
        rows=[
            {
                "step": 0,
                "split": "train",
                "loss": 6.0,
                "policy_loss": 5.9,
                "value_loss": 0.1,
                "lr": 1e-4,
                "grad_norm": 2.0,
                "samples_per_s": 100.0,
                "tokens_per_s": 9100.0,
                "elapsed_s": 1.0,
            },
            {
                "step": 99,
                "split": "train",
                "loss": 3.0,
                "policy_loss": 2.9,
                "value_loss": 0.05,
                "lr": 5e-4,
                "grad_norm": 1.2,
                "samples_per_s": 200.0,
                "tokens_per_s": 18200.0,
                "elapsed_s": 50.0,
            },
            {
                "step": 99,
                "split": "val",
                "loss": 3.2,
                "policy_loss": 3.1,
                "value_loss": None,
                "top1": 0.42,
                "top5": 0.78,
                "lr": 5e-4,
                "grad_norm": 1.2,
                "elapsed_s": 51.0,
            },
        ],
    )
    text = render_exposition(tmp_path)

    # Latest train step (99), not the first (0).
    assert 'dongfeng_train_step{run="board-5090-run1"} 99.0' in text
    assert 'dongfeng_train_loss{run="board-5090-run1",split="train"} 3.0' in text
    assert 'dongfeng_train_loss{run="board-5090-run1",split="val"} 3.2' in text
    assert 'dongfeng_train_top1{run="board-5090-run1"} 0.42' in text
    assert 'dongfeng_train_top5{run="board-5090-run1"} 0.78' in text
    assert 'dongfeng_run_params{run="board-5090-run1"} 42000000.0' in text
    assert 'status="running"' in text
    assert "dongfeng_runs_total 1.0" in text


def test_none_values_are_skipped(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "run-none",
        rows=[
            {
                "step": 10,
                "split": "val",
                "loss": 1.0,
                "policy_loss": 1.0,
                "top1": None,
                "top5": None,
            }
        ],
    )
    reg = collect(tmp_path)
    text = reg.render()
    # top1/top5 are None -> not emitted.
    assert "dongfeng_train_top1{" not in text
    assert 'dongfeng_train_loss{run="run-none",split="val"} 1.0' in text


def test_label_values_are_escaped(tmp_path: Path) -> None:
    d = tmp_path / "weird"
    d.mkdir()
    (d / "run.json").write_text(
        json.dumps({"id": 'we"ird', "preset": "a\\b", "status": "running"}),
        encoding="utf-8",
    )
    (d / "metrics.jsonl").write_text("", encoding="utf-8")
    text = render_exposition(tmp_path)
    assert 'run="we\\"ird"' in text
    assert 'preset="a\\\\b"' in text
