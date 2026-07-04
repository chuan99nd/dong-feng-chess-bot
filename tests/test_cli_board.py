"""Tests for WP5 CLI wiring: dfc data ingest-board, dfc train-board."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dongfeng.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixture: minimal PGN (same as tests/test_ingest.py)
# ---------------------------------------------------------------------------

_SAMPLE_PGN = """[Game "China Chess"]
[Event "Test Event"]
[Red "Red Player"]
[Black "Black Player"]
[Result "1-0"]
1. 炮二平五 马8进7
2. 马二进三 车9平8
3. 车一平二 炮8进4
"""


@pytest.fixture()
def pgn_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pgn"
    p.write_text(_SAMPLE_PGN, encoding="utf-8")
    return p


@pytest.fixture()
def manifest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DONGFENG_MANIFEST to a fresh temp file."""
    m = tmp_path / "manifest.json"
    monkeypatch.setenv("DONGFENG_MANIFEST", str(m))
    return m


# ---------------------------------------------------------------------------
# ingest-board
# ---------------------------------------------------------------------------


def test_ingest_board_creates_shards(pgn_file: Path, tmp_path: Path, manifest_env: Path) -> None:
    """ingest-board writes board_meta.json + shard files and updates manifest."""
    out_dir = tmp_path / "board_ds"
    result = runner.invoke(
        app,
        [
            "data",
            "ingest-board",
            str(pgn_file),
            "--out",
            str(out_dir),
            "--id",
            "test-board-ds",
        ],
    )
    assert result.exit_code == 0, result.output

    # §1.3 — board_meta.json must exist.
    meta_path = out_dir / "board_meta.json"
    assert meta_path.exists(), "board_meta.json not created"
    meta = json.loads(meta_path.read_text())
    assert meta["schema"] == "board-ds-v1"
    assert meta["tokenizer"] == "board-v1"
    assert meta["move_tokenizer"] == "move-v1"
    assert meta["num_samples"] > 0
    assert meta["num_games"] >= 1
    assert isinstance(meta["shards"], list) and len(meta["shards"]) > 0

    # At least one set of shard files exists.
    stem = meta["shards"][0]
    assert (out_dir / f"boards_{stem}.bin").exists()
    assert (out_dir / f"moves_{stem}.bin").exists()
    assert (out_dir / f"values_{stem}.bin").exists()


def test_ingest_board_manifest_entry(pgn_file: Path, tmp_path: Path, manifest_env: Path) -> None:
    """ingest-board creates a manifest dataset entry with board-v1 tokenizer."""
    out_dir = tmp_path / "board_ds2"
    runner.invoke(
        app,
        [
            "data",
            "ingest-board",
            str(pgn_file),
            "--out",
            str(out_dir),
            "--id",
            "board-ds-manifest",
        ],
    )
    manifest = json.loads(manifest_env.read_text())
    datasets = manifest.get("datasets", [])
    entry = next((d for d in datasets if d.get("id") == "board-ds-manifest"), None)
    assert entry is not None, "dataset entry not found in manifest"
    assert entry["tokenizer"] == "board-v1"
    assert "move-v1" in entry.get("notes", "")
    assert entry["format"] == "board-ds-v1"

    # board-v1 tokenizer registered in manifest.
    tokenizer_ids = [t["id"] for t in manifest.get("tokenizers", [])]
    assert "board-v1" in tokenizer_ids


def test_data_stats_shows_board_dataset(pgn_file: Path, tmp_path: Path, manifest_env: Path) -> None:
    """dfc data stats shows the board dataset after ingest-board."""
    out_dir = tmp_path / "board_ds3"
    runner.invoke(
        app,
        [
            "data",
            "ingest-board",
            str(pgn_file),
            "--out",
            str(out_dir),
            "--id",
            "stats-test-ds",
        ],
    )
    result = runner.invoke(app, ["data", "stats"])
    assert result.exit_code == 0
    assert "stats-test-ds" in result.output


# ---------------------------------------------------------------------------
# train-board --smoke (torch-gated)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not installed")


def test_train_board_smoke_exits_zero() -> None:
    """train-board --smoke builds model, runs 2 steps, prints param count, exits 0."""
    result = runner.invoke(app, ["train-board", "--smoke", "--preset", "m1-dev"])
    assert result.exit_code == 0, result.output
    # Param count must appear in output.
    output = result.output
    # Should contain digits (param count).
    import re

    assert re.search(r"\d{4,}", output), f"param count not found in output:\n{output}"
    assert "Smoke test" in output or "params" in output.lower()


# ---------------------------------------------------------------------------
# train-board normal run (torch-gated, tiny override)
# ---------------------------------------------------------------------------


def test_train_board_creates_run_artifacts(
    pgn_file: Path, tmp_path: Path, manifest_env: Path
) -> None:
    """train-board with tiny config creates run.json, metrics.jsonl, ckpt.pt."""

    from dongfeng.data import build_board_shards, iter_games_in  # noqa: PLC0415
    from dongfeng.model.board_transformer import BoardTransformerConfig  # noqa: PLC0415
    from dongfeng.training.board_loop import BoardTrainConfig, bc_train_board  # noqa: PLC0415

    # Build a tiny board dataset first.
    ds_dir = tmp_path / "tiny_board_ds"
    build_board_shards(list(iter_games_in(str(pgn_file))), ds_dir, shard_size=10_000)

    run_dir = tmp_path / "tiny_run"
    # Use a tiny config_override so the test is fast (2 steps, batch 2).
    tiny_cfg = BoardTransformerConfig(d_model=32, n_layer=1, n_head=2, ffn_hidden=64)
    tcfg = BoardTrainConfig(
        data_dir=ds_dir,
        out_dir=run_dir,
        preset="m1-dev",
        id="tiny-run",
        batch_size=2,
        max_steps=3,
        warmup=1,
        eval_every=2,
        device="cpu",
        seed=0,
        config_override=tiny_cfg,
    )
    ckpt = bc_train_board(tcfg)

    # §1.4 — required files must exist.
    assert (run_dir / "run.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert ckpt.exists()

    run_data = json.loads((run_dir / "run.json").read_text())
    assert run_data["status"] == "done"
    assert run_data["kind"] == "bc-board"

    # metrics.jsonl must have at least one line with required keys.
    lines = [
        json.loads(ln) for ln in (run_dir / "metrics.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert lines, "metrics.jsonl is empty"
    row = lines[0]
    for key in ("step", "split", "loss", "policy_loss", "lr", "samples_per_s", "elapsed_s"):
        assert key in row, f"missing key {key!r} in metrics row"
