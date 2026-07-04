"""Tests for the board-state BC training loop (WP3 / M3.5).

All tests skip when torch is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dongfeng.core import STARTING_FEN, GameResult, Move
from dongfeng.data import Game, build_board_shards

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")

# Required §1.4 metric keys.
_METRIC_KEYS = {
    "step",
    "split",
    "loss",
    "policy_loss",
    "value_loss",
    "top1",
    "lr",
    "samples_per_s",
    "elapsed_s",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_games(n: int = 4, result: GameResult = GameResult.RED_WIN) -> list[Game]:
    """Build *n* short synthetic games (4 plies each)."""
    moves = [Move.from_iccs(s) for s in ("h2e2", "h9g7", "h0g2", "i9h9")]
    return [Game(start_fen=STARTING_FEN, moves=list(moves), result=result) for _ in range(n)]


@pytest.fixture()
def board_data_dir(tmp_path: Path) -> Path:
    """Build a small board shard dataset and return its directory."""
    out = tmp_path / "board_ds"
    build_board_shards(_make_games(4), out)
    return out


# ---------------------------------------------------------------------------
# Tiny model config for fast CPU tests
# ---------------------------------------------------------------------------


def _tiny_model_config():  # type: ignore[return]
    """Return a very small BoardTransformerConfig for fast CPU tests."""
    from dongfeng.model.board_transformer import BoardTransformerConfig

    return BoardTransformerConfig(
        d_model=32,
        n_layer=2,
        n_head=4,
        ffn_hidden=64,
        vocab_size=21,
        seq_len=91,
        n_moves=2554,
        gradient_checkpointing=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bc_train_board_loss_decreases(tmp_path: Path, board_data_dir: Path) -> None:
    """Loss at last train step should be < loss at step 0 (learning signal)."""
    from dongfeng.training.board_loop import BoardTrainConfig, bc_train_board

    cfg = BoardTrainConfig(
        data_dir=board_data_dir,
        out_dir=tmp_path / "run1",
        preset="m1-dev",
        id="test-run-1",
        batch_size=8,
        lr=1e-3,
        warmup=2,
        max_steps=20,
        value_weight=0.5,
        device="cpu",
        seed=42,
        grad_checkpoint=False,
        eval_every=10,
        config_override=_tiny_model_config(),
    )
    ckpt_path = bc_train_board(cfg)
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    # Parse metrics.jsonl and extract train losses.
    metrics_path = tmp_path / "run1" / "metrics.jsonl"
    assert metrics_path.exists(), "metrics.jsonl not written"

    raw_lines = metrics_path.read_text().splitlines()
    lines = [json.loads(raw) for raw in raw_lines if raw.strip()]
    train_lines = [row for row in lines if row["split"] == "train"]
    assert len(train_lines) >= 2, "Expected at least 2 train metric entries"

    first_loss = train_lines[0]["loss"]
    last_loss = train_lines[-1]["loss"]
    assert last_loss < first_loss, (
        f"Expected loss to decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )


def test_metrics_jsonl_keys(tmp_path: Path, board_data_dir: Path) -> None:
    """Every line in metrics.jsonl must have exactly the §1.4 keys."""
    from dongfeng.training.board_loop import BoardTrainConfig, bc_train_board

    cfg = BoardTrainConfig(
        data_dir=board_data_dir,
        out_dir=tmp_path / "run2",
        preset="m1-dev",
        id="test-run-2",
        batch_size=8,
        lr=1e-3,
        warmup=1,
        max_steps=20,
        device="cpu",
        seed=0,
        eval_every=10,
        config_override=_tiny_model_config(),
    )
    bc_train_board(cfg)

    metrics_path = tmp_path / "run2" / "metrics.jsonl"
    raw_lines = metrics_path.read_text().splitlines()
    lines = [json.loads(raw) for raw in raw_lines if raw.strip()]
    assert len(lines) > 0, "metrics.jsonl is empty"

    for i, row in enumerate(lines):
        missing = _METRIC_KEYS - row.keys()
        assert not missing, f"Line {i} missing keys: {missing} — row: {row}"


def test_run_json_status_done(tmp_path: Path, board_data_dir: Path) -> None:
    """run.json must end with status == 'done'."""
    from dongfeng.training.board_loop import BoardTrainConfig, bc_train_board

    cfg = BoardTrainConfig(
        data_dir=board_data_dir,
        out_dir=tmp_path / "run3",
        preset="m1-dev",
        id="test-run-3",
        batch_size=8,
        lr=1e-3,
        warmup=1,
        max_steps=20,
        device="cpu",
        seed=7,
        eval_every=10,
        config_override=_tiny_model_config(),
    )
    bc_train_board(cfg)

    run_json_path = tmp_path / "run3" / "run.json"
    assert run_json_path.exists(), "run.json not written"
    meta = json.loads(run_json_path.read_text())

    assert meta["status"] == "done", f"Expected status='done', got {meta['status']!r}"
    assert meta["id"] == "test-run-3"
    assert meta["kind"] == "bc-board"
    assert meta["finished"] is not None
    assert meta["started"] is not None


def test_ckpt_loads_and_forwards(tmp_path: Path, board_data_dir: Path) -> None:
    """ckpt.pt must load via BoardTransformer.load and produce correct output shapes."""
    import torch

    from dongfeng.model.board_transformer import BoardTransformer
    from dongfeng.training.board_loop import BoardTrainConfig, bc_train_board

    cfg = BoardTrainConfig(
        data_dir=board_data_dir,
        out_dir=tmp_path / "run4",
        preset="m1-dev",
        id="test-run-4",
        batch_size=8,
        lr=1e-3,
        warmup=1,
        max_steps=20,
        device="cpu",
        seed=3,
        eval_every=10,
        config_override=_tiny_model_config(),
    )
    ckpt_path = bc_train_board(cfg)
    assert ckpt_path.exists()

    model, _extra = BoardTransformer.load(ckpt_path, map_location="cpu")
    assert isinstance(model, BoardTransformer)

    # Forward pass smoke test.
    dummy = torch.zeros(2, 91, dtype=torch.long)
    with torch.no_grad():
        policy_logits, value = model(dummy)

    assert policy_logits.shape == (2, 2554), f"Unexpected policy shape: {policy_logits.shape}"
    assert value.shape == (2,), f"Unexpected value shape: {value.shape}"
    # Value head uses tanh → must be in (−1, 1).
    assert (value.abs() <= 1.0).all(), "Value head output outside (−1, 1)"


def test_run_json_required_fields(tmp_path: Path, board_data_dir: Path) -> None:
    """run.json must contain all §1.4 required fields."""
    from dongfeng.training.board_loop import BoardTrainConfig, bc_train_board

    cfg = BoardTrainConfig(
        data_dir=board_data_dir,
        out_dir=tmp_path / "run5",
        preset="m1-dev",
        id="test-run-5",
        batch_size=8,
        lr=1e-3,
        warmup=1,
        max_steps=20,
        device="cpu",
        seed=0,
        eval_every=10,
        config_override=_tiny_model_config(),
    )
    bc_train_board(cfg)

    meta = json.loads((tmp_path / "run5" / "run.json").read_text())
    required = {
        "id",
        "kind",
        "preset",
        "params",
        "device",
        "dtype",
        "data_dir",
        "started",
        "finished",
        "status",
        "config",
    }
    missing = required - meta.keys()
    assert not missing, f"run.json missing fields: {missing}"
    assert isinstance(meta["params"], int) and meta["params"] > 0
