"""Tests for the board-state transformer (WP2 of board-1b plan). Skipped without torch."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dongfeng.model.board_transformer import BoardTransformer, BoardTransformerConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_cfg(**overrides: object) -> BoardTransformerConfig:
    """A very small config for fast CPU tests."""
    return BoardTransformerConfig(
        d_model=32,
        n_layer=2,
        n_head=4,
        ffn_hidden=64,
        vocab_size=21,
        seq_len=91,
        n_moves=2554,
        **overrides,  # type: ignore[arg-type]
    )


def _boards(batch: int = 2) -> torch.Tensor:
    """Synthetic [B, 91] board token ids (all zeros — PAD id, valid for shape tests)."""
    return torch.zeros((batch, 91), dtype=torch.long)


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------


def test_forward_shapes() -> None:
    model = BoardTransformer(_tiny_cfg())
    model.eval()
    b = 3
    policy, value = model(_boards(b))
    assert tuple(policy.shape) == (b, 2554), f"policy shape wrong: {policy.shape}"
    assert tuple(value.shape) == (b,), f"value shape wrong: {value.shape}"


def test_value_range() -> None:
    """Value head output must be strictly in (−1, 1) via tanh."""
    model = BoardTransformer(_tiny_cfg())
    model.eval()
    _, value = model(_boards(4))
    assert (value > -1.0).all() and (value < 1.0).all(), f"value out of range: {value}"


# ---------------------------------------------------------------------------
# Parameter count tests (§1.1 presets)
# ---------------------------------------------------------------------------


def test_num_params_m1dev() -> None:
    cfg = BoardTransformerConfig.presets()["m1-dev"]
    model = BoardTransformer(cfg)
    params = model.num_params()
    # ~21M ± 10%
    assert 0.9 * 21e6 < params < 1.1 * 21e6, f"m1-dev param count unexpected: {params:,}"


def test_num_params_mid() -> None:
    cfg = BoardTransformerConfig.presets()["mid"]
    model = BoardTransformer(cfg)
    params = model.num_params()
    # ~142M ± 10%
    assert 0.9 * 142e6 < params < 1.1 * 142e6, f"mid param count unexpected: {params:,}"


def test_num_params_1b_meta() -> None:
    """Instantiate the 1B model on meta device — no real memory allocation."""
    cfg = BoardTransformerConfig.presets()["1b"]
    with torch.device("meta"):
        model = BoardTransformer(cfg)
    params = model.num_params()
    # ~1.02B ± 10% (plan says 0.9e9 < p < 1.15e9)
    assert 0.9e9 < params < 1.15e9, f"1b param count unexpected: {params:,}"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = BoardTransformer(_tiny_cfg())
    model.eval()
    boards = _boards(2)
    policy_before, value_before = model(boards)

    ckpt = tmp_path / "board_model.pt"
    model.save(ckpt, extra={"step": 42, "note": "test"})
    loaded, extra = BoardTransformer.load(ckpt, map_location="cpu")
    loaded.eval()

    assert extra["step"] == 42
    assert extra["note"] == "test"

    policy_after, value_after = loaded(boards)
    assert torch.allclose(policy_before, policy_after, atol=1e-5), "policy mismatch after reload"
    assert torch.allclose(value_before, value_after, atol=1e-5), "value mismatch after reload"


# ---------------------------------------------------------------------------
# Gradient checkpointing: on vs off produces same output
# ---------------------------------------------------------------------------


def test_grad_checkpoint_same_loss() -> None:
    """Gradient checkpointing should not change forward values (atol 1e-4)."""
    boards = _boards(2)
    targets = torch.zeros(2, dtype=torch.long)  # dummy move targets

    def _run(grad_ckpt: bool) -> float:
        torch.manual_seed(0)
        cfg = _tiny_cfg(gradient_checkpointing=grad_ckpt)
        model = BoardTransformer(cfg)
        model.train()
        policy, _ = model(boards)
        loss = torch.nn.functional.cross_entropy(policy, targets)
        return loss.item()

    loss_off = _run(False)
    loss_on = _run(True)
    assert abs(loss_off - loss_on) < 1e-4, (
        f"grad checkpointing changes loss: off={loss_off:.6f} on={loss_on:.6f}"
    )


# ---------------------------------------------------------------------------
# Backward pass runs on CPU
# ---------------------------------------------------------------------------


def test_backward_runs() -> None:
    """A full forward + backward should complete without errors on CPU."""
    cfg = _tiny_cfg()
    model = BoardTransformer(cfg)
    model.train()
    boards = _boards(2)
    policy, value = model(boards)
    # Dummy targets
    move_targets = torch.zeros(2, dtype=torch.long)
    value_targets = torch.zeros(2)
    loss = torch.nn.functional.cross_entropy(policy, move_targets) + torch.nn.functional.mse_loss(
        value, value_targets
    )
    loss.backward()
    # Check at least one gradient is non-None and non-zero
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert has_grad, "no non-zero gradients after backward"
