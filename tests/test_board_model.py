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


# Exact documented param counts for the default (n_bias_head=0) presets. These
# are byte-for-byte regression guards: WP-BIAS with n_bias_head=0 must not move
# them at all.
_EXPECTED_PARAMS = {
    "m1-dev": 22_277_088,
    "mid": 143_730_624,
    "1b": 1_023_905_664,
}


def test_num_params_m1dev() -> None:
    cfg = BoardTransformerConfig.presets()["m1-dev"]
    model = BoardTransformer(cfg)
    assert model.num_params() == _EXPECTED_PARAMS["m1-dev"], (
        f"m1-dev param count unexpected: {model.num_params():,}"
    )


def test_num_params_mid() -> None:
    cfg = BoardTransformerConfig.presets()["mid"]
    model = BoardTransformer(cfg)
    assert model.num_params() == _EXPECTED_PARAMS["mid"], (
        f"mid param count unexpected: {model.num_params():,}"
    )


def test_num_params_1b_meta() -> None:
    """Instantiate the 1B model on meta device — no real memory allocation."""
    cfg = BoardTransformerConfig.presets()["1b"]
    with torch.device("meta"):
        model = BoardTransformer(cfg)
    assert model.num_params() == _EXPECTED_PARAMS["1b"], (
        f"1b param count unexpected: {model.num_params():,}"
    )


def test_presets_default_n_bias_head_zero() -> None:
    """Presets keep n_bias_head=0 so default param counts stay stable."""
    for cfg in BoardTransformerConfig.presets().values():
        assert cfg.n_bias_head == 0


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


# ---------------------------------------------------------------------------
# WP-BIAS: additive per-head 2D relative-position bias (extra heads)
# ---------------------------------------------------------------------------

from dongfeng.model.board_transformer import (  # noqa: E402
    _BidirectionalAttention,
    _build_rel_index,
)


def _bias_cfg(preset: str, n_bias_head: int) -> BoardTransformerConfig:
    """A preset config with n_bias_head overridden (other fields unchanged)."""
    base = BoardTransformerConfig.presets()[preset]
    return BoardTransformerConfig(
        d_model=base.d_model,
        n_layer=base.n_layer,
        n_head=base.n_head,
        n_bias_head=n_bias_head,
        ffn_hidden=base.ffn_hidden,
        vocab_size=base.vocab_size,
        seq_len=base.seq_len,
        n_moves=base.n_moves,
        gradient_checkpointing=base.gradient_checkpointing,
    )


def test_bias_head_zero_is_byte_identical_param_count() -> None:
    """n_bias_head=0 must equal the exact documented preset counts (regression)."""
    for name, expected in _EXPECTED_PARAMS.items():
        cfg = _bias_cfg(name, 0)
        if name == "1b":
            with torch.device("meta"):
                model = BoardTransformer(cfg)
        else:
            model = BoardTransformer(cfg)
        assert model.num_params() == expected, f"{name} nbh=0 count moved: {model.num_params():,}"


@pytest.mark.parametrize("k", [1, 2, 4])
def test_bias_head_param_formula(k: int) -> None:
    """Extra params per layer = 4*d_model*k*head_dim + k*324, matching the plan."""
    base = BoardTransformerConfig.presets()["m1-dev"]
    head_dim = base.d_model // base.n_head
    m0 = BoardTransformer(_bias_cfg("m1-dev", 0))
    mk = BoardTransformer(_bias_cfg("m1-dev", k))
    per_layer = 4 * base.d_model * k * head_dim + k * 324
    expected = m0.num_params() + base.n_layer * per_layer
    assert mk.num_params() == expected, f"nbh={k}: got {mk.num_params():,}, expected {expected:,}"


def test_bias_head_forward_shapes_and_finite() -> None:
    """Adding bias heads leaves output shapes unchanged and outputs finite."""
    model = BoardTransformer(_tiny_cfg(n_bias_head=2))
    model.eval()
    b = 3
    policy, value = model(_boards(b))
    assert tuple(policy.shape) == (b, 2554)
    assert tuple(value.shape) == (b,)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()


def test_rel_bias_shape_zero_init_and_grad() -> None:
    """rel_bias is [n_bias_head, 324], zero at init, and gets a non-None grad."""
    model = BoardTransformer(_tiny_cfg(n_bias_head=3))
    attn = model.blocks[0].attn
    assert isinstance(attn, _BidirectionalAttention)
    assert tuple(attn.rel_bias.shape) == (3, 324)
    assert torch.count_nonzero(attn.rel_bias).item() == 0, "rel_bias must be zero-init"

    model.train()
    policy, value = model(_boards(2))
    loss = (
        torch.nn.functional.cross_entropy(policy, torch.zeros(2, dtype=torch.long))
        + value.pow(2).mean()
    )
    loss.backward()
    assert attn.rel_bias.grad is not None, "rel_bias received no gradient"


def test_rel_index_bounds_and_buckets() -> None:
    """rel_index values are in 0..323; board diagonal bucket = 161, side pairs = 323."""
    ri = _build_rel_index(91)
    assert tuple(ri.shape) == (91, 91)
    assert int(ri.min()) >= 0 and int(ri.max()) <= 323
    # diagonal (Δfile=0, Δrank=0) → (0+8)*19 + (0+9) = 161
    for i in (0, 45, 89):
        assert int(ri[i, i]) == 161, f"diagonal bucket wrong at {i}: {int(ri[i, i])}"
    # any pair touching the side-to-move token (index 90) → 323
    assert int(ri[90, 0]) == 323
    assert int(ri[0, 90]) == 323
    assert int(ri[90, 90]) == 323


def test_bias_head_1b_meta() -> None:
    """1b with n_bias_head=4 instantiates on the meta device."""
    with torch.device("meta"):
        model = BoardTransformer(_bias_cfg("1b", 4))
    # sanity: param count strictly larger than the content-only 1b
    assert model.num_params() > _EXPECTED_PARAMS["1b"]
