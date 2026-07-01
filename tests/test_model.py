"""Tests for the decoder-only transformer policy (M2). Skipped without torch."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dongfeng.model import TransformerConfig, TransformerPolicy  # noqa: E402
from dongfeng.tokenizer import MoveTokenizer  # noqa: E402


def _tiny() -> TransformerPolicy:
    return TransformerPolicy(
        TransformerConfig(vocab_size=MoveTokenizer().vocab_size, n_layer=2, n_embd=64, n_head=2)
    )


def test_forward_shape() -> None:
    model = _tiny()
    x = torch.zeros((3, 12), dtype=torch.long)
    logits = model(x)
    assert tuple(logits.shape) == (3, 12, model.config.vocab_size)


def test_num_params_positive() -> None:
    assert _tiny().num_params() > 0


def test_value_head_absent_by_default() -> None:
    model = _tiny()
    assert model.value(torch.zeros((1, 4), dtype=torch.long)) is None


def test_value_head_present_when_enabled() -> None:
    model = TransformerPolicy(
        TransformerConfig(vocab_size=100, n_layer=1, n_embd=32, n_head=2, with_value_head=True)
    )
    out = model.value(torch.zeros((2, 5), dtype=torch.long))
    assert out is not None
    assert tuple(out.shape) == (2, 5)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = _tiny()
    p = tmp_path / "ckpt.pt"
    model.save(p, extra={"step": 7})
    loaded, extra = TransformerPolicy.load(p)
    assert extra["step"] == 7
    x = torch.zeros((1, 6), dtype=torch.long)
    assert torch.allclose(model(x), loaded(x), atol=1e-5)
