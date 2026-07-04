"""Tests for the bias-head diversity diagnostics (WP-BIAS). Skipped without torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dongfeng.model.bias_diagnostics import (  # noqa: E402
    _attention_maps_from_input,
    format_report,
    head_diversity,
)
from dongfeng.model.board_transformer import BoardTransformer, BoardTransformerConfig  # noqa: E402


def _cfg(n_bias_head: int) -> BoardTransformerConfig:
    return BoardTransformerConfig(
        d_model=32,
        n_layer=2,
        n_head=4,
        n_bias_head=n_bias_head,
        ffn_hidden=64,
        vocab_size=21,
        seq_len=91,
        n_moves=2554,
    )


def _boards(batch: int = 3) -> torch.Tensor:
    # A spread of valid board-v1 ids (empty..side tokens) so attention is non-trivial.
    g = torch.Generator().manual_seed(0)
    return torch.randint(4, 21, (batch, 91), generator=g)


def test_attention_maps_are_probabilities() -> None:
    """Recomputed attention maps are non-negative and sum to 1 over keys."""
    model = BoardTransformer(_cfg(2))
    model.eval()
    attn = model.blocks[0].attn
    x = model.tok_emb(_boards(3)) + model._positional_embeddings(torch.device("cpu")).unsqueeze(0)
    maps = _attention_maps_from_input(attn, x)  # [total_heads, T, T]
    assert tuple(maps.shape) == (attn.total_heads, 91, 91)
    assert (maps >= 0).all()
    row_sums = maps.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_head_diversity_shapes_with_bias() -> None:
    model = BoardTransformer(_cfg(3))
    diags = head_diversity(model, _boards(4))
    assert len(diags) == model.config.n_layer
    for d in diags:
        assert d.n_head == 4
        assert d.n_bias_head == 3
        assert len(d.rel_bias_norms) == 3
        assert d.content_bias_similarity is not None
        assert -1.0 <= d.mean_head_similarity <= 1.0 + 1e-5


def test_zero_init_rel_bias_has_zero_norms() -> None:
    """At init, rel_bias is zero, so geometry-usage norms are all 0."""
    model = BoardTransformer(_cfg(2))
    diags = head_diversity(model, _boards(2))
    for d in diags:
        assert all(n == 0.0 for n in d.rel_bias_norms)


def test_trained_rel_bias_shows_nonzero_norm() -> None:
    """After a gradient step that moves rel_bias, its norm becomes > 0 (geometry used)."""
    model = BoardTransformer(_cfg(2))
    # Nudge rel_bias off zero to simulate training having used the geometry.
    with torch.no_grad():
        for block in model.blocks:
            block.attn.rel_bias.add_(0.1)
    diags = head_diversity(model, _boards(2))
    assert all(all(n > 0.0 for n in d.rel_bias_norms) for d in diags)


def test_no_bias_head_has_no_norms() -> None:
    """n_bias_head=0 => no rel_bias, no content-vs-bias similarity."""
    model = BoardTransformer(_cfg(0))
    diags = head_diversity(model, _boards(2))
    for d in diags:
        assert d.n_bias_head == 0
        assert d.rel_bias_norms == []
        assert d.content_bias_similarity is None


def test_format_report_smoke() -> None:
    model = BoardTransformer(_cfg(2))
    report = format_report(head_diversity(model, _boards(2)))
    assert "head_sim" in report
    assert "rel_bias" in report
