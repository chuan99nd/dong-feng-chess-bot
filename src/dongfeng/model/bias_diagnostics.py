"""Diagnostics for the additive 2D relative-position bias heads (WP-BIAS).

The bias heads only earn their extra parameters if, after training, they
(a) attend *differently* from the content heads (they are not redundant copies)
and (b) actually *use* the geometry table (``rel_bias`` moves away from its
zero init). This module measures exactly that, so an A/B/C ablation that holds
the model *size* fixed and varies only ``n_bias_head`` can be judged on evidence
rather than a single val-loss number.

Two per-layer signals:

* **Head redundancy** — mean pairwise cosine similarity between the flattened
  attention maps of every head in a layer. Lower ⇒ heads specialise (good);
  ``~1.0`` ⇒ heads are near-duplicates (the extra heads bought nothing).
* **Geometry usage** — the L2 norm of each bias head's ``rel_bias`` row. Zero
  ⇒ that head never left its "plain extra head" init; larger ⇒ it leaned on the
  (Δfile, Δrank) prior.

Attention weights are recomputed here because the model's forward uses the fused
:func:`F.scaled_dot_product_attention`, which does not expose them. We capture
each attention block's input with a forward pre-hook and replay the same
math (including the bias) to get the softmax maps — the model's hot path is
untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from .board_transformer import BoardTransformer, _BidirectionalAttention


@dataclass(slots=True)
class LayerDiagnostics:
    """Per-layer bias-head diagnostics (see module docstring)."""

    layer: int
    n_head: int
    n_bias_head: int
    #: Mean pairwise cosine similarity across *all* heads' attention maps.
    mean_head_similarity: float
    #: Mean pairwise similarity restricted to (content head, bias head) pairs, or
    #: ``None`` when the layer has no bias heads.
    content_bias_similarity: float | None
    #: L2 norm of each bias head's ``rel_bias`` row (length ``n_bias_head``).
    rel_bias_norms: list[float]


def _attention_maps_from_input(attn: _BidirectionalAttention, x: torch.Tensor) -> torch.Tensor:
    """Recompute softmax attention weights for one block, batch-averaged.

    Mirrors :meth:`_BidirectionalAttention.forward` exactly (scale, additive bias
    on the trailing bias heads) but returns the probabilities instead of the
    context. Returns ``[total_heads, T, T]``.
    """
    b, t, _ = x.shape
    qkv = attn.qkv(x)
    q, k, _ = qkv.split(attn.inner_dim, dim=2)
    q = q.view(b, t, attn.total_heads, attn.head_dim).transpose(1, 2)  # [B, H, T, hd]
    k = k.view(b, t, attn.total_heads, attn.head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(attn.head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]

    if attn.n_bias_head > 0:
        rel_index: torch.Tensor = attn.rel_index  # type: ignore[assignment]
        bias_rows = attn.rel_bias[:, rel_index]  # [n_bias_head, T, T]
        content = torch.zeros(attn.n_head, t, t, dtype=bias_rows.dtype, device=bias_rows.device)
        attn_bias = torch.cat([content, bias_rows], dim=0)  # [H, T, T]
        scores = scores + attn_bias.unsqueeze(0)

    probs = F.softmax(scores.float(), dim=-1)  # [B, H, T, T]
    return probs.mean(0)  # [H, T, T]


def _mean_pairwise_cosine(maps: torch.Tensor, rows: slice, cols: slice) -> float | None:
    """Mean cosine similarity between flattened attention maps for two head groups.

    ``maps`` is ``[H, T, T]``. ``rows``/``cols`` select head groups; the diagonal
    (a head vs itself) is excluded only when the two groups are identical.
    """
    flat = maps.reshape(maps.shape[0], -1)  # [H, T*T]
    flat = F.normalize(flat, dim=1)
    sim = flat[rows] @ flat[cols].T  # [len(rows), len(cols)]
    same_group = rows == cols
    if same_group:
        n = sim.shape[0]
        if n < 2:
            return None
        off_diag = sim.sum() - torch.diagonal(sim).sum()
        return float(off_diag / (n * (n - 1)))
    if sim.numel() == 0:
        return None
    return float(sim.mean())


def head_diversity(model: BoardTransformer, boards: torch.Tensor) -> list[LayerDiagnostics]:
    """Compute per-layer head-redundancy + geometry-usage diagnostics.

    Args:
        model: a (typically trained) :class:`BoardTransformer`.
        boards: a LongTensor ``[B, 91]`` of board-v1 token ids (a val batch).

    Returns:
        one :class:`LayerDiagnostics` per transformer layer.
    """
    captured: dict[int, torch.Tensor] = {}
    attn_modules = [cast(_BidirectionalAttention, block.attn) for block in model.blocks]
    index_of = {id(m): i for i, m in enumerate(attn_modules)}

    def pre_hook(module: _BidirectionalAttention, args: tuple[torch.Tensor, ...]) -> None:
        captured[index_of[id(module)]] = args[0].detach()

    handles = [m.register_forward_pre_hook(pre_hook) for m in attn_modules]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(boards)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    out: list[LayerDiagnostics] = []
    for i, attn in enumerate(attn_modules):
        with torch.no_grad():
            maps = _attention_maps_from_input(attn, captured[i])  # [H, T, T]
        h_content = attn.n_head
        h_total = attn.total_heads
        all_slice = slice(0, h_total)
        mean_sim = _mean_pairwise_cosine(maps, all_slice, all_slice)
        cb_sim: float | None = None
        norms: list[float] = []
        if attn.n_bias_head > 0:
            cb_sim = _mean_pairwise_cosine(maps, slice(0, h_content), slice(h_content, h_total))
            norms = attn.rel_bias.detach().norm(dim=1).tolist()
        out.append(
            LayerDiagnostics(
                layer=i,
                n_head=h_content,
                n_bias_head=attn.n_bias_head,
                mean_head_similarity=mean_sim if mean_sim is not None else float("nan"),
                content_bias_similarity=cb_sim,
                rel_bias_norms=[float(n) for n in norms],
            )
        )
    return out


def format_report(diags: list[LayerDiagnostics]) -> str:
    """Render :func:`head_diversity` output as a compact human-readable table."""
    lines = [
        "layer  heads(c+b)  head_sim  content_vs_bias_sim  rel_bias_norms",
        "-----  ----------  --------  -------------------  --------------",
    ]
    for d in diags:
        cb = "     —" if d.content_bias_similarity is None else f"{d.content_bias_similarity:6.3f}"
        norms = ", ".join(f"{n:.3f}" for n in d.rel_bias_norms) if d.rel_bias_norms else "—"
        lines.append(
            f"{d.layer:5d}  {d.n_head:>4d}+{d.n_bias_head:<4d}  "
            f"{d.mean_head_similarity:8.3f}  {cb:>19}  {norms}"
        )
    # Summary footer: averages that make an A/B/C comparison a one-glance read.
    valid = [d.mean_head_similarity for d in diags if not math.isnan(d.mean_head_similarity)]
    all_norms = [n for d in diags for n in d.rel_bias_norms]
    if valid:
        lines.append("")
        lines.append(f"mean head similarity (all layers): {sum(valid) / len(valid):.3f}")
    if all_norms:
        lines.append(
            f"mean |rel_bias| across bias heads: {sum(all_norms) / len(all_norms):.3f} "
            f"(0 = geometry unused)"
        )
    return "\n".join(lines)
