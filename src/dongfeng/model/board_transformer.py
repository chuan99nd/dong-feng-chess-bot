"""Board-state transformer policy+value model (M3.5 / board-1b plan WP2).

:class:`BoardTransformer` is an encoder-only (bidirectional) transformer that
consumes a 91-token board representation from
:class:`~dongfeng.tokenizer.board_tokenizer.BoardTokenizer` and emits:

* A **policy head**: logits over 2 554 legal Xiangqi moves (move-v1 vocab).
* A **value head**: a scalar in (−1, 1) representing expected outcome from the
  side-to-move's perspective.

Both heads read from the hidden state at position 90, the side-to-move token,
which acts as a [CLS]-like aggregate token.

Architecture details (pinned by §1.1–1.2 of docs/plans/board-1b.md):

* Pre-norm **RMSNorm**, **SwiGLU** FFN, **no biases** anywhere.
* **2D positional embeddings**: col_emb (size 9) + rank_emb (size 10) for the
  90 board squares; index 90 (side token) gets its own learned vector.
* :func:`F.scaled_dot_product_attention` with ``is_causal=False``.
* Optional ``gradient_checkpointing`` wraps each block in
  :func:`torch.utils.checkpoint.checkpoint`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Config + presets
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BoardTransformerConfig:
    """Architecture hyperparameters for :class:`BoardTransformer`."""

    d_model: int = 384
    n_layer: int = 12
    n_head: int = 6
    ffn_hidden: int = 1024
    vocab_size: int = 21  # board-v1 token vocabulary
    seq_len: int = 91  # 90 board squares + 1 side-to-move token
    n_moves: int = 2554  # move-v1 policy head output size
    gradient_checkpointing: bool = False

    @classmethod
    def presets(cls) -> dict[str, BoardTransformerConfig]:
        """Return the three canonical presets from §1.1 of the board-1b plan."""
        return {
            "m1-dev": cls(d_model=384, n_layer=12, n_head=6, ffn_hidden=1024),
            "mid": cls(d_model=768, n_layer=20, n_head=12, ffn_hidden=2048),
            "1b": cls(d_model=1536, n_layer=36, n_head=12, ffn_hidden=4096),
        }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias, no learnable mean shift)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class _SwiGLU(nn.Module):
    """SwiGLU feed-forward block — no biases anywhere."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        # Gate + value projections fused into one weight matrix for efficiency
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_val = nn.Linear(d_model, hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))


class _BidirectionalAttention(nn.Module):
    """Multi-head self-attention without causal mask (encoder-style), no biases."""

    def __init__(self, cfg: BoardTransformerConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3*d]
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        # Bidirectional: is_causal=False
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(y)


class _Block(nn.Module):
    """Single transformer encoder block: pre-norm RMSNorm, attn, pre-norm RMSNorm, SwiGLU."""

    def __init__(self, cfg: BoardTransformerConfig) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(cfg.d_model)
        self.attn = _BidirectionalAttention(cfg)
        self.norm2 = _RMSNorm(cfg.d_model)
        self.ffn = _SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class BoardTransformer(nn.Module):
    """Encoder-only board-state transformer with policy and value heads.

    Input: ``boards`` — a LongTensor of shape ``[B, 91]`` containing board-v1 token
    ids as produced by :class:`~dongfeng.tokenizer.board_tokenizer.BoardTokenizer`.

    Output: ``(policy_logits, value)`` where:

    * ``policy_logits`` — ``[B, 2554]`` raw logits over the move vocabulary.
    * ``value`` — ``[B]`` scalar predictions in (−1, 1) from side-to-move's perspective.
    """

    def __init__(self, config: BoardTransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # Token embedding (vocab 21 board tokens)
        self.tok_emb = nn.Embedding(config.vocab_size, d)

        # 2D positional embeddings: col_emb (9 files) + rank_emb (10 ranks)
        # Index 90 (side-to-move token) gets its own vector.
        self.col_emb = nn.Embedding(9, d)
        self.rank_emb = nn.Embedding(10, d)
        self.side_emb = nn.Parameter(torch.zeros(d))  # single learned vector for idx 90

        # Pre-compute board position indices (col/rank) — registered as buffers so
        # they follow device/dtype moves automatically.
        cols = torch.arange(90) % 9  # shape [90]
        ranks = 9 - torch.arange(90) // 9  # shape [90], rank_emb[9 - i//9]
        self.register_buffer("_cols", cols, persistent=False)
        self.register_buffer("_ranks", ranks, persistent=False)

        # Transformer blocks
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layer))

        # Final norms (pre-head)
        self.norm_out = _RMSNorm(d)

        # Policy head: linear over move-v1 vocab (applied to hidden[90])
        self.policy_head = nn.Linear(d, config.n_moves, bias=False)

        # Value head: RMSNorm → Linear(d, d//4) → SiLU → Linear(d//4, 1) → tanh
        self.value_norm = _RMSNorm(d)
        self.value_fc1 = nn.Linear(d, d // 4, bias=False)
        self.value_fc2 = nn.Linear(d // 4, 1, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
        nn.init.zeros_(self.side_emb)

    def _positional_embeddings(self, device: torch.device) -> torch.Tensor:
        """Build [91, d_model] positional embedding matrix."""
        # Board squares 0..89: col_emb + rank_emb
        cols: torch.Tensor = self._cols  # type: ignore[assignment]
        ranks: torch.Tensor = self._ranks  # type: ignore[assignment]
        board_pos = self.col_emb(cols) + self.rank_emb(ranks)  # [90, d]
        # Index 90: side-to-move token's own vector, unsqueezed to [1, d]
        side_pos = self.side_emb.unsqueeze(0)  # [1, d]
        return torch.cat([board_pos, side_pos], dim=0)  # [91, d]

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)  # type: ignore[assignment]
            else:
                x = block(x)
        return x

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            boards: LongTensor ``[B, 91]`` of board-v1 token ids.

        Returns:
            ``(policy_logits, value)`` where ``policy_logits`` is ``[B, 2554]`` and
            ``value`` is ``[B]``, both on the same device as ``boards``.
        """
        # Token embeddings + positional embeddings
        x = self.tok_emb(boards)  # [B, 91, d]
        pos = self._positional_embeddings(boards.device)  # [91, d]
        x = x + pos.unsqueeze(0)  # [B, 91, d]

        # Transformer blocks (with optional grad checkpointing)
        x = self._run_blocks(x)  # [B, 91, d]

        # Normalise output
        x = self.norm_out(x)  # [B, 91, d]

        # Extract the side-to-move token (index 90) as the aggregate representation
        cls = x[:, 90, :]  # [B, d]

        # Policy head
        policy_logits = self.policy_head(cls)  # [B, 2554]

        # Value head: RMSNorm → fc1 → SiLU → fc2 → tanh
        v = self.value_norm(cls)
        v = self.value_fc1(v)
        v = F.silu(v)
        v = self.value_fc2(v)  # [B, 1]
        value = torch.tanh(v).squeeze(-1)  # [B]

        return policy_logits, value

    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        """Save weights + config (and optional ``extra`` metadata) to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, map_location: Any = "cpu"
    ) -> tuple[BoardTransformer, dict[str, Any]]:
        """Load a checkpoint saved by :meth:`save`; return ``(model, extra)``."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(BoardTransformerConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt.get("extra", {})
