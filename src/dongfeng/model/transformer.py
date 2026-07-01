"""Decoder-only transformer policy model (M2).

:class:`TransformerPolicy` is the flagship Dong Feng model: an LLM-style
decoder-only transformer over the flat ICCS move vocabulary
(:class:`dongfeng.tokenizer.move_tokenizer.MoveTokenizer`). A game is a sequence of
move ids; the model predicts the next move — next-move prediction, the direct
analogue of next-token prediction. It optionally carries an action-value head
(reserved for M4 engine distillation; see :class:`dongfeng.model.base.PolicyModel`).

The architecture is a standard pre-norm GPT: token + learned positional
embeddings, a stack of causal self-attention / MLP blocks, a final layer norm, and
a policy head over the move vocabulary. Config knobs live in
:class:`TransformerConfig`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class TransformerConfig:
    """Architecture hyperparameters for :class:`TransformerPolicy`."""

    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    with_value_head: bool = False


class _CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = c // self.n_head
        q = q.view(b, t, self.n_head, head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_dim).transpose(1, 2)
        # Fused, causal attention (handles the lower-triangular mask internally).
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class _Block(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = _CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerPolicy(nn.Module):
    """Decoder-only transformer implementing :class:`~dongfeng.model.base.PolicyModel`."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.policy_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.value_head = nn.Linear(config.n_embd, 1) if config.with_value_head else None
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _backbone(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        if t > self.config.block_size:
            tokens = tokens[:, -self.config.block_size :]
            t = self.config.block_size
        pos = torch.arange(t, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return policy logits ``[batch, seq, vocab]`` over the move vocabulary."""
        return self.policy_head(self._backbone(tokens))

    def value(self, tokens: torch.Tensor) -> torch.Tensor | None:
        """Return per-position value estimates, or ``None`` if no value head."""
        if self.value_head is None:
            return None
        return self.value_head(self._backbone(tokens)).squeeze(-1)

    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        """Save weights + config (and optional ``extra`` metadata) to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": asdict(self.config), "state_dict": self.state_dict(), "extra": extra or {}},
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, map_location: Any = "cpu"
    ) -> tuple[TransformerPolicy, dict[str, Any]]:
        """Load a checkpoint saved by :meth:`save`; return ``(model, extra)``."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(TransformerConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt.get("extra", {})
