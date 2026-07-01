"""Model contract: the policy (and optional value) network interface.

A :class:`PolicyModel` maps a tokenized position (from
:mod:`dongfeng.tokenizer`) to a distribution over moves — the policy. Optionally
it also exposes an **action-value head** (per-move value / win-probability
estimates) used for engine distillation, where a strong teacher such as Pikafish
supplies WDL / centipawn targets (see the data research findings).

Roadmap: the concrete decoder-only transformer lands in milestone **M2**. This
module defines only the stable contract, so the inference engine
(:mod:`dongfeng.inference`) and training loop (:mod:`dongfeng.training`) can be
typed against it now.

Note on tensor types: to keep the FOUNDATION spine free of a hard tensor-library
dependency, tensor arguments/returns are annotated ``Any``. The concrete M2
implementation pins the framework (e.g. PyTorch) and its tensor types.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PolicyModel(Protocol):
    """A move-policy network (optionally with an action-value head).

    Implementations are typically decoder-only transformers trained by behavior
    cloning / SFT and refined by distillation and RL (see :mod:`dongfeng.training`).
    """

    def forward(self, tokens: Any) -> Any:
        """Run the model on a batch of tokenized positions and return policy logits.

        Args:
            tokens: A batch of token-id sequences (framework tensor) as produced by
                a :class:`~dongfeng.tokenizer.base.Tokenizer`.

        Returns:
            Policy logits over the move vocabulary (framework tensor). Callers apply
            legal-move masking (see :mod:`dongfeng.core`) before sampling.
        """
        ...

    def value(self, tokens: Any) -> Any:
        """Optional action-value head: per-move value / win-probability estimates.

        Used for engine distillation (teacher WDL / centipawn targets). Models
        without a value head may return ``None``.

        Args:
            tokens: A batch of token-id sequences (framework tensor).

        Returns:
            Per-move value estimates (framework tensor), or ``None`` if the model
            has no value head.
        """
        ...
