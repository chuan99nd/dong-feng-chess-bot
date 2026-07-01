"""Decoder-only transformer policy model (stub).

Roadmap milestone: **M2**.

:class:`TransformerPolicy` will be the primary Dong Feng model: an LLM-style
decoder-only transformer that consumes a tokenized position/history (from
:mod:`dongfeng.tokenizer`) and produces move-policy logits, optionally with an
action-value head for engine distillation (see
:class:`dongfeng.model.base.PolicyModel`).

Planned architecture (pinned in M2): token + positional embeddings, a stack of
pre-norm self-attention / MLP blocks, and output heads (a policy head over the move
vocabulary and an optional value head). Config knobs — layer count, model width,
head count, context length, vocab size — are supplied via a config object defined
in M2. The training recipes (BC pretrain / SFT / distillation / RL) live in
:mod:`dongfeng.training`.

Until M2, construction and the forward pass raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any


class TransformerPolicy:
    """Decoder-only transformer implementing :class:`~dongfeng.model.base.PolicyModel` (M2 stub).

    The concrete network, config schema, and weights are finalized in milestone M2.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct the transformer.

        Raises:
            NotImplementedError: always — planned for milestone M2.
        """
        raise NotImplementedError("TransformerPolicy is planned: M2")

    def forward(self, tokens: Any) -> Any:
        """Run the transformer and return policy logits over the move vocabulary.

        Raises:
            NotImplementedError: always — planned for milestone M2.
        """
        raise NotImplementedError("TransformerPolicy.forward is planned: M2")

    def value(self, tokens: Any) -> Any:
        """Optional action-value head for distillation targets.

        Raises:
            NotImplementedError: always — planned for milestone M2.
        """
        raise NotImplementedError("TransformerPolicy.value is planned: M2")
