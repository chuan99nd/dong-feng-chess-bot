"""Training loops: BC pretrain / SFT / distillation / RL (stub).

Roadmap milestones: **M2** (behavior cloning pretrain, SFT, engine distillation)
and **M5** (reinforcement learning from self-play).

This module will implement the concrete :class:`~dongfeng.training.base.Trainer`
regimes, mirroring the strongest published Xiangqi pipelines:

* **BC pretrain** — behavior cloning on tokenized human games (winning-side moves),
  a cross-entropy policy objective over the move vocabulary.
* **SFT** — supervised fine-tuning on curated / annotated positions.
* **Distillation** — match an engine teacher (e.g. Pikafish over UCI with
  ``UCI_ShowWDL`` + ``MultiPV``): softmaxed MultiPV move scores as the policy
  target and per-mille WDL as the action-value target (see the data research
  findings and :class:`dongfeng.model.base.PolicyModel`'s value head).
* **RL** — reinforcement learning from self-play, refining the distilled policy.

Until M2/M5, the loops below raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from pathlib import Path

from ..model.base import PolicyModel
from .base import TrainConfig


def bc_pretrain(model: PolicyModel, config: TrainConfig) -> Path:
    """Behavior-cloning pretrain on tokenized human games.

    Raises:
        NotImplementedError: always — planned for milestone M2.
    """
    raise NotImplementedError("bc_pretrain is planned: M2")


def sft(model: PolicyModel, config: TrainConfig) -> Path:
    """Supervised fine-tuning on curated positions.

    Raises:
        NotImplementedError: always — planned for milestone M2.
    """
    raise NotImplementedError("sft is planned: M2")


def distill(model: PolicyModel, config: TrainConfig) -> Path:
    """Distill from an engine teacher (policy + action-value targets).

    Raises:
        NotImplementedError: always — planned for milestone M2.
    """
    raise NotImplementedError("distill is planned: M2")


def rl_selfplay(model: PolicyModel, config: TrainConfig) -> Path:
    """Reinforcement learning from self-play.

    Raises:
        NotImplementedError: always — planned for milestone M5.
    """
    raise NotImplementedError("rl_selfplay is planned: M5")
