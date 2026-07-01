"""Dong Feng training: regimes that turn data + models into checkpoints.

Public surface:
    Trainer, TrainConfig                      -- training contracts
    bc_pretrain, sft, distill, rl_selfplay    -- training loops (stubs, M2/M5)

The loops are stubs landing in milestones M2 (BC / SFT / distillation) and M5 (RL);
the contracts are stable now.
"""

from __future__ import annotations

from .base import TrainConfig, Trainer
from .loop import bc_pretrain, distill, rl_selfplay, sft

__all__ = [
    "TrainConfig",
    "Trainer",
    "bc_pretrain",
    "distill",
    "rl_selfplay",
    "sft",
]
