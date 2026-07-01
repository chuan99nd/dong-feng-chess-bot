"""Training contracts: the trainer interface and its configuration.

A :class:`Trainer` drives a :class:`~dongfeng.model.base.PolicyModel` through one
of Dong Feng's training regimes — behavior cloning / SFT on human games,
distillation from an engine teacher, and reinforcement learning from self-play (see
:mod:`dongfeng.training.loop`). :class:`TrainConfig` collects the hyperparameters
shared across those regimes.

Roadmap: the concrete training loops land in milestones **M2** (BC pretrain / SFT /
distillation) and **M5** (RL). This module defines only the stable contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters and run settings shared by the training regimes.

    Attributes:
        data_dir: Directory of tokenized shards (see :mod:`dongfeng.data.dataset`).
        out_dir: Directory to write checkpoints and logs into.
        batch_size: Number of samples per optimization step.
        lr: Peak learning rate.
        weight_decay: AdamW-style weight decay.
        warmup_steps: Linear warmup steps before the main schedule.
        max_steps: Total optimization steps to run.
        grad_clip: Max global grad norm (``None`` disables clipping).
        seed: RNG seed for reproducibility.
        checkpoint_every: Save a checkpoint every N steps.
        device: Target device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    data_dir: Path
    out_dir: Path
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1_000
    max_steps: int = 100_000
    grad_clip: float | None = 1.0
    seed: int = 0
    checkpoint_every: int = 5_000
    device: str = "cpu"


@runtime_checkable
class Trainer(Protocol):
    """Drives a model through a training regime and checkpoints it.

    Concrete trainers (BC / SFT / distillation / RL) implement this contract so the
    CLI and experiment scripts can launch any regime uniformly.
    """

    def train(self, config: TrainConfig) -> Path:
        """Run training under ``config`` and return the path to the final checkpoint."""
        ...

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist current model + optimizer state to ``path``."""
        ...

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model + optimizer state from ``path``."""
        ...
