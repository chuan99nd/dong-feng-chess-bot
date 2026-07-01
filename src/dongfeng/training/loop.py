"""Training loops: BC pretrain (M2); SFT / distillation / RL (later milestones).

``bc_pretrain`` is the real M2 behavior-cloning loop: it streams the ``uint16``
move-id shards written by :func:`dongfeng.data.dataset.build_shards`, samples
fixed-length blocks, and trains the decoder-only
:class:`~dongfeng.model.transformer.TransformerPolicy` with a next-move
cross-entropy objective — the direct analogue of LLM next-token pretraining.
``sft`` / ``distill`` / ``rl_selfplay`` remain stubs for M4/M5.
"""

from __future__ import annotations

import math
from array import array
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model.transformer import TransformerPolicy
from .base import TrainConfig


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to mps/cuda/cpu; pass through an explicit device string."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_token_stream(data_dir: str | Path) -> torch.Tensor:
    """Concatenate all ``shard_*.bin`` (uint16) files in ``data_dir`` into a 1-D tensor."""
    shards = sorted(Path(data_dir).glob("shard_*.bin"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.bin files in {data_dir}")
    buf = array("H")
    for shard in shards:
        chunk = array("H")
        chunk.frombytes(shard.read_bytes())
        buf.extend(chunk)
    return torch.tensor(buf, dtype=torch.long)


def _get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_loss(
    model: TransformerPolicy,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    gen: torch.Generator,
    iters: int = 20,
) -> float:
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = _get_batch(data, block_size, batch_size, device, gen)
        logits = model(x)
        losses[k] = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return losses.mean().item()


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def bc_pretrain(model: TransformerPolicy, config: TrainConfig) -> Path:
    """Behavior-cloning pretrain on tokenized move shards; returns the checkpoint path.

    Trains next-move prediction (cross-entropy) over a 98/2 train/val split of the
    concatenated move-id stream, with AdamW, warmup+cosine LR, and grad clipping.
    The final checkpoint (weights + model config + training metadata) is written to
    ``config.out_dir / "ckpt.pt"``.
    """
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    gen = torch.Generator().manual_seed(config.seed)

    stream = load_token_stream(config.data_dir)
    n_val = max(int(len(stream) * 0.02), config.batch_size * model.config.block_size)
    train_data, val_data = stream[:-n_val], stream[-n_val:]

    model.to(device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95)
    )

    block_size = model.config.block_size
    ckpt_path = Path(config.out_dir) / "ckpt.pt"
    best_val = float("inf")

    for step in range(config.max_steps):
        lr = _lr_at(step, config)
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = _get_batch(train_data, block_size, config.batch_size, device, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()

        is_last = step == config.max_steps - 1
        if step % config.checkpoint_every == 0 or is_last:
            val_loss = _estimate_loss(model, val_data, block_size, config.batch_size, device, gen)
            marker = ""
            # Keep the best-by-val checkpoint (early stopping); never overwrite it
            # with a later, overfit model.
            if val_loss < best_val:
                best_val = val_loss
                marker = "  *saved"
                model.save(
                    ckpt_path,
                    extra={
                        "tokenizer": "move-v1",
                        "step": step,
                        "val_loss": val_loss,
                        "train_loss": loss.item(),
                        "device": device,
                    },
                )
            print(
                f"step {step:>6}/{config.max_steps}  lr {lr:.2e}  "
                f"train {loss.item():.4f}  val {val_loss:.4f}{marker}",
                flush=True,
            )
    return ckpt_path


def sft(model: TransformerPolicy, config: TrainConfig) -> Path:
    """Supervised fine-tuning on curated positions (planned: M4)."""
    raise NotImplementedError("sft is planned: M4")


def distill(model: TransformerPolicy, config: TrainConfig) -> Path:
    """Distill from an engine teacher (policy + action-value targets) (planned: M4)."""
    raise NotImplementedError("distill is planned: M4")


def rl_selfplay(model: TransformerPolicy, config: TrainConfig) -> Path:
    """Reinforcement learning from self-play (planned: M5)."""
    raise NotImplementedError("rl_selfplay is planned: M5")
