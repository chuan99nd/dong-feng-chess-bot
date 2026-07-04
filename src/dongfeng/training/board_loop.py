"""Board-state BC training loop + metrics writer (M3.5 / WP3).

Trains :class:`~dongfeng.model.board_transformer.BoardTransformer` via
behavior-cloning on board-state shards produced by WP1
(:func:`~dongfeng.data.board_dataset.build_board_shards`).

Public API
----------
resolve_device_dtype(device)  -- picks device + autocast dtype per §1.2.
BoardTrainConfig              -- dataclass with all training hyperparameters.
bc_train_board(config)        -- runs the loop; returns path to best checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..data.board_dataset import load_board_arrays
from ..model.board_transformer import BoardTransformer, BoardTransformerConfig

# ---------------------------------------------------------------------------
# Device / dtype helper (§1.2)
# ---------------------------------------------------------------------------

_MASK_VALUE: int = 127  # int8 sentinel for unknown game-result


def resolve_device_dtype(device: str = "auto") -> tuple[str, torch.dtype]:
    """Resolve a device string and the matching autocast dtype.

    Rules (§1.2):
    - ``"cuda"`` → ``(cuda, bf16)``
    - ``"mps"``  → ``(mps,  fp16)``  forward in fp16 autocast; fp32 master weights.
      **Escape hatch**: set ``DONGFENG_FORCE_FP32=1`` to force fp32 on MPS, which
      disables autocast entirely and avoids fp16 NaN instability (§5 risk). Use this
      if val loss becomes NaN during MPS training.
    - ``"cpu"``  → ``(cpu,  fp32)``  no autocast
    - ``"auto"`` → picks cuda > mps > cpu, then applies the above rules.

    Returns:
        ``(device_str, torch_dtype)``
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device.startswith("cuda"):
        return device, torch.bfloat16
    if device == "mps":
        # DONGFENG_FORCE_FP32=1 forces full fp32 on MPS to avoid fp16 NaN instability.
        if os.environ.get("DONGFENG_FORCE_FP32", "0") not in ("0", ""):
            return device, torch.float32
        return device, torch.float16
    return device, torch.float32


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class BoardTrainConfig:
    """Hyperparameters for :func:`bc_train_board`.

    Attributes:
        data_dir:        Directory of board shards written by WP1.
        out_dir:         Run directory (will hold run.json, metrics.jsonl, ckpt.pt).
        preset:          One of "m1-dev", "mid", "1b" — or None when config_override is set.
        id:              Run identifier written into run.json.
        batch_size:      Samples per gradient step.
        lr:              Peak learning rate.
        warmup:          Linear warmup steps.
        max_steps:       Total training steps.
        value_weight:    Weight on value MSE relative to policy CE.
        device:          ``"auto"`` or explicit device string.
        seed:            RNG seed for reproducibility.
        grad_checkpoint: Enable gradient checkpointing in the model.
        eval_every:      Validate (and maybe save) every this many steps.
        config_override: Optional :class:`~dongfeng.model.board_transformer.BoardTransformerConfig`
                         that replaces the preset lookup. Useful in tests with tiny models.
    """

    data_dir: str | Path
    out_dir: str | Path
    preset: str = "m1-dev"
    id: str = "run"
    batch_size: int = 256
    lr: float = 3e-4
    warmup: int = 1_000
    max_steps: int = 100_000
    value_weight: float = 0.5
    device: str = "auto"
    seed: int = 0
    grad_checkpoint: bool = False
    eval_every: int = 1_000
    compile: bool = True
    """torch.compile the model on CUDA for kernel fusion (T1). Ignored on mps/cpu
    where compile is flaky/slow. First steps are slow (JIT warmup)."""
    profile: bool = False
    """Run the PyTorch profiler over a short window once and write profile.json
    (per-op FLOPS + device time + measured TFLOP/s) for the UI monitor."""
    profile_at: int = 25
    """Step at which to run the profiler window (after compile/JIT warmup)."""
    profile_steps: int = 8
    """Number of steps to profile in the window."""
    optim: str = "adamw"
    """Optimizer to use: ``"adamw"`` (default) or ``"adam8bit"``.

    ``"adam8bit"`` uses ``bitsandbytes.optim.Adam8bit`` (cuda-only).  If
    ``bitsandbytes`` is not installed or the device is not CUDA, the trainer
    emits a warning and falls back to AdamW automatically.
    """
    resume: str | Path | None = None
    """Optional path to a ``ckpt.pt`` to resume from.

    When set, the model is rebuilt from the checkpoint's saved config (not the
    preset) and its weights are loaded; training continues from ``step + 1`` and
    the best-val bar is seeded from the checkpoint. Enables warm restarts on
    ephemeral machines. Optimizer momentum is not restored (the LR schedule is
    recomputed from the resumed step).
    """
    config_override: BoardTransformerConfig | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lr_at(step: int, warmup: int, max_steps: int, peak_lr: float) -> float:
    """Linear warmup → cosine decay to 10 % of peak."""
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def _compute_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    move_targets: torch.Tensor,
    value_targets: torch.Tensor,
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute combined loss (§1.7).

    Returns:
        ``(total_loss, policy_loss, value_loss_or_None)``
        value_loss is None when the entire batch is masked.
    """
    # Policy CE over full 2554 vocab — no masking at train time.
    policy_loss = F.cross_entropy(policy_logits, move_targets)

    # Value MSE only where target != 127.
    val_float = value_targets.float()  # −1 / 0 / +1 / 127
    mask = value_targets != _MASK_VALUE  # bool tensor [B]
    if mask.any():
        # Compute the value MSE in fp32: under autocast (fp16 on MPS, bf16 on
        # CUDA) ``value_pred`` is low-precision while ``val_float`` is fp32, and
        # MPS rejects the mixed-dtype subtract inside ``mse_loss``. Casting the
        # prediction up also keeps the value loss numerically stable.
        value_loss: torch.Tensor | None = F.mse_loss(
            value_pred[mask].float(), val_float[mask], reduction="mean"
        )
    else:
        value_loss = None

    total = policy_loss + value_weight * value_loss if value_loss is not None else policy_loss

    return total, policy_loss, value_loss


def _autocast_ctx(device: str, dtype: torch.dtype) -> Any:
    """Return a torch.autocast context manager, or a no-op on cpu/fp32."""
    if dtype == torch.float32:
        return torch.inference_mode.__class__  # dummy — overridden below
    device_type = "cuda" if device.startswith("cuda") else device
    return torch.autocast(device_type=device_type, dtype=dtype)


def _profile_window(
    model: Any,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    forward_ctx: Any,
    value_weight: float,
    device: str,
    n_steps: int,
    out_path: Path,
) -> None:
    """Profile a few fwd+bwd steps and write a per-op FLOPS/time breakdown.

    Answers "which op dominates compute / how efficient is it": ``with_flops``
    tags each aten op (matmuls in attention/FFN, norms, elementwise) with its
    FLOP count, and we pair that with device time to get an op table plus the
    achieved TFLOP/s over the window. Run on the *eager* model — FLOP counts are
    compile-invariant and op attribution is clean. Best-effort: never raises.
    """
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    b_batch, m_batch, v_batch = batch
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    def _dev_us(e: Any) -> float:
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                return float(v)
        return float(getattr(e, "self_cpu_time_total", 0) or 0)

    try:
        # Warm the exact path once so the window measures steady state.
        with forward_ctx():
            p, v = model(b_batch)
        loss, _, _ = _compute_loss(p, v, m_batch, v_batch, value_weight)
        loss.backward()
        model.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
            for _ in range(n_steps):
                with forward_ctx():
                    p, v = model(b_batch)
                loss, _, _ = _compute_loss(p, v, m_batch, v_batch, value_weight)
                loss.backward()
                model.zero_grad(set_to_none=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        rows: list[dict[str, Any]] = []
        total_flops = 0.0
        total_dev = 0.0
        for e in prof.key_averages():
            fl = float(getattr(e, "flops", 0) or 0)
            dev = _dev_us(e)
            total_flops += fl
            total_dev += dev
            rows.append(
                {
                    "name": e.key,
                    "count": int(e.count),
                    "device_us": round(dev, 1),
                    "gflops": round(fl / 1e9, 3),
                }
            )
        for r in rows:
            r["device_pct"] = round(100 * r["device_us"] / total_dev, 1) if total_dev > 0 else 0.0
        rows.sort(key=lambda r: r["device_us"], reverse=True)

        summary = {
            "generated": datetime.now(UTC).isoformat(),
            "device": device,
            "n_steps": n_steps,
            "wall_s": round(wall, 4),
            "ms_per_step": round(1000 * wall / max(n_steps, 1), 2),
            "gflops_per_step": round(total_flops / 1e9 / max(n_steps, 1), 1),
            # FLOPs cover fwd+bwd aten ops that with_flops recognises (mm/bmm/…).
            "measured_tflops": round(total_flops / wall / 1e12, 2) if wall > 0 else None,
            "ops": rows[:25],
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    except Exception as exc:  # profiling must never break training
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"error": f"{type(exc).__name__}: {exc}"}, fh)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def bc_train_board(config: BoardTrainConfig) -> Path:
    """Behavior-cloning training on board-state shards.

    Loads data from *config.data_dir*, builds a :class:`BoardTransformer` from
    *config.preset* (or *config.config_override*), trains with AdamW +
    warmup/cosine LR, writes ``run.json`` and ``metrics.jsonl`` to *config.out_dir*,
    saves ``ckpt.pt`` whenever val policy_loss improves, and returns the ckpt path.

    Args:
        config: All training hyperparameters (see :class:`BoardTrainConfig`).

    Returns:
        Path to the best checkpoint ``<out_dir>/ckpt.pt``.

    Raises:
        ValueError: If loss becomes NaN/Inf (run is marked ``"failed"`` first).
    """
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = resolve_device_dtype(config.device)

    # ------------------------------------------------------------------ model
    if config.config_override is not None:
        model_cfg = config.config_override
    else:
        presets = BoardTransformerConfig.presets()
        if config.preset not in presets:
            raise ValueError(f"Unknown preset {config.preset!r}; choose from {list(presets)}")
        model_cfg = presets[config.preset]
    model_cfg.gradient_checkpointing = config.grad_checkpoint

    torch.manual_seed(config.seed)
    # Resume: rebuild from the checkpoint's own config so shapes always match,
    # then continue from the saved step. Falls back to a fresh model otherwise.
    resume_step = 0
    resume_best_val = float("inf")
    if config.resume is not None and Path(config.resume).exists():
        model, resume_extra = BoardTransformer.load(config.resume, map_location="cpu")
        model.config.gradient_checkpointing = config.grad_checkpoint
        resume_step = int(resume_extra.get("step", -1)) + 1
        resume_best_val = float(resume_extra.get("val_policy_loss", float("inf")))
    else:
        model = BoardTransformer(model_cfg)
    model.to(device)
    model.train()

    # T1: torch.compile fuses kernels (fewer launches + less VRAM round-trip) —
    # the main lever on a compute/bandwidth-bound GPU. Only on CUDA (compile is
    # flaky/slow on mps/cpu). ``model`` stays the eager module (for attribute
    # access, save, and the profiler — FLOP counts are compile-invariant);
    # ``train_model`` is what the train/val forward passes call.
    train_model: Any = model
    compiled = False
    if config.compile and device.startswith("cuda"):
        try:
            train_model = torch.compile(model)
            compiled = True
        except Exception as exc:  # never let compile break a run
            import warnings  # noqa: PLC0415

            warnings.warn(f"torch.compile failed ({exc}); running eager.", stacklevel=2)

    # ------------------------------------------------------------------ data
    boards_np, moves_np, values_np = load_board_arrays(config.data_dir)
    n_total = len(boards_np)
    if n_total == 0:
        raise ValueError(f"No samples found in {config.data_dir}")

    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(n_total)
    n_val = max(1, int(n_total * 0.02))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    # Convert to tensors (keep on CPU — move per-batch to device).
    boards_t = torch.from_numpy(boards_np.astype(np.int64))  # [N, 91]
    moves_t = torch.from_numpy(moves_np.astype(np.int64))  # [N]
    values_t = torch.from_numpy(values_np.astype(np.int64))  # [N]

    train_boards = boards_t[train_idx]
    train_moves = moves_t[train_idx]
    train_values = values_t[train_idx]

    val_boards = boards_t[val_idx]
    val_moves = moves_t[val_idx]
    val_values = values_t[val_idx]

    n_train = len(train_boards)

    def _batch(
        b_arr: torch.Tensor, m_arr: torch.Tensor, v_arr: torch.Tensor, step: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch from the given split."""
        bs = min(config.batch_size, len(b_arr))
        start = (step * bs) % max(len(b_arr) - bs, 1)
        idx = torch.arange(start, start + bs) % len(b_arr)
        return (
            b_arr[idx].to(device),
            m_arr[idx].to(device),
            v_arr[idx].to(device),
        )

    # ------------------------------------------------------------------ optimizer
    fused_available = (
        device.startswith("cuda") and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    )
    opt_kwargs: dict[str, Any] = {
        "lr": config.lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
    }

    opt: torch.optim.Optimizer
    if config.optim == "adam8bit":
        # adam8bit is cuda-only and requires bitsandbytes.  Fall back gracefully.
        if not device.startswith("cuda"):
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"adam8bit requires a CUDA device (got {device!r}); falling back to AdamW.",
                stacklevel=2,
            )
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        else:
            try:
                import bitsandbytes as bnb  # type: ignore[import-untyped]  # noqa: PLC0415

                opt = bnb.optim.Adam8bit(model.parameters(), **opt_kwargs)
            except ImportError:
                import warnings  # noqa: PLC0415

                warnings.warn(
                    "bitsandbytes is not installed; falling back to AdamW.  "
                    "Install with: pip install bitsandbytes",
                    stacklevel=2,
                )
                opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    elif fused_available:
        try:
            opt_kwargs["fused"] = True
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        except TypeError:
            opt_kwargs.pop("fused")
            opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    else:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)

    # ------------------------------------------------------------------ run.json
    started_iso = datetime.now(UTC).isoformat()
    arch_cfg = model.config
    run_meta: dict[str, Any] = {
        "id": config.id,
        "kind": "bc-board",
        "preset": config.preset,
        "arch_hash": arch_cfg.arch_hash(),
        "arch": {
            "d_model": arch_cfg.d_model,
            "n_layer": arch_cfg.n_layer,
            "n_head": arch_cfg.n_head,
            "n_bias_head": arch_cfg.n_bias_head,
            "ffn_hidden": arch_cfg.ffn_hidden,
            "seq_len": arch_cfg.seq_len,
            "vocab_size": arch_cfg.vocab_size,
            "n_moves": arch_cfg.n_moves,
        },
        "params": model.num_params(),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "compiled": compiled,
        "data_dir": str(config.data_dir),
        "started": started_iso,
        "finished": None,
        "status": "running",
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(config).items()
            if k != "config_override"
        },
    }
    run_json_path = out_dir / "run.json"
    metrics_path = out_dir / "metrics.jsonl"
    ckpt_path = out_dir / "ckpt.pt"

    def _write_run_json(status: str, finished: str | None = None) -> None:
        run_meta["status"] = status
        run_meta["finished"] = finished
        with open(run_json_path, "w", encoding="utf-8") as fh:
            json.dump(run_meta, fh, indent=2)

    _write_run_json("running")

    metrics_fh = open(metrics_path, "a", encoding="utf-8")  # noqa: SIM115

    def _log_metrics(row: dict[str, Any]) -> None:
        metrics_fh.write(json.dumps(row) + "\n")
        metrics_fh.flush()

    # ------------------------------------------------------------------ autocast
    use_autocast = dtype != torch.float32

    def _make_autocast() -> Any:
        if not use_autocast:
            return torch.no_grad().__class__()  # will use contextlib.nullcontext below
        device_type = "cuda" if device.startswith("cuda") else device
        return torch.autocast(device_type=device_type, dtype=dtype)

    from contextlib import nullcontext  # noqa: PLC0415

    def _forward_ctx():  # type: ignore[return]
        if use_autocast:
            device_type = "cuda" if device.startswith("cuda") else device
            return torch.autocast(device_type=device_type, dtype=dtype)
        return nullcontext()

    # ------------------------------------------------------------------ eval helper
    @torch.no_grad()
    def _run_val() -> tuple[float, float, float, float]:
        """Return (val_loss, val_policy_loss, val_top1, val_top5) over the val set."""
        model.eval()
        total_loss_acc = 0.0
        total_policy_acc = 0.0
        total_correct = 0
        total_correct5 = 0
        total_n = 0
        n_batches = max(1, math.ceil(len(val_boards) / config.batch_size))
        for bi in range(n_batches):
            s = bi * config.batch_size
            e = min(s + config.batch_size, len(val_boards))
            vb = val_boards[s:e].to(device)
            vm = val_moves[s:e].to(device)
            vv = val_values[s:e].to(device)

            with _forward_ctx():
                p_logits, v_pred = train_model(vb)

            total, p_loss, _ = _compute_loss(p_logits, v_pred, vm, vv, config.value_weight)
            batch_n = e - s
            total_loss_acc += total.item() * batch_n
            total_policy_acc += p_loss.item() * batch_n
            preds = p_logits.argmax(dim=-1)
            total_correct += (preds == vm).sum().item()
            top5 = p_logits.topk(min(5, p_logits.shape[-1]), dim=-1).indices
            total_correct5 += (top5 == vm.unsqueeze(-1)).any(dim=-1).sum().item()
            total_n += batch_n

        model.train()
        n = max(total_n, 1)
        return (
            total_loss_acc / n,
            total_policy_acc / n,
            total_correct / n,
            total_correct5 / n,
        )

    # ------------------------------------------------------------------ train loop
    best_val_policy = resume_best_val
    step_0_loss: float | None = None
    t_start = time.time()
    gen = torch.Generator()
    gen.manual_seed(config.seed)

    try:
        for step in range(resume_step, config.max_steps):
            lr = _lr_at(step, config.warmup, config.max_steps, config.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            # Random batch sampling using step-based cycling with shuffle.
            bs = min(config.batch_size, n_train)
            idx = torch.randint(0, n_train, (bs,), generator=gen)
            b_batch = train_boards[idx].to(device)
            m_batch = train_moves[idx].to(device)
            v_batch = train_values[idx].to(device)

            t0 = time.time()
            with _forward_ctx():
                p_logits, v_pred = train_model(b_batch)

            total, p_loss, v_loss = _compute_loss(
                p_logits, v_pred, m_batch, v_batch, config.value_weight
            )

            # NaN guard (§5 risk)
            if not torch.isfinite(total):
                _write_run_json("failed", datetime.now(UTC).isoformat())
                metrics_fh.close()
                raise ValueError(f"Loss became {total.item()} at step {step}. Run marked failed.")

            opt.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            elapsed = time.time() - t_start
            dt = time.time() - t0
            sps = bs / max(dt, 1e-9)
            tps = sps * model.config.seq_len

            train_loss = total.item()
            if step_0_loss is None:
                step_0_loss = train_loss

            # Log train metrics every 10 steps (or step 0).
            if step == 0 or (step + 1) % max(1, min(10, config.eval_every // 10)) == 0:
                _log_metrics(
                    {
                        "step": step,
                        "split": "train",
                        "loss": train_loss,
                        "policy_loss": p_loss.item(),
                        "value_loss": v_loss.item() if v_loss is not None else None,
                        "top1": None,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "samples_per_s": sps,
                        "tokens_per_s": tps,
                        "elapsed_s": elapsed,
                    }
                )

            # Validate + maybe save.
            do_eval = (step + 1) % config.eval_every == 0 or step == config.max_steps - 1
            if do_eval:
                val_loss, val_policy_loss, val_top1, val_top5 = _run_val()
                val_elapsed = time.time() - t_start
                _log_metrics(
                    {
                        "step": step,
                        "split": "val",
                        "loss": val_loss,
                        "policy_loss": val_policy_loss,
                        "value_loss": None,
                        "top1": val_top1,
                        "top5": val_top5,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "samples_per_s": sps,
                        "tokens_per_s": tps,
                        "elapsed_s": val_elapsed,
                    }
                )
                if val_policy_loss < best_val_policy:
                    best_val_policy = val_policy_loss
                    model.save(
                        ckpt_path,
                        extra={
                            "step": step,
                            "val_policy_loss": val_policy_loss,
                            "val_loss": val_loss,
                            "val_top1": val_top1,
                            "val_top5": val_top5,
                            "preset": config.preset,
                            "arch_hash": model.config.arch_hash(),
                        },
                    )

            # One-shot profiler window (after compile/JIT warmup) → profile.json.
            if config.profile and step == config.profile_at:
                _profile_window(
                    model,
                    (b_batch, m_batch, v_batch),
                    _forward_ctx,
                    config.value_weight,
                    device,
                    config.profile_steps,
                    out_dir / "profile.json",
                )
                opt.zero_grad(set_to_none=True)

    except Exception:
        _write_run_json("failed", datetime.now(UTC).isoformat())
        metrics_fh.close()
        raise

    _write_run_json("done", datetime.now(UTC).isoformat())
    metrics_fh.close()
    return ckpt_path
