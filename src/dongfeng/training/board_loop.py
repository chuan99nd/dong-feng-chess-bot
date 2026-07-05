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

from ..data.board_dataset import load_board_arrays, load_eval_arrays
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
    value_eval_weight: float = 0.7
    """Blend weight on the dense Pikafish eval label vs the terminal outcome when
    ``values_eval_*.bin`` are present: ``target = w·eval + (1−w)·terminal``."""
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
    init_from: str | Path | None = None
    """Optional ckpt.pt to warm-start from (shape-matching weight copy).

    Unlike ``resume`` (which continues the SAME architecture from its step), this
    grafts a different checkpoint's weights into a freshly-built model — used to
    enable ``--n-think`` on top of a pre-think checkpoint. Only applied on a
    fresh run (``resume`` takes precedence); think_emb + grown bias columns stay
    at their zero init.
    """
    config_override: BoardTransformerConfig | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _git_info() -> dict[str, str | None]:
    """Best-effort git branch + short commit of the training code (for provenance)."""
    import subprocess  # noqa: PLC0415

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, timeout=3)
            return out.decode().strip() or None
        except Exception:
            return None

    return {
        "git_commit": _run(["rev-parse", "--short", "HEAD"]),
        "git_branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
    }


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
    value_eval: torch.Tensor | None = None,
    eval_alpha: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined loss (§1.7) with no host↔device sync.

    The value MSE is over the samples with a target: the terminal outcome
    (``!= 127``) and/or a dense Pikafish eval label (``value_eval``, NaN=absent).
    When both are present the target is ``eval_alpha·eval + (1−eval_alpha)·term``
    (Phase 4b distillation blend); when only one is present that one is used;
    when neither, the sample is masked out (0 contribution, no gradient). The
    masking is branchless so torch.compile / CUDA-graph capture is preserved.

    The value term is computed in fp32: under autocast (fp16 on MPS, bf16 on
    CUDA) ``value_pred`` is low-precision, and casting up keeps the MSE stable.
    """
    # Policy CE over full 2554 vocab — no masking at train time.
    policy_loss = F.cross_entropy(policy_logits, move_targets)

    has_term = value_targets != _MASK_VALUE  # bool [B]
    term = torch.where(has_term, value_targets.float(), torch.zeros_like(value_targets).float())
    if value_eval is None:
        target = term
        mask = has_term
    else:
        has_eval = ~torch.isnan(value_eval)
        ev = torch.nan_to_num(value_eval.float())  # 0 where NaN (masked out below)
        both = has_eval & has_term
        # both → blend; eval-only → eval; term-only → term.
        target = torch.where(
            both, eval_alpha * ev + (1.0 - eval_alpha) * term, torch.where(has_eval, ev, term)
        )
        mask = has_eval | has_term

    diff = value_pred.float() - target
    value_loss = (diff.pow(2) * mask).sum() / mask.sum().clamp(min=1)
    total = policy_loss + value_weight * value_loss

    return total, policy_loss, value_loss


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

    if device.startswith("cuda"):
        # TF32 for any fp32 matmuls outside the bf16 autocast region (e.g. the
        # fp32 value-loss path, grad-norm) — free throughput on Ampere+.
        torch.set_float32_matmul_precision("high")

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
        # Warm-start: graft a pre-think checkpoint's weights into the fresh model
        # (e.g. enabling --n-think). think_emb + grown bias cols stay zero-init.
        if config.init_from and Path(config.init_from).exists():
            res = model.warm_start_from(config.init_from, map_location="cpu")
            print(
                f"warm-start from {config.init_from}: "
                f"copied {len(res['copied'])}, kept-init {len(res['skipped'])} "
                f"({', '.join(res['skipped']) or 'none'})",
                flush=True,
            )
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

    # Keep the corpus in compact dtypes (boards uint8, moves int32, values int8)
    # instead of int64 — 8× less RAM and copy bandwidth for a multi-million-sample
    # corpus. Batches are widened to int64 on-device only where torch requires it.
    boards_t = torch.from_numpy(np.ascontiguousarray(boards_np))  # [N, 91] uint8
    moves_t = torch.from_numpy(moves_np.astype(np.int32))  # [N] int32
    values_t = torch.from_numpy(np.ascontiguousarray(values_np))  # [N] int8

    # Optional dense Pikafish value labels (Phase 4b). Aligned 1:1 with boards;
    # None when the dataset hasn't been labelled (dfc data label-eval).
    eval_np = load_eval_arrays(config.data_dir)
    has_eval = eval_np is not None
    eval_t = torch.from_numpy(np.ascontiguousarray(eval_np)) if eval_np is not None else None

    # T2: keep the whole corpus resident on the training device when it fits —
    # this removes the per-step CPU gather + synchronous H2D copy from the hot
    # loop entirely (2.5M samples ≈ 240 MB in compact dtypes).
    data_bytes = boards_t.nbytes + moves_t.nbytes + values_t.nbytes
    data_on_device = False
    if device.startswith("cuda"):
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            data_on_device = data_bytes < 0.5 * free_bytes
        except Exception:
            data_on_device = False
    elif device == "mps":
        data_on_device = data_bytes < (2 << 30)  # unified memory; stay modest
    if data_on_device:
        boards_t = boards_t.to(device)
        moves_t = moves_t.to(device)
        values_t = values_t.to(device)
        if eval_t is not None:
            eval_t = eval_t.to(device)

    # corpus_has_values: computed once so per-step logging never has to inspect
    # the mask (the loss itself is branchless — see _compute_loss).
    corpus_has_values = bool((values_np != _MASK_VALUE).any())

    train_idx_t = torch.from_numpy(train_idx).to(boards_t.device)
    val_idx_t = torch.from_numpy(val_idx).to(boards_t.device)

    train_boards = boards_t[train_idx_t]
    train_moves = moves_t[train_idx_t]
    train_values = values_t[train_idx_t]

    val_boards = boards_t[val_idx_t]
    val_moves = moves_t[val_idx_t]
    val_values = values_t[val_idx_t]

    train_eval = eval_t[train_idx_t] if eval_t is not None else None
    val_eval = eval_t[val_idx_t] if eval_t is not None else None

    # The gathers above copy; drop the originals so the corpus exists once.
    del boards_t, moves_t, values_t
    del eval_t

    n_train = len(train_boards)

    # non_blocking host→device copies are only safe on CUDA (pageable sources are
    # staged synchronously); on MPS they can race with the source temp being
    # freed and deliver garbage — so gate on the device.
    non_blocking = device.startswith("cuda")

    def _fetch_batch(
        b_arr: torch.Tensor, m_arr: torch.Tensor, v_arr: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather a batch and place it on the device in the dtypes the loss needs.

        With device-resident data only the tiny index tensor crosses the PCIe bus;
        the gather and the uint8→int64 widening both run on the GPU.
        """
        idx = idx.to(b_arr.device, non_blocking=non_blocking)
        return (
            b_arr[idx].to(device, non_blocking=non_blocking).long(),
            m_arr[idx].to(device, non_blocking=non_blocking).long(),
            v_arr[idx].to(device, non_blocking=non_blocking),
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
            "n_think": arch_cfg.n_think,
            "ffn_hidden": arch_cfg.ffn_hidden,
            "seq_len": arch_cfg.seq_len,
            "vocab_size": arch_cfg.vocab_size,
            "n_moves": arch_cfg.n_moves,
        },
        "params": model.num_params(),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "compiled": compiled,
        **_git_info(),
        "data_on_device": data_on_device,
        "has_eval_labels": has_eval,
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

    from contextlib import nullcontext  # noqa: PLC0415

    def _forward_ctx():  # type: ignore[return]
        if use_autocast:
            device_type = "cuda" if device.startswith("cuda") else device
            return torch.autocast(device_type=device_type, dtype=dtype)
        return nullcontext()

    # ------------------------------------------------------------------ eval helper
    @torch.no_grad()
    def _run_val() -> dict[str, float]:
        """Validate over the val set; return policy + VALUE-quality metrics.

        Value quality is measured only over non-masked (``!= 127``) samples:
        ``val_value_mse`` = MSE of tanh(value) vs the −1/0/+1 outcome, and
        ``val_value_corr`` = Pearson corr of the two. These are the gate for
        whether MCTS ``value_mode='head'`` can beat ``'rollout'``.
        """
        model.eval()
        total_loss_acc = 0.0
        total_policy_acc = 0.0
        total_correct = 0
        total_correct5 = 0
        total_n = 0
        # value-quality running sums over non-masked samples (for MSE + Pearson)
        vn = 0
        vse = 0.0
        sx = sy = sxx = syy = sxy = 0.0
        n_batches = max(1, math.ceil(len(val_boards) / config.batch_size))
        for bi in range(n_batches):
            s = bi * config.batch_size
            e = min(s + config.batch_size, len(val_boards))
            vb = val_boards[s:e].to(device).long()
            vm = val_moves[s:e].to(device).long()
            vv = val_values[s:e].to(device)
            ve = val_eval[s:e].to(device) if val_eval is not None else None

            with _forward_ctx():
                p_logits, v_pred = train_model(vb)

            total, p_loss, _ = _compute_loss(
                p_logits, v_pred, vm, vv, config.value_weight, ve, config.value_eval_weight
            )
            batch_n = e - s
            total_loss_acc += total.item() * batch_n
            total_policy_acc += p_loss.item() * batch_n
            preds = p_logits.argmax(dim=-1)
            total_correct += (preds == vm).sum().item()
            top5 = p_logits.topk(min(5, p_logits.shape[-1]), dim=-1).indices
            total_correct5 += (top5 == vm.unsqueeze(-1)).any(dim=-1).sum().item()
            total_n += batch_n

            vmask = vv != _MASK_VALUE
            if bool(vmask.any()):
                vp = v_pred.float()[vmask]
                vt = vv.float()[vmask]
                vn += int(vp.numel())
                vse += float(((vp - vt) ** 2).sum().item())
                sx += float(vp.sum().item())
                sy += float(vt.sum().item())
                sxx += float((vp * vp).sum().item())
                syy += float((vt * vt).sum().item())
                sxy += float((vp * vt).sum().item())

        model.train()
        n = max(total_n, 1)
        val_value_mse = (vse / vn) if vn else float("nan")
        val_value_corr = float("nan")
        if vn > 1:
            denom = (vn * sxx - sx * sx) * (vn * syy - sy * sy)
            if denom > 0:
                val_value_corr = (vn * sxy - sx * sy) / (denom**0.5)
        return {
            "val_loss": total_loss_acc / n,
            "val_policy_loss": total_policy_acc / n,
            "val_top1": total_correct / n,
            "val_top5": total_correct5 / n,
            "val_value_mse": val_value_mse,
            "val_value_corr": val_value_corr,
        }

    # ------------------------------------------------------------------ train loop
    best_val_policy = resume_best_val
    step_0_loss: float | None = None
    t_start = time.time()
    gen = torch.Generator()
    gen.manual_seed(config.seed)

    # Steady-state steps run with zero host↔device syncs: loss/grad-norm stay on
    # the GPU and are only pulled (.item()) at the logging cadence. The NaN
    # guard fires at the same cadence — a NaN loss poisons the weights either
    # way, so catching it ≤log_every steps later loses nothing.
    log_every = max(1, min(10, config.eval_every // 10))
    last_log_t = t_start
    last_log_step = resume_step - 1
    sps = 0.0
    tps = 0.0

    try:
        for step in range(resume_step, config.max_steps):
            lr = _lr_at(step, config.warmup, config.max_steps, config.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            bs = min(config.batch_size, n_train)
            idx = torch.randint(0, n_train, (bs,), generator=gen)
            b_batch, m_batch, v_batch = _fetch_batch(train_boards, train_moves, train_values, idx)
            e_batch = None
            if train_eval is not None:
                ei = idx.to(train_eval.device, non_blocking=non_blocking)
                e_batch = train_eval[ei].to(device, non_blocking=non_blocking)

            with _forward_ctx():
                p_logits, v_pred = train_model(b_batch)

            # Ramp the value weight 0 → config.value_weight over the first ~10% of
            # steps so the noisy early value head doesn't disturb policy learning.
            # (On resume, step is already high → full weight immediately.)
            vw_ramp = config.value_weight * min(
                1.0, (step + 1) / max(1, int(0.1 * config.max_steps))
            )
            total, p_loss, v_loss = _compute_loss(
                p_logits, v_pred, m_batch, v_batch, vw_ramp, e_batch, config.value_eval_weight
            )

            opt.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            do_log = step == 0 or (step + 1) % log_every == 0
            do_eval = (step + 1) % config.eval_every == 0 or step == config.max_steps - 1
            is_profile_step = config.profile and step == config.profile_at
            if not (do_log or do_eval or is_profile_step):
                continue

            # First host sync since the last log point. Throughput is measured
            # over the whole window between syncs — per-step wall timing under
            # async CUDA execution measures kernel-launch time, not step time.
            train_loss = float(total.item())
            elapsed = time.time() - t_start
            window_steps = step - last_log_step
            sps = window_steps * bs / max(time.time() - last_log_t, 1e-9)
            tps = sps * model.config.seq_len
            last_log_t = time.time()
            last_log_step = step

            # NaN guard (§5 risk)
            if not math.isfinite(train_loss):
                _write_run_json("failed", datetime.now(UTC).isoformat())
                metrics_fh.close()
                raise ValueError(f"Loss became {train_loss} at step {step}. Run marked failed.")

            if step_0_loss is None:
                step_0_loss = train_loss

            if do_log:
                _log_metrics(
                    {
                        "step": step,
                        "split": "train",
                        "loss": train_loss,
                        "policy_loss": p_loss.item(),
                        "value_loss": v_loss.item() if corpus_has_values else None,
                        "top1": None,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "samples_per_s": sps,
                        "tokens_per_s": tps,
                        "elapsed_s": elapsed,
                    }
                )

            # Validate + maybe save.
            if do_eval:
                vres = _run_val()
                val_policy_loss = vres["val_policy_loss"]
                val_elapsed = time.time() - t_start
                _log_metrics(
                    {
                        "step": step,
                        "split": "val",
                        "loss": vres["val_loss"],
                        "policy_loss": val_policy_loss,
                        # value_loss now measured: MSE of tanh(value) vs outcome.
                        "value_loss": vres["val_value_mse"] if corpus_has_values else None,
                        "value_corr": vres["val_value_corr"] if corpus_has_values else None,
                        "top1": vres["val_top1"],
                        "top5": vres["val_top5"],
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
                            "val_loss": vres["val_loss"],
                            "val_top1": vres["val_top1"],
                            "val_top5": vres["val_top5"],
                            "val_value_mse": vres["val_value_mse"],
                            "val_value_corr": vres["val_value_corr"],
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
