"""Measure a GPU's real capabilities to catch a mislabeled/underspec'd card.

Reports the *reported* identity (name, VRAM, compute capability, PCIe link) AND
the *measured* throughput (matmul TFLOPS at fp64/fp32/tf32/fp16/bf16 + VRAM
bandwidth). A provider swapping in a weaker card is caught because measured
tensor-core TFLOPS and memory bandwidth reveal the true silicon regardless of
what the name string says.

Run via the project venv so torch is available:
    uv run python deploy/files/gpu_bench.py

Prints a human table, then a single JSON line (prefixed ``JSON:``) for parsing.
"""

from __future__ import annotations

import json
import time

import torch

# Rough vendor peak specs for a few cards (dense, no sparsity) — used only to
# flag a gross mismatch. Values are approximate; treat <~70% as suspicious.
# keys: substrings matched against the reported device name (lowercased).
_REFERENCE: dict[str, dict[str, float]] = {
    "5090": {"bf16_tflops": 209, "fp16_tflops": 209, "tf32_tflops": 105, "membw_gbps": 1790, "vram_gb": 32},
    "5080": {"bf16_tflops": 112, "fp16_tflops": 112, "tf32_tflops": 56, "membw_gbps": 960, "vram_gb": 16},
    "5070": {"bf16_tflops": 62, "fp16_tflops": 62, "tf32_tflops": 31, "membw_gbps": 672, "vram_gb": 12},
    "5060": {"bf16_tflops": 48, "fp16_tflops": 48, "tf32_tflops": 24, "membw_gbps": 448, "vram_gb": 8},
    "4090": {"bf16_tflops": 165, "fp16_tflops": 165, "tf32_tflops": 83, "membw_gbps": 1008, "vram_gb": 24},
    "a100": {"bf16_tflops": 312, "fp16_tflops": 312, "tf32_tflops": 156, "membw_gbps": 1935, "vram_gb": 40},
    "h100": {"bf16_tflops": 989, "fp16_tflops": 989, "tf32_tflops": 495, "membw_gbps": 3350, "vram_gb": 80},
}


def _bench_matmul(dtype: torch.dtype, n: int, iters: int = 50, warmup: int = 10) -> float:
    """Return sustained matmul TFLOP/s for an n×n @ n×n product in ``dtype``."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return (2.0 * n**3) / dt / 1e12


def _bench_membw(n_elems: int = 1 << 27, iters: int = 100) -> float:
    """Return device memory bandwidth (GB/s) via a large fp32 copy (read+write)."""
    a = torch.empty(n_elems, device="cuda", dtype=torch.float32)
    b = torch.empty(n_elems, device="cuda", dtype=torch.float32)
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return (2.0 * n_elems * 4) / dt / 1e9


def main() -> None:
    if not torch.cuda.is_available():
        print("NO CUDA GPU visible to torch — cannot benchmark.")
        print("JSON:" + json.dumps({"cuda": False}))
        return

    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    name = props.name
    cc = f"{props.major}.{props.minor}"
    vram_gb = round(props.total_memory / 1e9, 1)

    # Enable TF32 path for the tf32 measurement only (restored after).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results: dict[str, float | None] = {}
    # fp64 is heavily rate-limited on consumer cards (a useful fingerprint).
    try:
        results["fp64_tflops"] = round(_bench_matmul(torch.float64, 4096), 2)
    except Exception:
        results["fp64_tflops"] = None
    torch.backends.cuda.matmul.allow_tf32 = False
    results["fp32_tflops"] = round(_bench_matmul(torch.float32, 8192), 2)
    torch.backends.cuda.matmul.allow_tf32 = True
    results["tf32_tflops"] = round(_bench_matmul(torch.float32, 8192), 2)
    results["fp16_tflops"] = round(_bench_matmul(torch.float16, 8192), 2)
    try:
        results["bf16_tflops"] = round(_bench_matmul(torch.bfloat16, 8192), 2)
    except Exception:
        results["bf16_tflops"] = None
    results["membw_gbps"] = round(_bench_membw(), 1)

    report: dict[str, object] = {
        "cuda": True,
        "name": name,
        "compute_capability": cc,
        "vram_gb": vram_gb,
        "driver": torch.version.cuda,
        "sm_count": props.multi_processor_count,
        **results,
    }

    # Flag mismatch vs known reference (if the name matches a known card).
    ref_key = next((k for k in _REFERENCE if k in name.lower()), None)
    verdict = "unknown-card (no reference to compare)"
    if ref_key:
        ref = _REFERENCE[ref_key]
        bf = results.get("bf16_tflops") or results.get("fp16_tflops") or 0
        bw = results.get("membw_gbps") or 0
        bf_ok = bf >= 0.7 * ref["bf16_tflops"]
        bw_ok = bw >= 0.7 * ref["membw_gbps"]
        vram_ok = vram_gb >= 0.9 * ref["vram_gb"]
        report["reference"] = ref
        report["checks"] = {"bf16": bf_ok, "membw": bw_ok, "vram": vram_ok}
        verdict = (
            "OK — matches reference"
            if (bf_ok and bw_ok and vram_ok)
            else "⚠ SUSPICIOUS — measured well below reference; possible mislabeled card"
        )
    report["verdict"] = verdict

    print("=" * 60)
    print(f"GPU (reported):  {name}")
    print(f"Compute cap:     {cc}   SMs: {props.multi_processor_count}")
    print(f"VRAM:            {vram_gb} GB   CUDA: {torch.version.cuda}")
    print("-" * 60)
    print("MEASURED throughput (matmul, dense):")
    for k in ("fp64_tflops", "fp32_tflops", "tf32_tflops", "fp16_tflops", "bf16_tflops"):
        v = results.get(k)
        print(f"  {k:14s}: {v if v is not None else 'n/a':>8} TFLOP/s")
    print(f"  {'membw':14s}: {results['membw_gbps']:>8} GB/s")
    print("-" * 60)
    if ref_key:
        print(f"Reference ({ref_key}): {report['reference']}")
    print(f"VERDICT: {verdict}")
    print("=" * 60)
    print("JSON:" + json.dumps(report))


if __name__ == "__main__":
    main()
