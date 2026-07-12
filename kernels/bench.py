"""Benchmark the custom kernel vs the PyTorch reference (CUDA-event timed).

    python kernels/bench.py --shape 800 512 --iters 1000

Prints the before/after latency and speedup that go in the README table and the
latency chart. Uses CUDA events + warmup so the numbers are real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import reference  # noqa: E402


def _time(fn, iters: int, warmup: int) -> float:
    """Return mean latency in milliseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["layernorm", "residual_layernorm"], default="residual_layernorm")
    ap.add_argument("--shape", type=int, nargs="+", default=[302, 512], help="input shape, last dim = normalized")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for benchmarking — run on the GPU box.")

    x = torch.randn(*args.shape, device="cuda")
    w = torch.randn(args.shape[-1], device="cuda")
    b = torch.randn(args.shape[-1], device="cuda")

    if args.op == "residual_layernorm":
        res = torch.randn(*args.shape, device="cuda")
        ref_fn = lambda: reference.pytorch_residual_layernorm(x, res, w, b)  # noqa: E731
        cuda_fn = lambda: reference.cuda_residual_layernorm(x, res, w, b)  # noqa: E731
    else:
        ref_fn = lambda: reference.pytorch_layernorm(x, w, b)  # noqa: E731
        cuda_fn = lambda: reference.cuda_layernorm(x, w, b)  # noqa: E731

    # correctness gate before timing
    torch.testing.assert_close(cuda_fn(), ref_fn(), rtol=1e-4, atol=1e-5)

    ref_ms = _time(ref_fn, args.iters, args.warmup)
    cuda_ms = _time(cuda_fn, args.iters, args.warmup)

    print(f"op={args.op}  shape={tuple(args.shape)}  iters={args.iters}")
    print(f"  PyTorch reference : {ref_ms * 1000:8.2f} us")
    print(f"  custom CUDA kernel: {cuda_ms * 1000:8.2f} us")
    print(f"  speedup           : {ref_ms / cuda_ms:8.2f}x")


if __name__ == "__main__":
    main()
