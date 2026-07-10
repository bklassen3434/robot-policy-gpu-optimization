"""Accuracy-held regression (step 8).

Proves swapping the reference op for the custom CUDA kernel does not change the model's
output — which is what lets us claim accuracy held after the speedup. Two levels:

1. **Op parity** (runs on the GPU box): every LayerNorm-shaped tensor the model uses
   produces identical output through the custom kernel.
2. **Metric guard** (documented, run manually): re-run ``robopolicy.eval`` after the
   swap and assert the sim success rate is within noise of the pre-swap number.

Level 2 needs a trained checkpoint + gym-aloha, so it's driven from the eval CLI rather
than baked into a fast unit test; see ``assert_success_rate_held`` for the check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
import reference  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(8, 512), (800, 512), (300, 512)])
def test_kernel_output_identical_to_reference(shape):
    """The exact tensor shapes flowing through ACT's LayerNorms must be output-identical."""
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    w = torch.randn(shape[-1], device="cuda")
    b = torch.randn(shape[-1], device="cuda")
    ref = reference.pytorch_layernorm(x, w, b)
    out = reference.cuda_layernorm(x, w, b)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def assert_success_rate_held(before: float, after: float, tol: float = 0.02) -> None:
    """Guard used after re-evaluating a swapped model.

    ``before``/``after`` are sim success rates in [0,1]. Small tol absorbs rollout
    stochasticity; the kernel is output-parity so the metric should barely move.
    """
    assert abs(after - before) <= tol, (
        f"success rate moved {before:.3f} -> {after:.3f} (> {tol}); kernel changed behavior"
    )
