"""Parity: the custom CUDA kernel must match the PyTorch reference.

Skips automatically without CUDA (so it's green on the Mac and only really runs on the
GPU box). This is the guardrail behind "keeping the output identical" (step 6) and
feeds the accuracy-held claim (step 8).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import reference  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel parity")


@pytest.mark.parametrize("shape", [(8, 512), (100 * 8, 512), (300, 512), (1, 3200)])
def test_layernorm_matches_reference(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    w = torch.randn(shape[-1], device="cuda")
    b = torch.randn(shape[-1], device="cuda")

    ref = reference.pytorch_layernorm(x, w, b)
    out = reference.cuda_layernorm(x, w, b)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_layernorm_preserves_shape():
    x = torch.randn(4, 100, 512, device="cuda")
    w = torch.ones(512, device="cuda")
    b = torch.zeros(512, device="cuda")
    out = reference.cuda_layernorm(x, w, b)
    assert out.shape == x.shape
