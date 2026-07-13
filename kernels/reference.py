"""Reference op + custom-kernel loader.

The op here is a **template**: fused LayerNorm forward. LayerNorm is the simplest
transformer op to get bit-close parity on, so it's a good first kernel to wire the
whole harness (parity + benchmark) end to end. After profiling (step 5), swap this
reference and the ``.cu`` kernel for whatever NSight flags as the hottest op — the
parity/benchmark harness doesn't change.

- ``pytorch_layernorm`` — the reference (source of truth for correctness).
- ``load_cuda_layernorm`` — JIT-compiles ``csrc/layernorm_kernel.cu`` via
  ``torch.utils.cpp_extension.load`` (needs a CUDA box with nvcc + ninja).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

_CSRC = Path(__file__).parent / "csrc"


def pytorch_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference LayerNorm over the last dimension."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def pytorch_residual_layernorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Reference residual-add + LayerNorm: ``LayerNorm(x + residual)``.

    This is the exact post-norm sublayer pattern (``norm(x + sublayer(x))``) the profile
    flagged, expressed as the two PyTorch kernels the fused kernel replaces: an
    elementwise add followed by ``F.layer_norm``.
    """
    return F.layer_norm(x + residual, (x.shape[-1],), weight, bias, eps)


@functools.lru_cache(maxsize=1)
def load_cuda_layernorm():
    """Compile & load the custom CUDA LayerNorm extension (cached)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — build the kernel on the GPU box.")
    from torch.utils.cpp_extension import load

    return load(
        name="fused_layernorm",
        sources=[str(_CSRC / "layernorm.cpp"), str(_CSRC / "layernorm_kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )


def cuda_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Custom fused LayerNorm. Output must match ``pytorch_layernorm`` within tolerance."""
    ext = load_cuda_layernorm()
    return ext.layernorm_forward(x.contiguous(), weight.contiguous(), bias.contiguous(), eps)


def cuda_residual_layernorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Custom fused residual-add + LayerNorm. Must match ``pytorch_residual_layernorm``."""
    ext = load_cuda_layernorm()
    return ext.residual_layernorm_forward(
        x.contiguous(), residual.contiguous(), weight.contiguous(), bias.contiguous(), eps
    )
