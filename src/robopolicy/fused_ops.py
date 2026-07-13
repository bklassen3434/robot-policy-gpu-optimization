"""Load the custom fused CUDA kernels and expose them to the model.

The kernels live in ``<repo>/kernels/csrc`` and are JIT-compiled by torch's
``cpp_extension`` (needs nvcc + ninja on a CUDA box). This module locates them
relative to the editable install and caches the compiled extension, so the model's
inference path can call the profiled fused residual+LayerNorm kernel (eval step 8).

Uses the same extension name/sources as ``kernels/reference.py`` so both share one
JIT build.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# <repo>/src/robopolicy/fused_ops.py -> parents[2] == <repo> (editable install layout).
_CSRC = Path(__file__).resolve().parents[2] / "kernels" / "csrc"


@functools.lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — the fused kernels need a GPU box.")
    from torch.utils.cpp_extension import load

    return load(
        name="fused_layernorm",
        sources=[str(_CSRC / "layernorm.cpp"), str(_CSRC / "layernorm_kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def residual_layernorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Fused ``LayerNorm(x + residual)`` via the custom CUDA kernel."""
    return _extension().residual_layernorm_forward(
        x.contiguous(), residual.contiguous(), weight.contiguous(), bias.contiguous(), eps
    )
