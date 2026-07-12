"""Run a few forward+backward steps for profiling with NSight Systems.

Uses synthetic inputs (no dataset needed) so profiling is isolated to the model.
NVTX ranges are attached to attention / layernorm / feedforward modules so the NSight
timeline is readable, and the measured region is bracketed by the CUDA profiler API so
``nsys --capture-range=cudaProfilerApi`` captures only the steady-state steps.

    python -m robopolicy.profile_step --config configs/act_aloha.yaml --steps 20
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from .model import ACT, ACTConfig
from .model.transformer import MultiheadAttention
from .utils import load_config, resolve_device, set_seed


def _nvtx_push(tag: str):
    # A forward_pre_hook must return None: any non-None return is interpreted by
    # PyTorch as replacement input args. range_push() returns an int, so the hook
    # body must not return it (that would clobber the module's inputs).
    def hook(module, inputs):
        torch.cuda.nvtx.range_push(tag)

    return hook


def _nvtx_pop(module, inputs, output):
    # Likewise a forward_hook must return None or it replaces the module output.
    torch.cuda.nvtx.range_pop()


def _attach_nvtx(model: nn.Module) -> None:
    """Emit NVTX ranges around notable submodules for a readable timeline."""
    if not torch.cuda.is_available():
        return  # NVTX is a CUDA-only concern; no-op on CPU/MPS

    def label(module: nn.Module, name: str) -> str | None:
        if isinstance(module, MultiheadAttention):
            return f"attention/{name}"
        if isinstance(module, nn.LayerNorm):
            return f"layernorm/{name}"
        if isinstance(module, nn.Linear) and module.out_features >= 2048:
            return f"ffn/{name}"
        return None

    for name, module in model.named_modules():
        tag = label(module, name)
        if tag is None:
            continue
        module.register_forward_pre_hook(_nvtx_push(tag))
        module.register_forward_hook(_nvtx_pop)


def synthetic_batch(cfg: dict, meta_dims: dict, device, batch_size: int) -> dict:
    m = cfg["model"]
    s, a = meta_dims["state_dim"], meta_dims["action_dim"]
    n_cam = meta_dims["n_cameras"]
    return {
        "observation.state": torch.randn(batch_size, s, device=device),
        "observation.images": torch.randn(batch_size, n_cam, 3, 480, 640, device=device),
        "action": torch.randn(batch_size, m["chunk_size"], a, device=device),
        "action_is_pad": torch.zeros(batch_size, m["chunk_size"], dtype=torch.bool, device=device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = resolve_device(cfg["train"].get("device", "auto"))
    if device.type != "cuda":
        print(f"WARNING: profiling is meant for CUDA; running on {device}.")

    meta_dims = {"state_dim": 14, "action_dim": 14, "n_cameras": len(cfg["dataset"]["image_keys"])}
    model_cfg = dict(cfg["model"], **meta_dims)
    model = ACT(ACTConfig.from_dict(model_cfg)).to(device).train()
    _attach_nvtx(model)

    batch = synthetic_batch(cfg, meta_dims, device, args.batch_size)

    for _ in range(args.warmup):
        model.compute_loss(batch)["loss"].backward()
        model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

    for _ in range(args.steps):
        loss = model.compute_loss(batch)["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    print(f"profiled {args.steps} steps on {device}")


if __name__ == "__main__":
    main()
