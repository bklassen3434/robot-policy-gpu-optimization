"""Training loop for the from-scratch ACT policy.

    python -m robopolicy.train --config configs/act_aloha.yaml
    python -m robopolicy.train --config configs/act_aloha.yaml --overfit-one-batch

``--overfit-one-batch`` trains on a single batch to near-zero loss — a fast
correctness check that runs on CPU/MPS without the GPU box.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from .model import ACT, ACTConfig
from .utils import load_config, move_to, resolve_device, set_seed


def build_model_from_meta(cfg: dict, meta: dict) -> ACT:
    model_cfg = dict(cfg["model"])
    model_cfg.update(
        state_dim=meta["state_dim"],
        action_dim=meta["action_dim"],
        n_cameras=meta["n_cameras"],
    )
    return ACT(ACTConfig.from_dict(model_cfg))


def make_optimizer(model: ACT, tcfg: dict) -> torch.optim.Optimizer:
    # separate (lower) lr for the pretrained backbone, like official ACT
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else other_params).append(p)
    groups = [
        {"params": other_params, "lr": tcfg["lr"]},
        {"params": backbone_params, "lr": tcfg.get("lr_backbone", tcfg["lr"])},
    ]
    return torch.optim.AdamW(groups, weight_decay=tcfg.get("weight_decay", 1e-4))


def train(config_path: str, overfit_one_batch: bool = False) -> None:
    cfg = load_config(config_path)
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    device = resolve_device(tcfg.get("device", "auto"))
    print(f"device: {device}")

    from .data import build_dataloaders  # lazy: needs lerobot

    train_loader, val_loader, meta = build_dataloaders(cfg, seed=tcfg["seed"])
    model = build_model_from_meta(cfg, meta).to(device)
    opt = make_optimizer(model, tcfg)
    grad_clip = tcfg.get("grad_clip_norm", 10.0)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    if overfit_one_batch:
        batch = move_to(next(iter(train_loader)), device)
        model.train()
        for step in range(2000):
            opt.zero_grad()
            out = model.compute_loss(batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if step % 100 == 0:
                print(f"step {step:4d} | loss {out['loss'].item():.4f} | l1 {out['l1'].item():.4f}")
        return

    model.train()
    steps = tcfg["steps"]
    loader = itertools.cycle(train_loader)
    for step in range(1, steps + 1):
        batch = move_to(next(loader), device)
        opt.zero_grad()
        out = model.compute_loss(batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % 100 == 0:
            msg = f"step {step}/{steps} | loss {out['loss'].item():.4f} | l1 {out['l1'].item():.4f}"
            if "kld" in out:
                msg += f" | kld {out['kld'].item():.4f}"
            print(msg)
        if step % tcfg.get("save_every", 10000) == 0:
            _save(model, meta, out_dir / "last.pt")
        if step % tcfg.get("eval_every", 20000) == 0:
            _validate(model, val_loader, device)

    _save(model, meta, out_dir / "last.pt")


def _save(model: ACT, meta: dict, path: Path) -> None:
    torch.save(
        {"model": model.state_dict(), "cfg": model.cfg.__dict__, "meta_keys": {
            "state_dim": meta["state_dim"], "action_dim": meta["action_dim"],
            "n_cameras": meta["n_cameras"], "image_keys": meta["image_keys"],
        }},
        path,
    )
    print(f"saved -> {path}")


@torch.no_grad()
def _validate(model: ACT, val_loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        batch = move_to(batch, device)
        out = model.compute_loss(batch)
        total += out["l1"].item()
        n += 1
        if n >= 50:
            break
    model.train()
    avg = total / max(n, 1)
    print(f"[val] L1 {avg:.4f}")
    return avg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit-one-batch", action="store_true")
    args = ap.parse_args()
    train(args.config, overfit_one_batch=args.overfit_one_batch)


if __name__ == "__main__":
    main()
