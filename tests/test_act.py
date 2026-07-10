"""Unit tests for the ACT policy — shapes, loss, inference queue, and an overfit check.

Uses a tiny config and a small fake image so it runs fast on CPU. The overfit test is
the real correctness signal: a working policy drives L1 on one batch toward zero.
"""

from __future__ import annotations

import torch

from robopolicy.model import ACT, ACTConfig


def tiny_cfg(**overrides) -> ACTConfig:
    base = dict(
        state_dim=6, action_dim=6, n_cameras=1,
        dim_model=32, n_heads=4, dim_feedforward=64,
        n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=1,
        dropout=0.0, latent_dim=8, chunk_size=5, n_action_steps=5,
        pretrained_backbone=False,
    )
    base.update(overrides)
    return ACTConfig(**base)


def fake_batch(cfg: ACTConfig, batch=2, h=64, w=64) -> dict:
    return {
        "observation.state": torch.randn(batch, cfg.state_dim),
        "observation.images": torch.randn(batch, cfg.n_cameras, 3, h, w),
        "action": torch.randn(batch, cfg.chunk_size, cfg.action_dim),
        "action_is_pad": torch.zeros(batch, cfg.chunk_size, dtype=torch.bool),
    }


def test_forward_shapes():
    cfg = tiny_cfg()
    model = ACT(cfg).eval()
    batch = fake_batch(cfg)
    actions, (mu, logvar) = model(batch)
    assert actions.shape == (2, cfg.chunk_size, cfg.action_dim)
    assert mu.shape == (2, cfg.latent_dim)
    assert logvar.shape == (2, cfg.latent_dim)


def test_inference_latent_is_zero_path():
    cfg = tiny_cfg()
    model = ACT(cfg).eval()
    batch = fake_batch(cfg)
    del batch["action"]  # no action -> VAE skipped, latent=0
    actions, (mu, logvar) = model(batch)
    assert actions.shape == (2, cfg.chunk_size, cfg.action_dim)
    assert mu is None and logvar is None


def test_loss_has_l1_and_kld():
    cfg = tiny_cfg()
    model = ACT(cfg)
    out = model.compute_loss(fake_batch(cfg))
    assert "l1" in out and "kld" in out and "loss" in out
    assert out["loss"].requires_grad


def test_select_action_queue():
    cfg = tiny_cfg()
    model = ACT(cfg).eval()
    obs = {
        "observation.state": torch.randn(1, cfg.state_dim),
        "observation.images": torch.randn(1, cfg.n_cameras, 3, 64, 64),
    }
    model.reset()
    a1 = model.select_action(obs)
    assert a1.shape == (1, cfg.action_dim)
    # serves n_action_steps actions from one forward pass
    for _ in range(cfg.n_action_steps - 1):
        model.select_action(obs)
    assert len(model._action_queue) == 0


def test_overfit_one_batch():
    torch.manual_seed(0)
    cfg = tiny_cfg(use_vae=False)  # deterministic (no latent sampling) for a clean overfit
    model = ACT(cfg).train()
    batch = fake_batch(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = model.compute_loss(batch)["l1"].item()
    for _ in range(200):
        opt.zero_grad()
        loss = model.compute_loss(batch)["loss"]
        loss.backward()
        opt.step()
    last = model.compute_loss(batch)["l1"].item()
    assert last < first * 0.25, f"failed to overfit: {first:.3f} -> {last:.3f}"
