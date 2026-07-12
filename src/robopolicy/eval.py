"""Evaluation: validation L1 on held-out data + sim rollout success rate.

    python -m robopolicy.eval --config configs/act_aloha.yaml --checkpoint outputs/last.pt

Validation L1 runs anywhere. The sim rollout needs ``gym-aloha`` (MuJoCo), which is
installed on the GPU box via the ``[data]`` extra. Success rate is *the* number that
proves task accuracy — track it before and after the CUDA kernel swap.
"""

from __future__ import annotations

import argparse

import torch

from .model import ACT, ACTConfig
from .utils import load_config, move_to, resolve_device


def load_checkpoint(path: str, device) -> tuple[ACT, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ACTConfig(**ckpt["cfg"])
    model = ACT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("meta_keys", {})


@torch.no_grad()
def eval_val_l1(model: ACT, val_loader, device, max_batches: int = 200) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        batch = move_to(batch, device)
        total += model.compute_loss(batch)["l1"].item()
        n += 1
        if n >= max_batches:
            break
    return total / max(n, 1)


@torch.no_grad()
def eval_sim_success(model: ACT, cfg: dict, meta: dict, device) -> float:
    """Roll out the policy in gym-aloha and return the task success rate.

    Requires ``gym-aloha``. This is the accuracy metric to keep constant across the
    kernel swap (step 8).
    """
    try:
        import gym_aloha  # noqa: F401
        import gymnasium as gym
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            'gym-aloha is required for sim eval. Install it: pip install -e ".[data]"'
        ) from exc

    from .data import Normalizer

    ecfg = cfg["eval"]
    normalizer: Normalizer = meta["normalizer"].to(device)
    image_keys = meta["image_keys"]

    # obs_type="pixels_agent_pos" -> obs = {"pixels": {"top": HWC uint8}, "agent_pos": (14,)};
    # the default "pixels" omits the joint state that the policy needs.
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=ecfg["max_episode_steps"],
    )
    successes = 0
    n_ep = ecfg["n_episodes"]
    for ep in range(n_ep):
        obs, _ = env.reset(seed=ep)
        model.reset()
        done = False
        ep_success = False
        while not done:
            observation = _obs_to_batch(obs, image_keys, normalizer, device)
            action = model.select_action(observation)  # normalized (1, action_dim)
            action = normalizer.unnormalize_action(action)[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_success = ep_success or bool(info.get("is_success", reward > 0))
            done = terminated or truncated
        successes += int(ep_success)
        print(f"[sim] episode {ep + 1}/{n_ep} success={ep_success} (running {successes}/{ep + 1})")
    env.close()
    return successes / n_ep


def _obs_to_batch(obs, image_keys, normalizer, device):
    import numpy as np

    state = torch.as_tensor(np.asarray(obs["agent_pos"]), dtype=torch.float32, device=device).unsqueeze(0)
    pixels = obs["pixels"]  # dict of camera -> HWC uint8
    cams = []
    for k in image_keys:
        cam_name = k.split(".")[-1]
        img = pixels[cam_name] if isinstance(pixels, dict) else pixels
        t = torch.as_tensor(np.asarray(img), dtype=torch.float32, device=device) / 255.0
        cams.append(t.permute(2, 0, 1))  # CHW
    images = torch.stack(cams, dim=0).unsqueeze(0)  # (1, n_cam, 3, H, W)
    images = Normalizer.normalize_image(images)
    return {
        "observation.state": normalizer.normalize_state(state),
        "observation.images": images,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--no-sim", action="store_true", help="skip the sim rollout")
    ap.add_argument("--with-val-l1", action="store_true",
                    help="also compute val L1 (builds the image cache; slow)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["train"].get("device", "auto"))
    model, _ = load_checkpoint(args.checkpoint, device)

    # Cheap metadata path (normalizer + dims), no image cache — enough for sim rollout.
    from .data import build_meta

    meta = build_meta(cfg)

    if args.with_val_l1:
        from .data import build_dataloaders

        _, val_loader, _ = build_dataloaders(cfg, seed=cfg["train"]["seed"])
        val_l1 = eval_val_l1(model, val_loader, device)
        print(f"validation L1: {val_l1:.4f}")

    if not args.no_sim:
        success = eval_sim_success(model, cfg, meta, device)
        print(f"sim success rate: {success * 100:.1f}%")


if __name__ == "__main__":
    main()
