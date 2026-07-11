"""LeRobot dataset loading, action chunking, and normalization.

``lerobot`` is a heavy dependency (only needed to actually train/eval), so it's
imported lazily — the model and kernel code import cleanly on a plain Mac without it.

Key detail: ACT predicts a *chunk* of future actions. We ask ``LeRobotDataset`` for
``chunk_size`` future action frames per sample via ``delta_timestamps``; it returns
``action`` of shape ``(chunk, action_dim)`` and a boolean ``action_is_pad`` marking
frames that ran past the end of an episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..model.backbone import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class Normalizer:
    """Mean/std normalization for state & action; ImageNet normalization for images."""

    state_mean: Tensor
    state_std: Tensor
    action_mean: Tensor
    action_std: Tensor

    def to(self, device) -> "Normalizer":
        return Normalizer(
            self.state_mean.to(device), self.state_std.to(device),
            self.action_mean.to(device), self.action_std.to(device),
        )

    def normalize_state(self, x: Tensor) -> Tensor:
        return (x - self.state_mean) / self.state_std

    def normalize_action(self, x: Tensor) -> Tensor:
        return (x - self.action_mean) / self.action_std

    def unnormalize_action(self, x: Tensor) -> Tensor:
        return x * self.action_std + self.action_mean

    @staticmethod
    def normalize_image(x: Tensor) -> Tensor:
        # x in [0,1], shape (..., 3, H, W)
        return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)

    @classmethod
    def from_stats(cls, stats: dict, state_key: str, action_key: str) -> "Normalizer":
        eps = 1e-6

        def ms(key):
            s = stats[key]
            if "mean" in s and "std" in s:
                mean = torch.as_tensor(s["mean"], dtype=torch.float32).flatten()
                std = torch.as_tensor(s["std"], dtype=torch.float32).flatten().clamp(min=eps)
            else:  # fall back to min/max -> center/half-range normalization
                lo = torch.as_tensor(s["min"], dtype=torch.float32).flatten()
                hi = torch.as_tensor(s["max"], dtype=torch.float32).flatten()
                mean = (hi + lo) / 2
                std = ((hi - lo) / 2).clamp(min=eps)
            return mean, std

        sm, ss = ms(state_key)
        am, as_ = ms(action_key)
        return cls(sm, ss, am, as_)


def make_collate(image_keys: list[str], state_key: str, action_key: str, normalizer: Normalizer):
    """Build a collate_fn that stacks LeRobot samples into a model-ready batch."""

    def collate(samples: list[dict]) -> dict:
        state = torch.stack([s[state_key] for s in samples])  # (B, state_dim)
        # each image key -> (B,3,H,W); stack cameras on dim 1
        cams = [torch.stack([s[k] for s in samples]) for k in image_keys]
        images = torch.stack(cams, dim=1)  # (B, n_cam, 3, H, W)
        images = Normalizer.normalize_image(images)

        batch = {
            "observation.state": normalizer.normalize_state(state),
            "observation.images": images,
        }
        if action_key in samples[0]:
            action = torch.stack([s[action_key] for s in samples])  # (B, chunk, action_dim)
            batch["action"] = normalizer.normalize_action(action)
        if "action_is_pad" in samples[0]:
            batch["action_is_pad"] = torch.stack([s["action_is_pad"] for s in samples])
        return batch

    return collate


# convenience re-export for tests / external callers
def collate_batch(samples, image_keys, state_key, action_key, normalizer):
    return make_collate(image_keys, state_key, action_key, normalizer)(samples)


def build_dataloaders(cfg: dict, seed: int = 0):
    """Return (train_loader, val_loader, meta) for the configured dataset.

    ``meta`` carries ``state_dim``, ``action_dim``, ``n_cameras``, ``image_keys``,
    ``fps`` and the ``Normalizer`` needed to build the model and un/normalize actions.
    """
    try:
        # LeRobot 0.6.x import path (dropped the old `lerobot.common.*` namespace).
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            'lerobot is required to load data. Install it: pip install -e ".[data]"'
        ) from exc

    dcfg = cfg["dataset"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    repo_id = dcfg["repo_id"]
    state_key = dcfg["state_key"]
    action_key = dcfg["action_key"]
    chunk = mcfg["chunk_size"]
    # PyAV decodes the video-backed image frames; LeRobot 0.6's default torchcodec
    # backend needs system FFmpeg libs and is ABI-pinned to a specific torch build.
    video_backend = dcfg.get("video_backend", "pyav")

    # first open the dataset to read fps + feature keys
    probe = LeRobotDataset(repo_id, video_backend=video_backend)
    fps = probe.fps
    features = list(probe.features)
    image_keys = dcfg.get("image_keys") or [k for k in features if k.startswith("observation.images")]

    # chunk of future actions
    delta_timestamps = {action_key: [i / fps for i in range(chunk)]}
    ds = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, video_backend=video_backend)

    normalizer = Normalizer.from_stats(ds.meta.stats, state_key, action_key)
    state_dim = normalizer.state_mean.numel()
    action_dim = normalizer.action_mean.numel()

    # train / val split
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * dcfg.get("val_fraction", 0.1))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    collate = make_collate(image_keys, state_key, action_key, normalizer)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg.get("num_workers", 4), collate_fn=collate, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg.get("num_workers", 4), collate_fn=collate,
    )
    meta = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_cameras": len(image_keys),
        "image_keys": image_keys,
        "fps": fps,
        "normalizer": normalizer,
    }
    return train_loader, val_loader, meta
