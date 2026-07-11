"""Pre-decoded image cache for fast, crash-free training.

LeRobot stores camera frames as video. Decoding them on every ``__getitem__`` with
PyAV is (a) slow — random-access seeks decode from the nearest keyframe — and (b)
not fork/-spawn-safe — FFmpeg segfaults/aborts inside DataLoader worker processes.

We sidestep both by decoding every frame **once**, sequentially, into a ``uint8``
memmap on disk, plus small in-memory ``state``/``action``/``episode`` arrays. Training
then reads frames by index: memmap reads are fork-safe (so DataLoader workers are
fine) and there is no video decode in the hot loop, so training is GPU-bound.

The cache lives under ``outputs/cache`` (persistent on the RunPod volume) and is
rebuilt only if missing.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def build_or_load_cache(base, image_keys, state_key, action_key, cache_dir):
    """Return (images_memmap, states, actions, episode_index, info).

    ``base`` is a LeRobotDataset opened WITHOUT ``delta_timestamps`` (single frames).
    On a cache hit nothing is decoded; on a miss every frame is decoded once.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = base.repo_id.replace("/", "__")
    img_path = os.path.join(cache_dir, f"{tag}__images.u8")
    meta_path = os.path.join(cache_dir, f"{tag}__meta.npz")
    info_path = os.path.join(cache_dir, f"{tag}__info.json")

    if all(os.path.exists(p) for p in (img_path, meta_path, info_path)):
        with open(info_path) as f:
            info = json.load(f)
        meta = np.load(meta_path)
        images = np.memmap(img_path, dtype=np.uint8, mode="r", shape=tuple(info["image_shape"]))
        print(f"[cache] hit: {img_path} {tuple(info['image_shape'])}", flush=True)
        return images, meta["states"], meta["actions"], meta["episode_index"], info

    n = base.num_frames
    s0 = base[0]
    ncam = len(image_keys)
    c, h, w = s0[image_keys[0]].shape
    state_dim = int(s0[state_key].shape[0])
    action_dim = int(s0[action_key].shape[0])
    image_shape = (n, ncam, int(c), int(h), int(w))

    print(f"[cache] miss: decoding {n} frames -> {image_shape} uint8 "
          f"({np.prod(image_shape) / 1e9:.1f} GB)", flush=True)
    tmp = img_path + ".tmp"
    images = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=image_shape)
    states = np.zeros((n, state_dim), dtype=np.float32)
    actions = np.zeros((n, action_dim), dtype=np.float32)
    episode_index = np.zeros((n,), dtype=np.int64)

    for i in range(n):
        s = base[i]
        for ci, k in enumerate(image_keys):
            img = (s[k].clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
            images[i, ci] = img
        states[i] = s[state_key].numpy()
        actions[i] = s[action_key].numpy()
        episode_index[i] = int(s["episode_index"])
        if i % 1000 == 0:
            print(f"[cache] {i}/{n}", flush=True)

    images.flush()
    del images
    os.replace(tmp, img_path)
    np.savez(meta_path, states=states, actions=actions, episode_index=episode_index)
    info = {
        "image_shape": list(image_shape),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_keys": list(image_keys),
        "num_frames": n,
    }
    with open(info_path, "w") as f:
        json.dump(info, f)
    print("[cache] built.", flush=True)

    images = np.memmap(img_path, dtype=np.uint8, mode="r", shape=image_shape)
    return images, states, actions, episode_index, info


def _episode_end_index(episode_index: np.ndarray) -> np.ndarray:
    """For each frame, the last frame index belonging to the same (contiguous) episode."""
    n = len(episode_index)
    ep_end = np.empty(n, dtype=np.int64)
    end = n - 1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and episode_index[i] != episode_index[i + 1]:
            end = i
        ep_end[i] = end
    return ep_end


class CachedACTDataset(Dataset):
    """Serves samples matching LeRobotDataset's structure, from the memmap cache.

    Returns per-camera images in ``[0, 1]`` plus a ``chunk``-length action window that
    is clipped at the episode boundary and padded (padded steps flagged in
    ``action_is_pad``, matching LeRobot so the same collate/loss masking applies).
    """

    def __init__(self, images, states, actions, episode_index,
                 image_keys, state_key, action_key, chunk):
        self.images = images
        self.states = states
        self.actions = actions
        self.image_keys = list(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.chunk = int(chunk)
        self.ep_end = _episode_end_index(episode_index)
        self.action_dim = actions.shape[1]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        end = int(self.ep_end[i])
        avail = min(self.chunk, end - i + 1)

        chunk_actions = np.zeros((self.chunk, self.action_dim), dtype=np.float32)
        chunk_actions[:avail] = self.actions[i:i + avail]
        if avail < self.chunk:
            chunk_actions[avail:] = self.actions[i + avail - 1]  # repeat last (masked anyway)
        is_pad = np.zeros((self.chunk,), dtype=bool)
        is_pad[avail:] = True

        sample = {}
        for ci, k in enumerate(self.image_keys):
            frame = np.ascontiguousarray(self.images[i, ci])  # (C,H,W) uint8
            sample[k] = torch.from_numpy(frame).float() / 255.0
        sample[self.state_key] = torch.from_numpy(self.states[i].copy())
        sample[self.action_key] = torch.from_numpy(chunk_actions)
        sample["action_is_pad"] = torch.from_numpy(is_pad)
        return sample
