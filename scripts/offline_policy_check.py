#!/usr/bin/env python
"""Offline sanity check: does the fine-tuned SmolVLA reproduce its OWN training data?

Isolates "is the model broken?" from "is the real-robot I/O broken?". We push real
frames from the training dataset through the EXACT inference pipeline the rollout uses
(preprocessor -> select_action -> postprocessor) and compare the predicted action to
the action a human actually teleoperated at that frame.

  predicted ~= recorded  ->  model + normalization are FINE. The crazy/jerky real-arm
                             behavior is deployment-loop only (inference latency, action
                             smoothing, cameras, state units) -- NOT the checkpoint.
  predicted is wild      ->  the model itself predicts junk; retrain / more data.

Run from the LeRobot venv:
    cd ~/lerobot && source .venv/bin/activate && export HF_HUB_DISABLE_XET=1
    python /Users/benklassen/conductor/workspaces/robot-policy-gpu-optimization/dublin/scripts/offline_policy_check.py
"""

from __future__ import annotations

import time

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

REPO = "bklassen3434/so101_pick_object_20260713_221513"
ROOT = "/Users/benklassen/.cache/huggingface/lerobot/bklassen3434/so101_pick_object_20260713_221513"
POLICY = "bklassen3434/smolvla_so101"
DEVICE = "cpu"  # cpu = numerics identical to training; also times a single inference honestly

RENAME = {
    "observation.images.overhead": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


def main() -> None:
    ds = LeRobotDataset(REPO, root=ROOT)

    cfg = PreTrainedConfig.from_pretrained(POLICY)
    cfg.pretrained_path = POLICY
    policy = get_policy_class(cfg.type).from_pretrained(POLICY).to(DEVICE).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=POLICY,
        preprocessor_overrides={
            "device_processor": {"device": DEVICE},
            "rename_observations_processor": {"rename_map": RENAME},
        },
    )

    ep0_len = int(ds.meta.episodes[0]["length"])
    idxs = [int(ep0_len * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]

    print(f"\n{'idx':>5}  {'infer_s':>7}  {'|pred-gt|':>9}  pred vs ground-truth action (degrees)")
    for i in idxs:
        item = ds[i]
        obs = {
            "observation.state": item["observation.state"],
            "observation.images.overhead": item["observation.images.overhead"],
            "observation.images.wrist": item["observation.images.wrist"],
            "task": item["task"] if isinstance(item.get("task"), str) else "pick up the sanitizer",
        }
        policy.reset()  # clear the action queue -> re-infer fresh from THIS frame
        t0 = time.perf_counter()
        with torch.inference_mode():
            action = post(policy.select_action(pre(obs)))
        dt = time.perf_counter() - t0

        pred = action.squeeze(0).cpu().float()
        gt = item["action"].cpu().float()
        if gt.ndim > 1:
            gt = gt[0]
        err = (pred - gt).abs().mean().item()
        print(f"{i:>5}  {dt:>7.2f}  {err:>9.3f}  pred={pred.numpy().round(1)}  gt={gt.numpy().round(1)}")

    # Reference: the per-joint spread of the whole dataset, so |pred-gt| is interpretable.
    span = (torch.as_tensor(ds.meta.stats["action"]["max"]) - torch.as_tensor(ds.meta.stats["action"]["min"]))
    print(f"\nper-joint action range (max-min): {span.numpy().round(1)}  mean range={span.mean():.1f} deg")
    print(
        "Interpret: |pred-gt| a few degrees (<~5% of range) => model GOOD; chase the deploy loop "
        "(inference latency/smoothing/cameras). Tens of degrees / wild => model predicts junk.\n"
        "Also note infer_s: if one inference is ~0.5-2s, the 30Hz control loop is starving -> jitter."
    )


if __name__ == "__main__":
    main()
