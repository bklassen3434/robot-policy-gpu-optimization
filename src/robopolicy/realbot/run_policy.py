"""Load a fine-tuned SmolVLA and run it on the SO-101 from an instruction string.

This is the inference/control half of Path B. It runs on the M3 Mac (MPS). The
LeRobot touch-points (robot class, policy class, observation/action key names)
moved across releases — every one is marked `VERIFY:` so you can reconcile with
the installed version (§7, §12). The control structure around them is stable.

Typed smoke test (Phase 3, before voice):
    python -m robopolicy.realbot.run_policy --instruction "pick up the red block"
"""

from __future__ import annotations

import argparse
import sys
import time

from .config import DEFAULT_CONFIG, has_placeholder, load_config


# ─────────────────────────────────────────────────────────────────────────────
# Version-tolerant LeRobot imports. LeRobot renamed lerobot.common.* -> lerobot.*
# around 0.6; try both so this works on either.
# ─────────────────────────────────────────────────────────────────────────────
def _import_first(paths: list[tuple[str, str]], what: str):
    """Return the first importable attr from a list of (module, attr) candidates."""
    import importlib
    errors = []
    for module, attr in paths:
        try:
            return getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError) as e:
            errors.append(f"  {module}.{attr}: {e}")
    raise ImportError(
        f"Could not import {what} from any known LeRobot path. Tried:\n"
        + "\n".join(errors)
        + "\n\nInstall with:  pip install -e '.[realbot]'  and VERIFY the import "
          "path against your installed lerobot version (§12)."
    )


def _load_smolvla(policy_path: str, device):
    """Load a fine-tuned SmolVLA policy from a local dir or HF repo id."""
    SmolVLAPolicy = _import_first([
        ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
        ("lerobot.common.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
    ], "SmolVLAPolicy")
    policy = SmolVLAPolicy.from_pretrained(policy_path)  # VERIFY: from_pretrained signature
    policy.to(device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    return policy


def _apply_action_horizon(policy, runtime: dict) -> None:
    """Set how many actions per predicted chunk are executed before re-planning.

    SmolVLA predicts an action *chunk* (~chunk_size, default ~50). `select_action`
    runs the model only when its internal queue is empty and replays the queued
    actions on every other call, IGNORING new observations. At the default horizon
    the arm executes a full ~1.6 s chunk open-loop / blind to the cameras, which on a
    real robot shows up as "moves roughly right, then acts on a stale plan and wanders"
    (compounding covariate shift). Lowering `n_action_steps` forces frequent closed-loop
    re-planning at the cost of more inference calls. This is the single biggest lever
    for erratic real-arm rollouts. See runtime.n_action_steps / chunk_execution.
    """
    n = runtime.get("n_action_steps")
    if n is None and str(runtime.get("chunk_execution", "full")).lower() == "replan":
        n = 1  # legacy knob: "replan" == re-plan every control step
    cfg = getattr(policy, "config", None)
    if cfg is None or not hasattr(cfg, "n_action_steps"):
        return
    if n is None:
        print(f"[run_policy] n_action_steps unchanged (={cfg.n_action_steps}, "
              f"chunk_size={getattr(cfg, 'chunk_size', '?')})")
        return
    chunk = getattr(cfg, "chunk_size", None)
    n = max(1, min(int(n), int(chunk))) if chunk else max(1, int(n))
    old, cfg.n_action_steps = cfg.n_action_steps, n
    print(f"[run_policy] n_action_steps {old} -> {n}  (re-plan every {n} step(s), "
          f"chunk_size={chunk})")


def _build_robot(robot_cfg: dict):
    """Instantiate + connect the SO-101 follower from the config's robot block."""
    # Module is `so_follower` in recent lerobot (exports SO101Follower alias);
    # older layouts used so101_follower / lerobot.common.*.
    SO101Follower = _import_first([
        ("lerobot.robots.so_follower", "SO101Follower"),
        ("lerobot.robots.so101_follower", "SO101Follower"),
        ("lerobot.common.robots.so101_follower", "SO101Follower"),
    ], "SO101Follower")
    SO101FollowerConfig = _import_first([
        ("lerobot.robots.so_follower", "SO101FollowerConfig"),
        ("lerobot.robots.so101_follower", "SO101FollowerConfig"),
        ("lerobot.common.robots.so101_follower", "SO101FollowerConfig"),
    ], "SO101FollowerConfig")

    # VERIFY: camera config construction. Recent lerobot wants OpenCVCameraConfig
    # objects keyed by name; the index_or_path field name may differ by version.
    cameras = _build_cameras(robot_cfg.get("cameras", {}))
    cfg = SO101FollowerConfig(
        port=robot_cfg["port"],
        id=robot_cfg.get("id", "so101_follower"),
        cameras=cameras,
    )
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def _build_cameras(cams: dict) -> dict:
    """Turn the yaml camera block into lerobot camera configs, keyed by name."""
    OpenCVCameraConfig = _import_first([
        ("lerobot.cameras.opencv.configuration_opencv", "OpenCVCameraConfig"),
        ("lerobot.common.cameras.opencv.configuration_opencv", "OpenCVCameraConfig"),
    ], "OpenCVCameraConfig")
    out = {}
    for name, c in cams.items():
        if has_placeholder(c.get("index_or_path")):
            raise SystemExit(
                f"Camera '{name}' still has a placeholder index_or_path. Run "
                f"`make detect-cameras` and fill configs/smolvla_so101.yaml first (§10.5)."
            )
        out[name] = OpenCVCameraConfig(
            index_or_path=c["index_or_path"],
            fps=c.get("fps", 30),
            width=c.get("width", 640),
            height=c.get("height", 480),
        )
    return out


def _resolve_device(name: str):
    import torch
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# The policy was fine-tuned with LeRobot's `--rename_map` mapping the SO-101 cameras
# onto SmolVLA base's expected slots (overhead->camera1, wrist->camera2), so its
# preprocessor/input_features are keyed on camera1/camera2. Inference MUST apply the
# same rename or the policy receives no image for its declared slots. Keep this in
# sync with the rename_map in scripts/runpod_smolvla.sh.
_CAMERA_RENAME = {
    "observation.images.overhead": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


def _prep_value(key: str, v, device):
    """Convert one observation entry to the tensor SmolVLA.select_action expects.

    Mirrors LeRobot's `predict_action`: images become CHW float32 in [0,1], every
    entry gets a leading batch dim, then moves to device. The image branch is guarded
    so it's idempotent — if the observation already arrives float/CHW (version-
    dependent) it isn't double-normalized or wrongly permuted. Getting this wrong
    (feeding HWC uint8 straight in) yields subtly-wrong actions, not a crash.
    """
    import torch
    t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
    if "image" in key:
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
            if float(t.max()) > 1.5:  # float image still in 0..255
                t = t / 255.0
        # HWC -> CHW when the channel axis is trailing (3/1) and not already leading
        if t.ndim == 3 and t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1).contiguous()
    else:
        t = t.float()
    if t.ndim and t.shape[0] != 1:  # add batch dim if not already batched
        t = t.unsqueeze(0)
    return t.to(device)


def _observation_to_batch(obs: dict, instruction: str, device):
    """Shape a robot observation dict into the batch SmolVLA.select_action expects.

    VERIFY: SmolVLA conditions on language via a "task" key (list[str]); image keys
    are "observation.images.<name>" and state is "observation.state". robot.get_observation()
    returns overhead/wrist keys, which we rename to the camera1/camera2 the policy was
    trained on before moving to device.
    """
    batch = {}
    for k, v in obs.items():
        key = _CAMERA_RENAME.get(k, k)
        batch[key] = _prep_value(key, v, device)
    batch["task"] = [instruction]
    return batch


class SO101Runner:
    """Holds the connected robot + loaded policy so one process can serve many commands."""

    def __init__(self, config_path=DEFAULT_CONFIG):
        self.cfg = load_config(config_path)
        rt = self.cfg["runtime"]
        self.control_hz = float(rt.get("control_hz", 30))
        self.max_steps = int(rt.get("max_episode_steps", 900))
        self.device = _resolve_device(rt.get("device", "auto"))
        print(f"[run_policy] device={self.device} control_hz={self.control_hz}")

        self.policy = _load_smolvla(rt["policy_path"], self.device)
        _apply_action_horizon(self.policy, rt)
        self.robot = _build_robot(self.cfg["robot"])
        print("[run_policy] robot connected, policy loaded.")

    def run_instruction(self, instruction: str, max_steps: int | None = None) -> None:
        """Run one conditioned rollout at the target control rate until timeout/stop."""
        import torch

        steps = max_steps or self.max_steps
        period = 1.0 / self.control_hz
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        print(f"[run_policy] running: {instruction!r}  ({steps} steps @ {self.control_hz} Hz)")

        try:
            for i in range(steps):
                t0 = time.perf_counter()
                obs = self.robot.get_observation()       # VERIFY: method name
                batch = _observation_to_batch(obs, instruction, self.device)
                with torch.no_grad():
                    action = self.policy.select_action(batch)  # returns next action
                if hasattr(action, "squeeze"):
                    action = action.squeeze(0).to("cpu")  # drop batch dim for the robot
                # VERIFY: send_action expects a dict/tensor matching robot.action_features
                self.robot.send_action(action)
                # maintain the control period
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
                elif i and i % 30 == 0:
                    print(f"   step {i}: loop {dt*1000:.1f} ms > {period*1000:.1f} ms budget "
                          f"(inference too slow for {self.control_hz} Hz — see Phase 5)")
        except KeyboardInterrupt:
            print("\n[run_policy] interrupted — stopping rollout.")
        print("[run_policy] rollout done.")

    def close(self) -> None:
        try:
            self.robot.disconnect()
        except Exception:
            pass


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Run fine-tuned SmolVLA on the SO-101.")
    ap.add_argument("--instruction", required=True, help='e.g. "pick up the red block"')
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)

    runner = SO101Runner(args.config)
    try:
        runner.run_instruction(args.instruction, max_steps=args.max_steps)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
