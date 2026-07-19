# Path B — Current Status & Next Steps (handoff)

Living status doc for the physical SO-101 + SmolVLA work. Read this first, then
`docs/PATH_B_PLAN.md` for the full plan. **Last updated: 2026-07-19.**

## TL;DR — where we are
- Goal: fine-tune **SmolVLA** so a physical **SO-101** picks the object you *name*
  ("pick up the pen") out of distractors, driven by voice. First = language-conditioned
  object selection. See `docs/PATH_B_PLAN.md`.
- **Phase 1 (data collection) is DONE.** Dataset **`bklassen3434/so101_pick_object_20260713_221513`**
  (public HF Hub): **90 episodes, balanced 30/30/30** (pen / keys / sanitizer), 41,070 frames.
- **Phase 2 (SmolVLA fine-tune on RunPod) is essentially done.** Fine-tuned model on the Hub at
  **`bklassen3434/smolvla_so101`** (public). Trained to 10k/20k steps, resumed to 20k (loss ≈0.04).
  See "Phase 2 — the RunPod recipe that actually worked" below; `scripts/runpod_smolvla.sh` now
  encodes it (fresh-run + resume).
- **Next: Phase 3 typed eval on the arm** (`make run-typed`), then Phase 4 voice.

## Two repos / two machines (important)
1. **This repo** (`robot-policy-gpu-optimization`, workspace `basseterre`, branch
   `execute-path-b-plan`, PR #17) — the *new code*: `src/robopolicy/realbot/`
   (voice, run_policy, agent, detect_cameras, record_runbook), `configs/smolvla_so101.yaml`,
   `scripts/runpod_smolvla.sh`, Makefile targets. This is the deploy/voice layer + configs.
2. **`~/lerobot`** (user's fork clone, uv project) — the *installed LeRobot* used for the
   actual `lerobot-record` / `lerobot-train` CLIs on the Mac. Recording happens here, NOT
   from this repo. Has local patches (see "LeRobot patches" below).

## Hardware (SO-101, all connected to the M3 Mac)
- **Follower arm:** `--robot.type=so101_follower --robot.id=my_awesome_follower_arm
  --robot.port=/dev/tty.wchusbserial5B3E1213311`
- **Leader arm (teleop):** `--teleop.type=so101_leader --teleop.id=my_awesome_leader_arm
  --teleop.port=/dev/tty.wchusbserial5B3E1187321`
- **Cameras (macOS OpenCV integer indices):** overhead = **index 0** (C270, object grounding),
  wrist = **index 1** (icspring, grasp). Index 2 = built-in FaceTime. Re-verify each session
  with `uv run lerobot-find-cameras opencv` (indices can shuffle on replug).
- Arms are **already calibrated** (ids above reuse existing calibration in
  `~/.cache/huggingface/lerobot/calibration` — machine-global, no recalibration needed).
- **Serials are stable** across replugs (WCH USB bridge).

## Environment (recording runs from `~/lerobot`)
```bash
cd ~/lerobot
source .venv/bin/activate          # uv-created venv, but usable as a plain venv
export HF_HUB_DISABLE_XET=1         # REQUIRED — see "Gotchas"
```
- The `.venv` has extras installed: `feetech` (motors), `dataset`, `core_scripts`
  (pynput for keyboard control + rerun for the camera viewer). Use `--display_data=true`
  to see the camera streams (Rerun viewer).
- HF: logged in as `bklassen3434` (token cached). Dataset repo is **public**.

## The record command (the ONLY knob is `num_episodes`)
Everything below is fixed every run; change only `--dataset.single_task` (per object
batch) and `--dataset.num_episodes` (how many NEW episodes to add this session).
`--resume=true` + `--dataset.root=...` are mandatory to append to the existing dataset.

```bash
lerobot-record \
  --robot.type=so101_follower --robot.id=my_awesome_follower_arm --robot.port=/dev/tty.wchusbserial5B3E1213311 \
  --robot.cameras='{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }' \
  --teleop.type=so101_leader --teleop.id=my_awesome_leader_arm --teleop.port=/dev/tty.wchusbserial5B3E1187321 \
  --display_data=true \
  --dataset.repo_id=bklassen3434/so101_pick_object_20260713_221513 --resume=true \
  --dataset.root=/Users/benklassen/.cache/huggingface/lerobot/bklassen3434/so101_pick_object_20260713_221513 \
  --dataset.fps=30 --dataset.single_task="pick up the sanitizer" \
  --dataset.num_episodes=30 --dataset.episode_time_s=30 --dataset.reset_time_s=15 \
  --dataset.push_to_hub=true
```
- **Object set / canonical tasks:** `pick up the pen` | `pick up the keys` | `pick up the sanitizer`
  (object token is `sanitizer`, not "hand sanitizer"). All 3 objects on the table EVERY
  episode as distractors; only the *target* changes per batch. Randomize positions +
  decouple identity from location (§5 of the plan) to avoid shortcut learning.
- **Keys during recording:** `→` end episode · `←` re-record last episode · `Esc` stop.
- **Batch by target** — `single_task` is stamped on every episode in a run, so record one
  object per run.
- Check progress anytime:
  ```bash
  HF_HUB_OFFLINE=1 python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D;import collections;ds=D('bklassen3434/so101_pick_object_20260713_221513',root='/Users/benklassen/.cache/huggingface/lerobot/bklassen3434/so101_pick_object_20260713_221513');print(collections.Counter(str(ds.meta.episodes[i]['tasks']) for i in range(ds.num_episodes)))"
  ```

## Gotchas we already hit (and the fixes)
- **HF Xet backend fails commits** (`BadRequestError: ... hook: lfs-verify` / `Xet ... Unauthorized`).
  Fix: **`export HF_HUB_DISABLE_XET=1`** before any push (including record auto-push).
- **Empty-episode crash on stop** (`ValueError: must add frames before add_episode`) — pressing
  `Esc` right after a reset saves a 0-frame episode. Avoid by setting `num_episodes` to the
  session target and letting it finish, or pressing `Esc` mid-episode. Already-saved episodes
  are never lost; the crash just skips the auto-push (then push manually — see below).
- **Flaky motor-bus comms** — intermittent dropouts (`no status packet`, `Torque_Enable id_=3`,
  or the port vanishing). Root cause: the arm cable gets tugged when over-extended and/or power
  brownout at high torque. **Fix: tape the USB + power cables down (strain relief), reseat
  connectors, USB direct to Mac (no hub), keep teleop within a comfortable reach.** Each dropout
  loses only the in-progress episode; just re-run the command to continue.
- **Manual backup push** (after a crash skipped the auto-push):
  ```bash
  cd ~/lerobot && HF_HUB_DISABLE_XET=1 python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; D('bklassen3434/so101_pick_object_20260713_221513', root='/Users/benklassen/.cache/huggingface/lerobot/bklassen3434/so101_pick_object_20260713_221513').push_to_hub()"
  ```

## LeRobot patches applied to `~/lerobot` (not upstream)
- `src/lerobot/scripts/lerobot_record.py` — guard so an empty final episode doesn't crash the run
  (empty-episode fix).
- `src/lerobot/datasets/dataset_tools.py` (`_copy_and_reindex_videos`) — extract exactly `length`
  frames per episode instead of trusting `to_timestamp`, so `delete_episodes` works on datasets
  with a few phantom video-tail frames from comms glitches. Used to delete a garbage episode (the
  old ep39, an accidental skip) cleanly. **If `~/lerobot` is reinstalled/reset, these revert.**

## Phase 2 — the RunPod recipe that actually worked (2026-07-19)
Ran on image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8.0+cu128, Ubuntu 24.04),
A100 80GB, CA-MTL-3. `scripts/runpod_smolvla.sh` now encodes all of this (fresh-run **and** resume).
Every one of these was a wall we hit — do not regress them:
- **PEP 668:** Ubuntu 24.04 python is externally-managed → `pip install --break-system-packages`.
- **Two extras:** `pip install -e ".[smolvla,dataset]"` — `lerobot-train` needs HF `datasets` (the
  `dataset` extra), which `smolvla` does NOT pull in.
- **Pin torch:** constrain torch/torchvision/torchaudio to the image's `+cu128` builds so the install
  can't swap in a CPU/other-CUDA wheel.
- **Video backend = pyav, NOT torchcodec.** PyPI torchcodec is built for a newer torch
  (`undefined symbol: torch_dtype_float4_e2m1fn_x2` vs torch 2.8) → pass `--dataset.video_backend=pyav`.
  PyAV links the system FFmpeg 6 fine.
- **Load base via `--policy.path=lerobot/smolvla_base`** (this build intercepts it and infers the type).
  The old `--policy.type=smolvla --policy.pretrained_path=...` combo is WRONG here.
- **Camera rename:** SmolVLA base declares `camera1/2/3`; our data has `overhead/wrist`. Pass
  `--rename_map` overhead→camera1, wrist→camera2 (dataset ⊆ policy passes the validator).
  **Inference must apply the same map** — done in `src/robopolicy/realbot/run_policy.py`.
- **Dataset videos are AV1** → pyav decode is CPU+RAM heavy and the container was mem-capped (~109GB).
  `--num_workers=48` OOM'd (`av.error.MemoryError`); **20 is the stable sweet spot** (~2.3 s/step on A100).
- **Push:** `export HF_HUB_DISABLE_XET=1` (Xet lfs-verify fails). `push_to_hub` fires only at the END.

### Recovery / resume (learned the hard way)
- The pod was **terminated twice** — once by RunPod when the account **balance hit $0** (auto-terminate),
  which wipes the container disk. **Fix: mount a network volume at `/workspace`** (ours: `lqsnzyd0x6`,
  50GB, CA-MTL-3) so `outputs/`, the HF cache, and the LeRobot checkout survive. Checkpoints save every
  5k steps to the volume.
- **Resume:** re-run `scripts/runpod_smolvla.sh` on a fresh pod with the same volume mounted at
  `/workspace`; it auto-detects `outputs/.../checkpoints/last` and runs
  `lerobot-train --config_path=<last>/pretrained_model/train_config.json --resume=true`
  (train_config.json carries dataset, rename_map, video_backend, steps, workers, push_to_hub).
- **Deploy via API** (Cloudflare blocks non-curl UAs → use `curl`): `podFindAndDeployOnDemand` with
  `networkVolumeId` + `volumeMountPath:/workspace` + `dataCenterId` matching the volume, and inject the
  SSH pubkey via `env:[{key:PUBLIC_KEY,value:<pubkey>}]`. Community A100 was out of stock in CA-MTL-3;
  SECURE ($1.39/hr) had capacity.
- **On-pod watchers** (own tmux, survive the driving session dying): a checkpoint-sync that pushes each
  new checkpoint to the Hub as a safety net, and an autostop that `runpodctl stop`s the pod when the
  `train` session ends (needs `runpodctl config --apiKey`).

## Next steps
1. **Phase 3 — typed eval on the arm.** `make run-typed` (= `python -m robopolicy.realbot.agent --typed
   --config configs/smolvla_so101.yaml`). Pulls `bklassen3434/smolvla_so101` (public) for inference;
   config already has real ports + cameras (overhead=0, wrist=1). All 3 objects on the table; type e.g.
   `pick up the sanitizer`. `run_policy.py` now renames cameras to camera1/camera2 to match training.
   Run from an env with BOTH this repo (`pip install -e .`) and LeRobot+smolvla. If MPS errors, set
   `runtime.device: cpu` in the config.
2. **Phase 4 — voice** (`make run-voice`; `agent.py` uses MLX-Whisper). See the realbot package.
- If eval is weak: resume for more steps (volume + `training_state` preserved) or collect more/better data.

## Key files (this repo)
- `docs/PATH_B_PLAN.md` — full plan (phases, data budget, rules).
- `configs/smolvla_so101.yaml` — objects, canonical phrasing, robot/camera config, train hparams.
  NOTE: its `dataset.repo_id` is the un-timestamped base; the ACTUAL recorded dataset is
  `..._20260713_221513` (LeRobot auto-appends a timestamp at creation).
- `src/robopolicy/realbot/{voice,run_policy,agent,detect_cameras}.py`, `record_runbook.md`.
- `scripts/runpod_smolvla.sh` — RunPod fine-tune setup.
