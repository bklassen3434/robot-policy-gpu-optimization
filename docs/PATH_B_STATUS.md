# Path B — Current Status & Next Steps (handoff)

Living status doc for the physical SO-101 + SmolVLA work. Read this first, then
`docs/PATH_B_PLAN.md` for the full plan. **Last updated: 2026-07-14.**

## TL;DR — where we are
- Goal: fine-tune **SmolVLA** so a physical **SO-101** picks the object you *name*
  ("pick up the pen") out of distractors, driven by voice. First = language-conditioned
  object selection. See `docs/PATH_B_PLAN.md`.
- **Phase 1 (data collection) is in progress and going well.** Dataset:
  **`bklassen3434/so101_pick_object_20260713_221513`** (public HF Hub).
  - **60 episodes recorded, local + Hub in sync**: **30 "pick up the pen" + 30 "pick up the keys"** (balanced).
  - **Next: record 30 "pick up the sanitizer" episodes** to complete a 3-object set (~90 total).
- After the sanitizer batch → **Phase 2: SmolVLA fine-tune on a rented RunPod GPU**, then
  Phase 3 typed eval, Phase 4 voice.

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

## Next steps
1. **Record 30 "pick up the sanitizer" episodes** (command above). Target ~30 for balance → ~90 total.
   - Recommend `num_episodes=5` batches given the flaky comms (less to lose per dropout, more frequent
     auto-push). Re-run the same command until sanitizer ≈ 30.
2. **Phase 2 — SmolVLA fine-tune on RunPod** (`scripts/runpod_smolvla.sh`; user chose RunPod, not the
   Modal `~/lerobot/train.py` used for their earlier ACT/pen-lift work). Fine-tune `lerobot/smolvla_base`
   on `bklassen3434/so101_pick_object_20260713_221513`; push checkpoint to `bklassen3434/smolvla_so101`.
   Verify `lerobot-train` flags against the installed version.
3. **Phase 3 — typed eval** on the arm (`python -m robopolicy.realbot.agent --typed`), then
   **Phase 4 — voice** (`--display_data` off; `agent.py` uses MLX-Whisper). See the realbot package.

## Key files (this repo)
- `docs/PATH_B_PLAN.md` — full plan (phases, data budget, rules).
- `configs/smolvla_so101.yaml` — objects, canonical phrasing, robot/camera config, train hparams.
  NOTE: its `dataset.repo_id` is the un-timestamped base; the ACTUAL recorded dataset is
  `..._20260713_221513` (LeRobot auto-appends a timestamp at creation).
- `src/robopolicy/realbot/{voice,run_policy,agent,detect_cameras}.py`, `record_runbook.md`.
- `scripts/runpod_smolvla.sh` — RunPod fine-tune setup.
