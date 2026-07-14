# SO-101 object-selection — data collection runbook

How to collect the language-conditioned pick dataset for Path B Phase 1. Read
§5–§6 of `docs/PATH_B_PLAN.md` for the *why*; this is the *how* (commands +
checklist). The rules here are what make the dataset teach grounding instead of a
shortcut — follow them exactly.

> **Verify the CLI first.** LeRobot entry points move across releases
> (`lerobot-record` vs `python -m lerobot.record`, flag names, dataset version).
> Run `lerobot-record --help` on your installed version and reconcile before a big
> session. The *concepts* below are stable; the exact flag strings may not be.

## 0. One-time setup

1. Assemble + calibrate the SO-101 leader and follower (LeRobot's calibration flow).
2. Mount the **overhead** and **wrist** cameras. Overhead sees the whole workspace
   (object grounding); wrist rides the gripper (grasp).
3. Discover stable device ids and fill the config:
   ```bash
   make detect-cameras          # prints serial ports + camera unique ids/indices
   # paste robot.port, teleop.port, and each cameras.<name>.index_or_path into
   # configs/smolvla_so101.yaml   (pin by stable id/path, NOT a bare index — §10.5)
   ```
4. `huggingface-cli login` (the dataset pushes to the **public** Hub repo
   `bklassen3434/so101_pick_object`).

## 1. Objects & layout

- **Objects (all present every episode):** pen, keys, sanitizer (small hand-sanitizer
  bottle). Visually distinct on purpose — makes the grounding easier to learn.
- One is the **target**; the other two are **distractors**.
- Workspace within the follower's reach and both cameras' view.
- Grasp gotchas: place the **pen** so it can't roll; give the **keys** a graspable
  part (not a flat pile); the **sanitizer** is the easiest grab. Keep grasps consistent.

## 2. The non-negotiable rules (§5)

- **Balance targets equally** — same number of episodes per target object.
- **Randomize BOTH** object positions **and** which object is the target, every
  episode — so the model can't shortcut on location ("always grab left").
- **Distractors always present** (never a single object alone).
- **Consistent teleop** — smooth, similar grasp strategy each time.
- **Success = target lifted clear** of the surface. If you fumble, discard/redo it.
- The per-episode **`task` string is the label** — it must read exactly
  `pick up the {object}` where {object} ∈ {pen, keys, sanitizer} (this is what
  conditions SmolVLA). Use `sanitizer` (not "hand sanitizer") so it matches the
  canonical form `voice.py` produces. `voice.py` normalizes "grab"→"pick up".

## 3. Per-episode loop

1. Randomize the three objects' positions.
2. Pick the next target so counts stay balanced (rotate pen → keys → sanitizer → …).
3. Set the episode task string to `pick up the {target}`.
4. Teleoperate the follower with the leader: approach, grasp the **target**, lift clear.
5. Stop the episode. Reset the scene. Repeat.

## 4. Record command (verify flags)

Conceptually (reconcile flag names with `lerobot-record --help`):

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.wchusbserial5B3E1213311 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.wchusbserial5B3E1187321 \
  --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=bklassen3434/so101_pick_object \
  --dataset.fps=30 \
  --dataset.single_task="pick up the pen" \
  --dataset.num_episodes=<N for this target> \
  --dataset.push_to_hub=true
```

Ports + calibration ids reuse the existing SO-101 calibration (no recalibrate).
Set the camera `index_or_path` values from `make detect-cameras` (macOS OpenCV uses
integer indices — cover a lens to tell the C270 from the icspring).

Record in **per-target batches** (all "pen" episodes with `single_task="pick up the
pen"`, then keys, then sanitizer), or set the task per episode if your version
supports it. Keep the counts balanced across objects.

## 5. Data budget (§6) — collect the *validate* set FIRST

| Stage | Data | Purpose |
|---|---|---|
| Validate the loop | 2 objects × ~30 (~60) | Prove record→train→run→voice end-to-end. |
| Working MVP | 3 objects × ~50 (~150), balanced+randomized+distractors | Reliably picks the named object. |
| Robust | 3–5 objects × ~75–100 (~300–500), varied lighting/background | Generalizes. |

**Do not skip straight to the MVP set.** Collect ~60 episodes, run the *entire*
pipeline (train on RunPod → typed eval on the arm → voice), fix what breaks, *then*
invest in the MVP set. That ordering is the biggest time-saver in the whole plan.

## 6. Verify the dataset (before training)

- Dataset loads and episodes **replay** correctly.
- Each episode has a **task string** attached, reading `pick up the {object}`.
- **Both** camera streams present (`observation.images.overhead` + `.wrist`).
- Target counts are **balanced** across objects.
- Pushed to the Hub and visible at `bklassen3434/so101_pick_object`.

Next: `scripts/runpod_smolvla.sh` on a rented GPU (Phase 2).
