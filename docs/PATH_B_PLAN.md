# Path B — Voice- & Vision-Commanded SO-101 via SmolVLA

Concrete, repo-specific plan to extend this repo from a simulated from-scratch ACT
policy to a **physical SO-101 arm that picks the object you name out loud.**

> **For the assistant picking this up:** you have this repo + this plan. Don't start
> coding until you've confirmed §10 (open decisions) with the user. Then work the
> phases in order and **verify each before the next**. Get the *entire* loop working
> on a small dataset before scaling data — that's the biggest time-saver. Where this
> plan names a LeRobot CLI/flag, **verify it against the installed LeRobot version**
> (entry points change across releases); the *concepts* are stable, the exact command
> strings may not be.

## 1. Goal (locked decisions)

Make the user's **physical SO-101** (assembled + calibrated, **wrist + overhead
cameras**) accept **spoken commands** and act on what it sees. Core = a **fine-tuned
SmolVLA** (`lerobot/smolvla_base`, ~450M language-conditioned VLA) + a **Whisper**
speech-to-text front-end on the Mac.

- **First task: language-conditioned object selection** — "pick up the red block"
  with other objects present as distractors. (More tasks added later.)
- **Extend THIS repo** (add a real-robot subpackage + configs + a voice layer).
- **Compute split:** data collection, Whisper, and SmolVLA *inference* run on the
  **M3 Mac**; SmolVLA *fine-tuning* on a rented **RunPod GPU** (reuse the existing
  `scripts/runpod_setup.sh` workflow).

## 2. Reality checks (internalize)

1. **The existing sim ACT does NOT transfer** — different robot (bimanual ALOHA) +
   sim. You must collect **new real SO-101 demos**. No shortcut.
2. **Object selection is a *grounding* task, not a picking task.** SmolVLA already
   knows how to pick; you're teaching it to pick *the object the words name, with
   distractors present*. The dataset must be structured to force that (see §5) or the
   model will cheat (e.g. learn "grab the thing on the left").
3. **Data collection is the dominant cost**, not modeling.
4. SmolVLA is **pretrained** → fine-tuning needs *far* less data than from-scratch.

## 3. System architecture

```
 mic ──► Whisper (MLX, on Mac) ──► instruction string ("pick up the red block")
                                        │
 wrist cam + overhead cam ─────────────►├──► SmolVLA (fine-tuned) ──► joint actions ──► SO-101
                                        │        (image(s) + language → action chunk)
                                        └──► run until done / timeout, then await next command
```
Vision is inherent (SmolVLA conditions on the camera frames). The new capability is
**language conditioning**; voice is a thin front-end that produces the instruction.

## 4. What to add to this repo

Keep the existing sim-ACT + kernel work untouched; add a parallel real-robot track:

```
src/robopolicy/realbot/           NEW — physical-arm track
  __init__.py
  voice.py            Whisper (MLX-Whisper / whisper.cpp) mic-capture → text
  run_policy.py       load fine-tuned SmolVLA, run on SO-101 from an instruction string
  agent.py            glue: voice → instruction → run_policy rollout loop
  record_runbook.md   how to collect the object-selection dataset (commands + checklist)
configs/
  smolvla_so101.yaml  dataset repo_id, task list, robot + 2-camera config, train hparams
scripts/
  runpod_smolvla.sh   fine-tuning setup on RunPod (mirror runpod_setup.sh; add smolvla deps)
docs/
  PATH_B_PLAN.md      this file
```
Add `Makefile` targets: `teleop`, `record`, `train-smolvla`, `run-voice`.

Heavy lifting (teleop / record / train) is **LeRobot's own CLI**; the genuinely new,
repo-specific code is the **voice layer + the run/agent glue + configs + runbook**.

## 5. Phase 1 — Data collection (the real work)

**Object set:** start with **2–3 visually distinct objects** (e.g. red cube, blue
cube, yellow duck). **Every episode has all objects present** (target + distractors).

**Per episode:**
1. Randomize object positions in the workspace.
2. Choose ONE target (rotate so targets are **balanced** — equal count each).
3. Instruction string = `"pick up the {object}"` (match how you'll speak it).
4. Teleoperate the follower to grasp the target and lift it clear. Success = lifted.
5. Record wrist + overhead frames + joint states + actions at ~30 fps.

**The rules that make or break it (§2.2):**
- **Balance** targets equally.
- **Randomize position AND which object is the target** every episode — so it can't
  shortcut via location.
- **Distractors always present.**
- **Consistent teleop** (smooth, similar grasp strategy).
- Overhead cam carries object grounding; wrist cam carries the grasp.

**Collect with** LeRobot's record flow (set the per-episode `task` string; the dataset
`task` field is what conditions SmolVLA). Push the dataset to the HF Hub.

**Verify:** dataset loads, episodes replay correctly, task strings attached, both
camera streams present.

## 6. Data budget (the numbers)

SmolVLA is pretrained, so **~50 demos per (object-as-target) condition** is the anchor.

| Stage | Data | Purpose | ~teleop time |
|---|---|---|---|
| **Validate the loop** | 2 objects × ~30 (~60) | Prove record→train→run→voice end-to-end. Not meant to be good. | ~1 hr |
| **Working MVP** | 2–3 objects × ~50 (~100–150), balanced + randomized + distractors | Reliably picks the named object in trained-like layouts | ~2–3 hrs |
| **Robust** | 3–5 objects × ~75–100 (~300–500), varied lighting/background/arrangement | Generalizes to new positions, some new phrasings, more objects | ~1 day (spread) |

**Quality/diversity/balance > raw count.** A tight balanced 120-episode set beats a
sloppy 300-episode one. **Collect the ~60 "validate" set FIRST** and run the whole
pipeline before investing in the MVP set.

## 7. Phase 2 — Fine-tune SmolVLA (RunPod GPU)

1. Rent a GPU pod (one A100/4090 is plenty). Reuse the `scripts/runpod_setup.sh`
   pattern → `scripts/runpod_smolvla.sh`.
2. Install LeRobot with the **SmolVLA extra** (pulls the VLM/transformers deps) —
   verify the exact extra name for the installed version.
3. Pull the dataset from the Hub.
4. Fine-tune from the base checkpoint — conceptually:
   `lerobot-train policy.type=smolvla policy.path=lerobot/smolvla_base
   dataset.repo_id=<your-dataset> ...` (**verify flags**). ~20k steps is a common
   starting point; a few hours on one GPU.
5. Push the fine-tuned checkpoint to the Hub (so the Mac can pull it for inference).

**Verify:** loss converges; checkpoint loads on the Mac.

## 8. Phase 3 — Real-robot eval (typed commands first)

Load the fine-tuned policy on the Mac (MPS), run it on the arm conditioned on a
**typed** instruction (before wiring voice). Measure success rate over N trials per
object, on **held-out positions/arrangements**. If low, **fix the data** (more
balance/diversity) before more training — data quality dominates here.

**Verify:** arm reliably picks the correct named object among distractors.

## 9. Phase 4 — Voice front-end, then Phase 5 — latency

- **Phase 4 (voice):** `realbot/voice.py` — capture mic audio, transcribe with
  **MLX-Whisper** or **whisper.cpp (Metal)** locally on the M3 (a small/base model is
  plenty for short commands). `agent.py`: utterance → transcription → pass as the
  policy instruction → run one conditioned rollout → await next command.
  **Verify:** spoken "pick up the {X}" → correct action, end to end.
- **Phase 5 (optional, plays to the user's strengths):** real control runs a tight
  loop (~30–50 Hz). If SmolVLA inference on the Mac is too slow, profile it and
  optimize (CoreML/MPS conversion, reduced precision, cache/batch the vision encoder).
  This is exactly the inference-latency work the user already did for the ACT kernel.

## 10. Open decisions (confirm with the user before coding)

1. Exact **starting object set** (2–3 distinct objects).
2. **Instruction phrasing** vocabulary (e.g. "pick up the red block" / "grab the duck").
3. Target **control rate** (Hz) for the run loop.
4. **HF Hub** dataset/checkpoint visibility (public vs private).
5. Camera device pinning — **pin cameras by stable path/serial**, not index (USB
   enumeration order can change between sessions).

## 11. Definition of done

Spoken `"pick up the {object}"` → Whisper transcription → fine-tuned SmolVLA (given the
wrist+overhead frames + instruction) → **SO-101 picks the correct object among
distractors**, at a measured success rate on held-out arrangements, for the agreed
object set. Extends to more tasks by adding language-labeled demos.

## 12. Gotchas (from this project's experience)

- **LeRobot version:** verify CLI entry points (`lerobot-record`/`-train`/`-teleoperate`
  vs `python -m ...`) and the dataset-format version against what's installed. Install
  the SmolVLA extra for training.
- **Shortcut learning** is the main failure mode for object selection — balance +
  position randomization + distractors are non-negotiable (§5).
- **Consistent teleoperation** quality matters more than a few extra episodes.
- **Camera indices** can shuffle across USB re-plugs — pin by path/serial and name
  them (`wrist`, `overhead`) in the robot config.
- **Inference device:** SmolVLA runs on MPS on the Mac; confirm `select_action`
  latency meets your control rate, else Phase 5.
