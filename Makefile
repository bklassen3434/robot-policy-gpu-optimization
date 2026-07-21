.PHONY: help install install-gpu install-realbot test overfit train eval bench parity profile clean \
	detect-cameras teleop record train-smolvla run-typed run-voice run-rollout

help:
	@echo "Sim ACT + CUDA kernel:"
	@echo "  install      - local (Mac/CPU/MPS) dev install"
	@echo "  install-gpu  - install on the rented CUDA box (adds ninja for kernel builds)"
	@echo "  test         - run unit tests (model shapes + overfit)"
	@echo "  overfit      - overfit the model on one batch (sanity check, CPU/MPS ok)"
	@echo "  train        - full training run (config-driven)"
	@echo "  eval         - validation L1 + sim rollout success rate"
	@echo "  parity       - custom CUDA kernel vs PyTorch reference (needs CUDA)"
	@echo "  bench        - latency before/after the kernel (needs CUDA)"
	@echo "  profile      - run NSight Systems on one forward+backward (needs CUDA)"
	@echo ""
	@echo "Path B (physical SO-101 + SmolVLA):"
	@echo "  install-realbot - Mac-side install: SmolVLA inference + voice (.[realbot])"
	@echo "  detect-cameras  - discover camera ids + serial ports for the config (§10.5)"
	@echo "  teleop          - teleoperate the SO-101 (LeRobot CLI; see record_runbook.md)"
	@echo "  record          - collect the object-selection dataset (LeRobot CLI; runbook)"
	@echo "  train-smolvla   - fine-tune SmolVLA on RunPod (bash scripts/runpod_smolvla.sh)"
	@echo "  run-rollout     - Phase 3: run the arm via lerobot-rollout (OBJECT=\"pick up the keys\")"
	@echo "  run-typed       - Phase 3 (legacy loop): TYPED commands — see run-rollout instead"
	@echo "  run-voice       - Phase 4: run the arm from SPOKEN commands"

install:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[data,dev]"
	pip install -r requirements-gpu.txt

test:
	pytest tests -q

overfit:
	python -m robopolicy.train --config configs/act_aloha.yaml --overfit-one-batch

train:
	python -m robopolicy.train --config configs/act_aloha.yaml

resume:
	python -m robopolicy.train --config configs/act_aloha.yaml --resume

eval:
	python -m robopolicy.eval --config configs/act_aloha.yaml --checkpoint outputs/last.pt

parity:
	pytest kernels/test_parity.py -q

bench:
	python kernels/bench.py

profile:
	bash profiling/run_nsight.sh

# ── Path B: physical SO-101 + SmolVLA ────────────────────────────────────────
# The realbot targets (detect-cameras/run-typed/run-voice) need an interpreter that
# has BOTH LeRobot+smolvla AND this repo (robopolicy). That lives in the ~/lerobot uv
# venv — lerobot is pinned there at 0.5.2, so we do NOT `pip install` this repo into it
# (which would drag lerobot>=0.6 via the realbot extra). Instead PYTHONPATH=src makes
# robopolicy importable in-place. Override REALBOT_PY to point at a different venv.
# NOTE: run these from a GUI terminal (Terminal.app/iTerm) so macOS grants the camera
# permission — a headless/agent-spawned shell can't get it.
REALBOT_PY ?= $(HOME)/lerobot/.venv/bin/python
REALBOT    = PYTHONPATH=src $(REALBOT_PY)
# lerobot ships `lerobot-rollout` — the OFFICIAL engine for running a trained policy on
# a real robot (obs->preprocess->policy->postprocess->send_action, with the rename_map +
# normalization loaded from the checkpoint). run-rollout below drives it; prefer it over
# the hand-rolled run-typed loop, which assumed a different lerobot observation format.
ROLLOUT    = $(dir $(REALBOT_PY))lerobot-rollout
# Object to fetch + how long each rollout runs. Override per attempt:
#   make run-rollout OBJECT="pick up the keys"
OBJECT   ?= pick up the sanitizer
DURATION ?= 30
# VIEW=true opens the Rerun camera viewer — needs the `rerun` VIEWER binary on PATH
# (pip's rerun-sdk ships only the Python lib). Default off so the rollout just runs.
VIEW     ?= false
# SmolVLA inference on MPS is ~0.5-0.6s/call, far slower than 30Hz. Driving at 30Hz
# makes the arm burst through a chunk then stall ("spaz then dormant"). Lower FPS to
# match inference speed for smooth motion. INFER=sync (one call/tick) or rtc (chunking).
FPS      ?= 30
INFER    ?= rtc

install-realbot:
	pip install -e ".[realbot]"

detect-cameras:
	$(REALBOT) -m robopolicy.realbot.detect_cameras

# teleop/record are thin pointers to LeRobot's own CLI: the exact flags depend on
# your filled-in ports/cameras and your installed LeRobot version, so the runbook
# is the source of truth (verify with `lerobot-record --help`).
teleop:
	@echo "Teleoperate the SO-101 with LeRobot's CLI. Fill ports in configs/smolvla_so101.yaml,"
	@echo "then see the template command in src/robopolicy/realbot/record_runbook.md (§0)."
	@echo "Verify flags:  lerobot-teleoperate --help"

record:
	@echo "Collect the object-selection dataset with LeRobot's record CLI."
	@echo "Follow src/robopolicy/realbot/record_runbook.md exactly (balance + randomize + distractors)."
	@echo "Verify flags:  lerobot-record --help"

train-smolvla:
	bash scripts/runpod_smolvla.sh

run-typed:
	$(REALBOT) -m robopolicy.realbot.agent --typed --config configs/smolvla_so101.yaml

run-voice:
	$(REALBOT) -m robopolicy.realbot.agent --config configs/smolvla_so101.yaml

# Phase 3 eval via lerobot's own rollout engine. One object per run (re-run for each).
# RTC inference is used because SmolVLA is too slow for one-call-per-tick at 30 Hz.
# rename_map (overhead->camera1, wrist->camera2) MUST match training or the policy gets
# no images. Cameras/port mirror the validated record command (overhead=0, wrist=1).
run-rollout:
	@echo ">>> rollout: '$(OBJECT)'  (${DURATION}s)  — all 3 objects on the table; Ctrl-C to stop early"
	$(ROLLOUT) \
	  --strategy.type=base \
	  --policy.path=bklassen3434/smolvla_so101 \
	  --policy.device=mps \
	  --robot.type=so101_follower \
	  --robot.port=/dev/tty.wchusbserial5B3E1213311 \
	  --robot.id=my_awesome_follower_arm \
	  --robot.cameras='{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30} }' \
	  --rename_map='{"observation.images.overhead": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
	  --inference.type=$(INFER) \
	  --fps=$(FPS) \
	  --task="$(OBJECT)" \
	  --duration=$(DURATION) \
	  --display_data=$(VIEW)

clean:
	rm -rf kernels/build outputs/*.tmp __pycache__ .pytest_cache
