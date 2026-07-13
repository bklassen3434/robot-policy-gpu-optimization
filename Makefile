.PHONY: help install install-gpu install-realbot test overfit train eval bench parity profile clean \
	detect-cameras teleop record train-smolvla run-typed run-voice

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
	@echo "  run-typed       - Phase 3: run the arm from TYPED commands"
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
install-realbot:
	pip install -e ".[realbot]"

detect-cameras:
	python -m robopolicy.realbot.detect_cameras

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
	python -m robopolicy.realbot.agent --typed --config configs/smolvla_so101.yaml

run-voice:
	python -m robopolicy.realbot.agent --config configs/smolvla_so101.yaml

clean:
	rm -rf kernels/build outputs/*.tmp __pycache__ .pytest_cache
