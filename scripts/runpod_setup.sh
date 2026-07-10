#!/usr/bin/env bash
# One-shot setup for a fresh RunPod A100/H100 pod (Ubuntu + CUDA + PyTorch image).
# Usage:  bash scripts/runpod_setup.sh
set -euo pipefail

echo ">>> GPU + CUDA check"
nvidia-smi
nvcc --version || echo "WARN: nvcc not found — pick a RunPod template WITH the CUDA toolkit (needed to build the kernel)."

echo ">>> System build deps"
apt-get update -y && apt-get install -y --no-install-recommends ninja-build git

echo ">>> Python deps"
python -m pip install --upgrade pip
# torch is expected to be present in the RunPod CUDA image; if not, install the
# matching CUDA build from https://pytorch.org before this line.
pip install -e ".[data,dev]"
pip install -r requirements-gpu.txt

echo ">>> Sanity: torch sees CUDA"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available to torch"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0))
PY

echo ">>> Download dataset"
python scripts/download_dataset.py

cat <<'MSG'

Setup done. Typical flow:
  make train      # train ACT to convergence
  make eval       # val L1 + sim success rate
  make profile    # NSight -> find hottest op
  make parity     # custom kernel == reference
  make bench      # latency before/after

Remember to STOP the pod when idle. Checkpoints land in outputs/ — put that on a
persistent volume so you can shut down between sessions.
MSG
