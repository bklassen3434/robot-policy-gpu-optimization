#!/usr/bin/env bash
# One-shot setup + fine-tune of SmolVLA on the SO-101 object-selection dataset,
# on a fresh RunPod GPU pod (one A100 or 4090 is plenty). Mirrors runpod_setup.sh.
# Usage:  bash scripts/runpod_smolvla.sh
#
# Prereqs: dataset already recorded + pushed to the Hub (see realbot/record_runbook.md).
#
# NOTE (§7, §12): LeRobot's train CLI + the SmolVLA extra name move across releases.
# VERIFY `lerobot-train --help` and the extra name on THIS pod's version before a long
# run. The concepts (fine-tune from lerobot/smolvla_base, push to Hub) are stable.
set -euo pipefail

# ── Config values mirrored from configs/smolvla_so101.yaml ────────────────────
DATASET_REPO="bklassen3434/so101_pick_object"
BASE_CKPT="lerobot/smolvla_base"
OUTPUT_REPO="bklassen3434/smolvla_so101"      # fine-tuned checkpoint -> Hub
OUTPUT_DIR="outputs/smolvla_so101"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
SEED="${SEED:-1000}"

echo ">>> GPU + CUDA check"
nvidia-smi

echo ">>> System build deps"
apt-get update -y && apt-get install -y --no-install-recommends git ffmpeg

echo ">>> Python deps (LeRobot + SmolVLA extra)"
python -m pip install --upgrade pip
# torch is expected in the RunPod CUDA image. Installs the SmolVLA extra (VLM/transformers).
# VERIFY the extra name for this lerobot version (may be [smolvla]).
pip install -e ".[smolvla]"

echo ">>> Sanity: torch sees CUDA + SmolVLA policy importable"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available to torch"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0))
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
except ImportError:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
print("SmolVLAPolicy import OK")
PY

echo ">>> HuggingFace auth (needed to pull dataset + push checkpoint)"
# Expects HF_TOKEN in the environment, or run `huggingface-cli login` first.
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo ">>> Fine-tune SmolVLA from ${BASE_CKPT} on ${DATASET_REPO}"
# VERIFY every flag against `lerobot-train --help` on this version.
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path="${BASE_CKPT}" \
  --dataset.repo_id="${DATASET_REPO}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --seed="${SEED}" \
  --output_dir="${OUTPUT_DIR}" \
  --policy.push_to_hub=true \
  --policy.repo_id="${OUTPUT_REPO}"

cat <<MSG

Fine-tune launched/finished. Checkpoint dir: ${OUTPUT_DIR}
Pushed to Hub: ${OUTPUT_REPO}  (the Mac pulls this for inference — configs/smolvla_so101.yaml runtime.policy_path)

VERIFY before leaving the pod:
  * training loss converged (watch the logs / wandb)
  * the checkpoint loads:  SmolVLAPolicy.from_pretrained("${OUTPUT_REPO}")

Remember to STOP the pod when idle. Put ${OUTPUT_DIR} on a persistent volume if you
plan to resume between sessions.
MSG
