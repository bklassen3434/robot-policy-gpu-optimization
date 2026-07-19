#!/usr/bin/env bash
# Setup + fine-tune (or resume) SmolVLA on the SO-101 object-selection dataset on a
# RunPod pod. Recipe VALIDATED 2026-07-19 on the image
#   runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404   (torch 2.8.0+cu128, Ubuntu 24.04)
# on an A100 80GB. See docs/PATH_B_STATUS.md for the full story.
#
# Usage:  HF_TOKEN=hf_xxx bash scripts/runpod_smolvla.sh
#
# Lessons baked in (each of these bit us once — do not "simplify" them away):
#   * Ubuntu 24.04 is PEP-668 externally-managed -> pip needs --break-system-packages.
#   * The dataset extra (HF `datasets`) is REQUIRED by lerobot-train, separate from
#     the smolvla extra -> install ".[smolvla,dataset]".
#   * PyPI torchcodec is built against a newer torch (undefined symbol vs torch 2.8)
#     -> use the pyav video backend (--dataset.video_backend=pyav), not torchcodec.
#   * Load base weights via --policy.path (NOT --policy.type/--policy.pretrained_path);
#     this LeRobot build intercepts --policy.path and infers the type from the ckpt.
#   * SmolVLA base declares camera1/2/3; SO-101 data has overhead/wrist -> --rename_map
#     (inference must apply the same map — see src/robopolicy/realbot/run_policy.py).
#   * Dataset videos are AV1; pyav decode is CPU+RAM heavy and the container is mem-
#     capped (~109GB on our pod). 48 workers OOM'd; 20 is the stable sweet spot.
#   * export HF_HUB_DISABLE_XET=1 before any push (Xet lfs-verify fails on this repo).
#   * MOUNT A NETWORK VOLUME at /workspace. Pods get terminated (e.g. balance hits $0)
#     and lose their container disk; a network volume survives and lets you RESUME.
set -euo pipefail

# ── Config (all overridable via env) ──────────────────────────────────────────
DATASET_REPO="${DATASET_REPO:-bklassen3434/so101_pick_object_20260713_221513}"  # timestamped id (LeRobot stamps _YYYYMMDD_HHMMSS at creation)
BASE_CKPT="${BASE_CKPT:-lerobot/smolvla_base}"
OUTPUT_REPO="${OUTPUT_REPO:-bklassen3434/smolvla_so101}"     # fine-tuned ckpt -> Hub (runtime.policy_path)
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/smolvla_so101}" # keep on the network volume
LEROBOT_DIR="${LEROBOT_DIR:-/workspace/lerobot}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
NUM_WORKERS="${NUM_WORKERS:-20}"
SEED="${SEED:-1000}"
# Maps the SO-101 cameras onto SmolVLA base's expected slots. Dataset {camera1,camera2}
# is a subset of policy {camera1,camera2,camera3}, which the validator accepts.
RENAME_MAP='{"observation.images.overhead": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

export HF_HOME="${HF_HOME:-/workspace/hf}"     # cache dataset + base model on the volume
export HF_HUB_DISABLE_XET=1
export PIP_BREAK_SYSTEM_PACKAGES=1

echo ">>> GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ">>> System deps"
apt-get update -y && apt-get install -y --no-install-recommends git ffmpeg tmux

echo ">>> LeRobot checkout"
[ -d "$LEROBOT_DIR/.git" ] || git clone https://github.com/huggingface/lerobot "$LEROBOT_DIR"
cd "$LEROBOT_DIR"

echo ">>> Pin the preinstalled torch build so the install can't swap in a CPU/other-CUDA wheel"
python - <<'PY' > /tmp/constraints.txt
import torch, torchvision, torchaudio
for m in (torch, torchvision, torchaudio):
    print(f"{m.__name__.split('.')[0]}=={m.__version__}")
PY
cat /tmp/constraints.txt

echo ">>> Install LeRobot + SmolVLA + dataset extras"
python -m pip install -e ".[smolvla,dataset]" -c /tmp/constraints.txt

echo ">>> Sanity: CUDA + SmolVLA import"
python - <<'PY'
import torch; assert torch.cuda.is_available(), "CUDA not visible to torch"
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
except ImportError:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
print("torch", torch.__version__, "| SmolVLA OK |", torch.cuda.get_device_name(0))
PY

echo ">>> HF auth (needed to pull dataset + push checkpoint)"
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
    || hf auth login --token "$HF_TOKEN" 2>/dev/null || true
fi

# ── Resume if a checkpoint already exists on the volume, else fresh fine-tune ──
LAST_CFG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
if [ -f "$LAST_CFG" ]; then
  echo ">>> RESUME from $(readlink -f "$OUTPUT_DIR/checkpoints/last")"
  # train_config.json already carries dataset, rename_map, video_backend, steps,
  # num_workers, push_to_hub, output_dir — so --config_path restores everything.
  lerobot-train --config_path="$LAST_CFG" --resume=true
else
  echo ">>> FRESH fine-tune: ${BASE_CKPT} on ${DATASET_REPO}"
  lerobot-train \
    --policy.path="$BASE_CKPT" \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id="$OUTPUT_REPO" \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.video_backend=pyav \
    --rename_map="$RENAME_MAP" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --num_workers="$NUM_WORKERS" \
    --seed="$SEED" \
    --wandb.enable=false \
    --output_dir="$OUTPUT_DIR"
fi

cat <<MSG

Done. Checkpoint dir: ${OUTPUT_DIR}   Hub: ${OUTPUT_REPO}
The Mac pulls ${OUTPUT_REPO} for inference (configs/smolvla_so101.yaml runtime.policy_path).
Checkpoints save every ${SAVE_FREQ} steps to the volume; push_to_hub uploads at the end.
STOP the pod when idle. Because /workspace is a network volume, you can resume by
re-running this script on a fresh pod with the same volume mounted at /workspace.
MSG
