# Engineering plan

One-line pitch: *Built a transformer from scratch, wrote a custom CUDA kernel to make it N× faster, and proved accuracy held with automated evals.*

The three places the real signal lives — and where to spend the most time — are the **from-scratch model** (`src/robopolicy/model`), the **CUDA kernel** (`kernels/`), and the **writeup** (`docs/writeup.md`).

## Current status (2026-07-12)

**Phase 1 (build + train + eval) — DONE.**
- From-scratch ACT transformer implemented and unit-tested.
- Trained 100k steps on an A100 80GB: final L1 `0.077`, loss `0.084`.
- **Sim eval: 52.0% success (26/50)** on gym-aloha `AlohaTransferCube` — a working policy, in line with the official ACT reproduction (~50%).
- Trained checkpoint backed up locally at `~/runpod_checkpoints/robot-policy-gpu/last.pt` (620 MB, byte-verified).

**Phase 2 (profile + kernel + prove) — NEXT. This is the high-signal half.**
1. **Profile** — restart a GPU pod, re-upload the checkpoint, `bash profiling/run_nsight.sh` → find the single hottest op (steps 5).
2. **CUDA kernel** — rewrite that op in `kernels/` (fuse or hand-tune), keep output identical (step 6).
3. **Benchmark** — `kernels/bench.py`, report before→after latency (step 7).
4. **Accuracy held** — `evals/test_accuracy_regression.py`: kernel parity + re-run sim eval, confirm 52% holds (step 8).
5. **Writeup + latency chart** — `docs/writeup.md`, `docs/latency.png` (step 9); can be done locally.

### To resume on a fresh GPU pod
Old EU-RO-1 network volume was deleted, so any region works now:
```bash
cd /workspace && git clone https://github.com/bklassen3434/robot-policy-gpu-optimization.git
cd robot-policy-gpu-optimization && bash scripts/runpod_setup.sh
# upload the backed-up checkpoint from the Mac to outputs/last.pt (scp via the pod's direct TCP SSH), then:
make eval     # sanity: should reproduce ~52%
make profile  # NSight -> hottest op
```
Eval needs `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` and the GLVND EGL loader
(`apt-get install -y libegl1 libglvnd0 libgles2 libgl1`) for headless MuJoCo.

## Decisions (locked)

- **Policy:** ACT (Action Chunking Transformer). LeRobot's flagship *transformer* policy, so "match the official accuracy" is a concrete, citable target.
- **Dataset:** `lerobot/aloha_sim_transfer_cube_human`. Has a matching sim env (`gym-aloha`) so accuracy is a real task **success rate**, not just a loss.
- **GPU:** A100/H100 by the hour on RunPod. On only while actively working.
- **Accuracy metrics:**
  - Match-official (step 4): validation L1 on held-out actions **+** sim rollout success rate.
  - Held-after-kernel (step 8): numerical parity of kernel vs reference (tight tol) **+** identical eval metric.

## Local (Mac / Apple Silicon) vs GPU box

Everything CUDA-specific — `nvcc`, NSight, kernel parity — runs only on the rented box. Everything else is developed and tested locally on CPU/MPS first.

| Do locally (M3, CPU/MPS) | Do on the GPU box (A100/H100) |
| --- | --- |
| Build the from-scratch model | Full training to convergence |
| Unit tests, overfit-one-batch | NSight profiling (find hottest op) |
| Write train / eval / bench / parity harnesses | Compile the CUDA kernel (`nvcc`) |
| Dry-run the training loop on a tiny subset | Kernel parity + latency benchmark |
|  | Sim rollout eval (`gym-aloha` / MuJoCo) |

## Steps

1. **Rent GPU** — `scripts/runpod_setup.sh`. Turn off when idle.
2. **Dataset** — `scripts/download_dataset.py` pulls `aloha_sim_transfer_cube_human` via `LeRobotDataset`.
3. **From-scratch model** — `model/transformer.py` (MHA, enc/dec, pos emb) + `model/act.py` (CVAE + policy). ResNet-18 backbone reused from torchvision, like official ACT.
4. **Train to match official** — `train.py`. Target: match official ACT success rate on transfer-cube. Track val L1 + success rate.
5. **Profile** — `profiling/run_nsight.sh` → NSight Systems → identify the single op eating the most time.
6. **Custom CUDA kernel** — `kernels/`. Rewrite that op: either **fuse** several steps (cut launch overhead) or build a **fast version** of a step with no tuned library equivalent. Keep output identical. Likely candidates for a small transformer: fused LayerNorm+residual, fused bias+activation, or a FlashAttention-style fused attention.
7. **Benchmark** — `kernels/bench.py`. Report `before ms → after ms (N×)`.
8. **Evals** — `evals/test_accuracy_regression.py`. Prove task accuracy held: kernel output parity + unchanged eval metric.
9. **Package** — README results table, `docs/writeup.md`, `docs/latency.png`.

## Kernel target — decide from the profile, not now

Don't pre-commit the op. NSight picks it. The `kernels/` harness is written generic so you swap in whatever the profile says:
- `reference.py` — the PyTorch reference (source of truth for parity).
- `csrc/` — the `.cu`/`.cpp` custom kernel (template: fused LayerNorm forward).
- `test_parity.py` — asserts kernel == reference within tolerance.
- `bench.py` — CUDA-event-timed latency, before vs after.

The template targets fused LayerNorm because it's the simplest op to get bit-close parity on; swap the reference + kernel for the profiled hotspot (attention fusion is the likely real win).

## Cost discipline

- Develop/validate locally; only rent the GPU for train + profile + kernel.
- Save checkpoints to persistent volume so you can stop the pod.
- A single training run on transfer-cube is short; most GPU time goes to the kernel iterate-profile loop.
