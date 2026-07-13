# robot-policy-gpu-optimization

**Built a transformer robot-arm policy from scratch in PyTorch, profiled it with NSight, and wrote a custom fused CUDA kernel for its hottest hand-optimizable op — 1.3–1.4× on that op, numerically parity-verified, task accuracy held.**

This repo trains an [ACT](https://tonyzhaozh.github.io/aloha/) (Action Chunking Transformer) policy — implemented from scratch, no pre-built transformer blocks — on a public [LeRobot](https://github.com/huggingface/lerobot) dataset, matches the official implementation's task success rate, then profiles the model to find where the GPU time actually goes and writes a hand-tuned CUDA kernel for the one operation a custom kernel can beat. The profile shows the model is GEMM/conv-bound (backbone convs + attention/FFN matmuls are already on optimal NVIDIA libraries, and attention is already fused), so the honest target is the launch/memory-bound **residual-add + LayerNorm** glue — fused into one kernel, output-identical to PyTorch, with the end-to-end impact reported truthfully (~1–2%). The point is the profiling judgment, not a cherry-picked headline.

## Results

| Metric | Value |
| --- | --- |
| Dataset | `lerobot/aloha_sim_transfer_cube_human` (50 episodes, 20k frames) |
| Training | 100k steps on an A100 80GB; final L1 `0.077`, loss `0.084` |
| **Task success (ours)** | **52.0%** (26/50 sim rollouts, gym-aloha `AlohaTransferCube`) |
| Task success (official ACT) | ~50% on this task (LeRobot reproduction range) |
| Profiled bottleneck (NSight) | GEMM/conv-bound: ResNet backbone + attention/FFN matmuls (already optimal libs); attention already fused |
| Custom kernel | fused **residual-add + LayerNorm** (the hand-optimizable pointwise op) |
| Kernel latency (RTX 4090) | **1.3–1.4×** vs PyTorch `add + layer_norm`, op-level (e.g. `8.5 → 6.4 µs`) |
| End-to-end impact | ~1–2% (model is GEMM-bound — see writeup for the honest framing) |
| Accuracy after kernel | **held** — parity `rtol 1e-4` (11/11) + sim success 52% vs 56% baseline (within run-to-run noise) |

![latency chart](docs/latency.png)

See [`docs/writeup.md`](docs/writeup.md) for the full profile → kernel → benchmark → accuracy-held story.

## What's from scratch

- Multi-head attention, transformer encoder/decoder, sinusoidal 1D/2D position embeddings — `src/robopolicy/model/transformer.py`
- The ACT CVAE + encoder/decoder policy assembly — `src/robopolicy/model/act.py`
- The custom fused CUDA kernel + PyTorch reference + parity harness — `kernels/`

The only reused component is the ResNet-18 image backbone (torchvision), exactly as the official ACT does — the transformer is the point.

## Repo layout

```
src/robopolicy/       from-scratch model, data, train, eval
kernels/              PyTorch reference op, custom CUDA kernel, parity test, benchmark
evals/                accuracy-regression test (accuracy held after the kernel swap)
profiling/            NSight Systems run + how-to
configs/              training config (mirrors official ACT hyperparams)
scripts/              RunPod setup, dataset download
docs/                 engineering plan + writeup + latency chart
```

## Quickstart

**Local (Mac / CPU / MPS)** — build & validate the model, no GPU needed:
```bash
make install
make test        # model shape + overfit-one-batch sanity checks
```

**GPU box (A100/H100 on RunPod)** — train, profile, build the kernel:
```bash
bash scripts/runpod_setup.sh
make install-gpu
make train
make eval
make profile     # NSight -> find the hottest op
make parity      # custom kernel == PyTorch reference
make bench       # latency before/after
```

See [`docs/PLAN.md`](docs/PLAN.md) for the full engineering plan and the local-vs-GPU split.
