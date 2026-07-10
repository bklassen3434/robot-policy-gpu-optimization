# robot-policy-gpu-optimization

**Built a transformer robot-arm policy from scratch in PyTorch, wrote a custom CUDA kernel to make it _N×_ faster, and proved accuracy held with automated evals.**

This repo trains an [ACT](https://tonyzhaozh.github.io/aloha/) (Action Chunking Transformer) policy — implemented from scratch, no pre-built transformer blocks — on a public [LeRobot](https://github.com/huggingface/lerobot) dataset, matches the official implementation's task success rate, then profiles the model and replaces the single hottest operation with a hand-written CUDA kernel that produces numerically identical outputs.

## Results

> Filled in after training + profiling on the GPU box. Placeholders below.

| Metric | Value |
| --- | --- |
| Dataset | `lerobot/aloha_sim_transfer_cube_human` |
| Task success (ours / official) | `__% / __%` |
| Hottest op (NSight) | `TBD` |
| Latency before → after | `__ ms → __ ms` (`__×`) |
| Accuracy after kernel | unchanged (parity to `1e-_`) |

![latency chart](docs/latency.png)

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
