# Writeup

> Draft after training + kernel work. Keep it short and concrete — this is where reviewers look.

## 1. The model (from scratch)

- What ACT is in two sentences; why a transformer fits action-chunking.
- What I implemented by hand (MHA, enc/dec, CVAE, pos embeddings) vs reused (ResNet-18 backbone).
- Training result: matched the official success rate (`__%` vs `__%`) on `aloha_sim_transfer_cube_human`.

## 2. Finding the hotspot (NSight)

- Screenshot / table from NSight Systems.
- The single op eating the most time: `TBD`. Why (launch overhead / memory-bound / no tuned lib equivalent).

## 3. The custom CUDA kernel

- What I fused and why it's faster (fewer launches / better memory traffic / occupancy).
- Kernel design notes: block/grid layout, shared memory, reductions.
- Parity: output identical to the PyTorch reference within `1e-_`.

## 4. Results

- Latency: `__ ms → __ ms` (`__×`). Chart: `latency.png`.
- Accuracy held: automated eval (`evals/test_accuracy_regression.py`) green.

## 5. What I'd do next

- Backward-pass kernel, autotuning, fp16/bf16, multi-op fusion.
