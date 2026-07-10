# Profiling

Goal of this step: find the **single op eating the most time**, so the custom CUDA
kernel targets something that actually moves latency.

## Run

```bash
bash profiling/run_nsight.sh
nsys stats --report cuda_gpu_kern_sum profiling/reports/act.nsys-rep
```

`robopolicy.profile_step` runs a fixed number of forward+backward steps wrapped in
`torch.cuda.cudaProfilerApi` start/stop and NVTX ranges (`attention`, `layernorm`,
`ffn`, `backbone`) so the timeline is readable.

## Reading it

- Sort kernels by total GPU time. The top one is your target.
- Note *why* it's hot: many tiny launches (→ **fuse**), memory-bound (→ better
  traffic/occupancy), or an op with no tuned library kernel (→ **write one**).
- For a small transformer the usual suspects are attention (many small GEMMs +
  softmax) and the pointwise LayerNorm/residual/activation chains (launch-overhead
  bound → great fusion candidates).

## Record

Drop the summary table + a screenshot into `docs/writeup.md` §2 and set the
"Hottest op" row in the README results table.
