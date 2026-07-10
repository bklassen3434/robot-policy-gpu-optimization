#!/usr/bin/env bash
# Profile one forward+backward step with NSight Systems to find the hottest op.
# Run on the GPU box. Produces profiling/reports/act.nsys-rep (open in nsys-ui).
set -euo pipefail

mkdir -p profiling/reports

nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --output=profiling/reports/act \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  python -m robopolicy.profile_step --config configs/act_aloha.yaml

echo
echo "Report: profiling/reports/act.nsys-rep"
echo "Summarize the top CUDA kernels by time:"
echo "  nsys stats --report cuda_gpu_kern_sum profiling/reports/act.nsys-rep"
echo
echo "Also useful (per-kernel deep dive with NSight Compute):"
echo "  ncu --set full -o profiling/reports/act_ncu python -m robopolicy.profile_step --config configs/act_aloha.yaml --steps 1"
