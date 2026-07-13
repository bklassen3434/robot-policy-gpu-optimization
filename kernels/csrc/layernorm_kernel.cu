// Fused LayerNorm forward kernel (float32), normalized over the last dim.
//
// One block per row. Threads in the block cooperatively reduce the row's sum and
// sum-of-squares (single pass), compute mean + inverse std once in shared memory,
// then write y = (x - mean) * rstd * weight + bias. Fusing the reduction, the
// normalization, and the affine transform into one launch is the whole point:
// it removes the extra kernel launches and global-memory round trips that a
// naive (mean) -> (subtract) -> (var) -> (scale) sequence would incur.
//
// This is the template op that wires up the parity + benchmark harness. Swap it
// for the profiled hotspot; keep the (x, weight, bias) -> y contract so the tests
// stay valid.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int kThreads = 256;

__global__ void layernorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N,
    float eps) {
  const int row = blockIdx.x;
  const float* x_row = x + static_cast<long>(row) * N;
  float* y_row = y + static_cast<long>(row) * N;

  // per-thread partial sums
  float sum = 0.f;
  float sq = 0.f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    const float v = x_row[i];
    sum += v;
    sq += v * v;
  }

  // block reduction in shared memory
  __shared__ float s_sum[kThreads];
  __shared__ float s_sq[kThreads];
  s_sum[threadIdx.x] = sum;
  s_sq[threadIdx.x] = sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  __shared__ float mean;
  __shared__ float rstd;
  if (threadIdx.x == 0) {
    mean = s_sum[0] / N;
    const float var = s_sq[0] / N - mean * mean;
    rstd = rsqrtf(var + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    y_row[i] = (x_row[i] - mean) * rstd * weight[i] + bias[i];
  }
}

torch::Tensor layernorm_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  auto x2d = x.reshape({-1, x.size(-1)}).contiguous();
  const int M = x2d.size(0);
  const int N = x2d.size(1);
  auto y = torch::empty_like(x2d);

  const dim3 grid(M);
  const dim3 block(kThreads);
  layernorm_forward_kernel<<<grid, block>>>(
      x2d.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      y.data_ptr<float>(),
      N,
      static_cast<float>(eps));
  return y.reshape(x.sizes());
}

// Fused residual-add + LayerNorm: y = LayerNorm(x + residual) * weight + bias.
//
// This is the actual profiled target. Every post-norm transformer sublayer computes
// norm(x + sublayer(x)); the profile showed the standalone elementwise add and the
// LayerNorm as separate launch-overhead-bound kernels. Folding the add into the
// LayerNorm's single-pass reduction removes one kernel launch and one full
// read+write of the (x+residual) tensor through global memory — the sum is consumed
// straight from registers in both the reduction and the normalization pass.
__global__ void residual_layernorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N,
    float eps) {
  const int row = blockIdx.x;
  const long base = static_cast<long>(row) * N;
  const float* x_row = x + base;
  const float* res_row = residual + base;
  float* y_row = y + base;

  float sum = 0.f;
  float sq = 0.f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    const float v = x_row[i] + res_row[i];
    sum += v;
    sq += v * v;
  }

  __shared__ float s_sum[kThreads];
  __shared__ float s_sq[kThreads];
  s_sum[threadIdx.x] = sum;
  s_sq[threadIdx.x] = sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  __shared__ float mean;
  __shared__ float rstd;
  if (threadIdx.x == 0) {
    mean = s_sum[0] / N;
    const float var = s_sq[0] / N - mean * mean;
    rstd = rsqrtf(var + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    const float v = x_row[i] + res_row[i];
    y_row[i] = (v - mean) * rstd * weight[i] + bias[i];
  }
}

torch::Tensor residual_layernorm_forward_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, double eps) {
  auto x2d = x.reshape({-1, x.size(-1)}).contiguous();
  auto r2d = residual.reshape({-1, residual.size(-1)}).contiguous();
  const int M = x2d.size(0);
  const int N = x2d.size(1);
  auto y = torch::empty_like(x2d);

  const dim3 grid(M);
  const dim3 block(kThreads);
  residual_layernorm_forward_kernel<<<grid, block>>>(
      x2d.data_ptr<float>(),
      r2d.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      y.data_ptr<float>(),
      N,
      static_cast<float>(eps));
  return y.reshape(x.sizes());
}
