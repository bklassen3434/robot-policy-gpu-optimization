// Torch binding for the fused LayerNorm CUDA kernel.
#include <torch/extension.h>

// defined in layernorm_kernel.cu
torch::Tensor layernorm_forward_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "weight/bias must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 is supported");
  TORCH_CHECK(weight.numel() == x.size(-1), "weight size must equal normalized dim");
  TORCH_CHECK(bias.numel() == x.size(-1), "bias size must equal normalized dim");
  return layernorm_forward_cuda(x, weight, bias, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layernorm_forward", &layernorm_forward, "Fused LayerNorm forward (CUDA)");
}
