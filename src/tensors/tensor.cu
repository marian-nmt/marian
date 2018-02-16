
#include "tensors/tensor.h"

#include "kernels/cuda_helpers.h"
#include "kernels/tensor_operators.h"

namespace marian {

void TensorBase::setSparse(const std::vector<size_t> &k,
                           const std::vector<float> &v) {
  CUDA_CHECK(cudaSetDevice(backend_->getDevice().no));
  SetSparse(data(), k, v);
  cudaStreamSynchronize(0);
}

}
