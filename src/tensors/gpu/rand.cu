#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensors/tensor_operators.h"
#include "tensors/gpu/backend.h"

// @TODO move this to a header file
#define CUDA_CALL(x)                                  \
  do {                                                \
    if((x) != cudaSuccess) {                          \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      exit(1);                                        \
    }                                                 \
  } while(0)

#define CURAND_CALL(x)                                \
  do {                                                \
    if((x) != CURAND_STATUS_SUCCESS) {                \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      exit(1);                                        \
    }                                                 \
  } while(0)

namespace marian {
namespace gpu {

void Uniform(Tensor tensor, float a, float b) {
  ABORT_IF(a >= b, "");

  auto gpuBackend
      = std::static_pointer_cast<gpu::Backend>(tensor->getBackend());
  curandGenerator_t gen = gpuBackend->getCurandGenerator();
  CURAND_CALL(curandGenerateUniform(gen, tensor->data(), tensor->size()));

  // curandGenerateUniform has no range parameters (why?) so we need to
  // scale and shift inplace if range is different than [0, 1). 
  using namespace functional;
  if(a != 0.f || b != 1.f)
    gpu::Element(_1 = (b - a) * _1 + a, tensor);
}

void Normal(Tensor tensor, float mju, float sigma) {
  auto gpuBackend
      = std::static_pointer_cast<gpu::Backend>(tensor->getBackend());
  curandGenerator_t gen = gpuBackend->getCurandGenerator();
  CURAND_CALL(curandGenerateNormal(gen, tensor->data(), tensor->size(), mju, sigma));
}

}  // namespace gpu
}  // namespace marian
