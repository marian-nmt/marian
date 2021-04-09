#include <stdint.h>
#include "tensors/tensor.h"
#include "tensors/gpu/cuda_helpers.h"
#include "tensors/gpu/backend.h"

namespace marian {
namespace gpu {

template <typename T, typename ActFunc>
__global__ static void gBiasAddFused(T* tensor, T* bias, size_t tensor_size, size_t bias_size, ActFunc f) {
  const size_t row_start = blockIdx.x * bias_size;
  for(int bias_offset = threadIdx.x; bias_offset < bias_size; bias_offset+=blockDim.x) {
    size_t offset_into_tensor = row_start + bias_offset;
    if(offset_into_tensor < tensor_size) {
      T added_bias = tensor[offset_into_tensor] + bias[bias_offset];
      tensor[offset_into_tensor] = f(added_bias);
    }  
  }
}

struct identity {
  template <typename T>
  __device__ constexpr T&& operator() (T&& t) const noexcept {
    return std::forward<T>(t);
  }
};

struct reluAct {
  template <typename T>
  __device__ T operator() (T t) const noexcept {
    return t > (T) 0? t : (T) 0;
  }
};

void BiasAdd(marian::Tensor C, const marian::Tensor& bias, bool do_relu) {
  auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
  CUDA_CHECK(cudaSetDevice(backend->getDeviceId().no));

  size_t size = C->shape().elements();
  size_t bias_size = bias->shape().elements();

  int m = C->shape().elements() / C->shape().back();
  int n = C->shape().back();

  ABORT_IF(n != bias_size, "The number of elements in the bias must match the number of columns in C");

  int threads_per_block = std::min(MAX_THREADS, n);
  int blocks = m;

  if(C->type() == Type::float32) {
    if (do_relu)
      gBiasAddFused<<<blocks, threads_per_block>>>(C->data<float>(), bias->data<float>(), size, bias_size, reluAct());
    else
      gBiasAddFused<<<blocks, threads_per_block>>>(C->data<float>(), bias->data<float>(), size, bias_size, identity());
    
#if COMPILE_FP16
  } else if(C->type() == Type::float16) {
      if (do_relu)
        gBiasAddFused<<<blocks, threads_per_block>>>(C->data<half>(), bias->data<half>(), size, bias_size, reluAct());
      else 
        gBiasAddFused<<<blocks, threads_per_block>>>(C->data<half>(), bias->data<half>(), size, bias_size, identity());
#endif
  } else {
    ABORT("Prod not implemented for type {}", C->type());
  } 
}

}
}