#pragma once

#include "tensors/tensor.h"

#ifdef __CUDACC__
#include "tensors/gpu/cuda_helpers.h"
#endif

namespace marian {
namespace gpu {

#ifdef __CUDACC__
template <size_t K, bool broadcast, class Functor>
__global__ void gElement(Functor functor,
                         gpu::Array<gpu::Tensor<float>, K> tensors) {

  int length = tensors[0].shape().elements();
  gpu::Array<int, gpu::Shape::size()> dims;
  gpu::Array<int, K> indices;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {

      indices.fill(index);

      if(broadcast) {
        tensors[0].shape().dims(index, dims);
        for(int i = 1; i < K; ++i)
          indices[i] = tensors[i].shape().bindex(dims);
      }

      tensors[0][index] = gpu::apply(functor, tensors, indices);
    }
  }
}
#endif

template <class Functor, class ...Tensors>
void Element(Functor functor, marian::Tensor out, Tensors ...tensors) {
#ifdef __CUDACC__
  cudaSetDevice(out->getDevice().no);

  constexpr size_t K = sizeof...(tensors) + 1;
  gpu::Array<gpu::Tensor<float>, K> gTensors = {out, tensors...};

  int length = gTensors[0].shape().elements();
  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  bool broadcast = false;
  for(int i = 1; i < K; ++i)
    broadcast = broadcast || gTensors[0].shape() != gTensors[i].shape();

  if(broadcast)
    gpu::gElement<K, true><<<blocks, threads>>>(functor, gTensors);
  else
    gpu::gElement<K, false><<<blocks, threads>>>(functor, gTensors);
#else
  ABORT("Not implemented");
#endif
}

}
}
