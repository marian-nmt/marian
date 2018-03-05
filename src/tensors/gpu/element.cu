

#include "tensors/gpu/element.h"
#include "tensors/gpu/cuda_helpers.h"
#include "functional/array.h"
#include "functional/tensor.h"
#include "functional/tmp.h"
#include "functional/functional.h"

namespace marian {
namespace gpu {

template <size_t K, bool broadcast, class Functor>
__global__ void gElement(Functor functor,
                         functional::Array<functional::Tensor<float>, K> tensors) {

  int length = tensors[0].shape().elements();
  functional::Array<int, functional::Shape::size()> dims;
  functional::Array<int, K> indices;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {

      indices.fill(index);

      if(broadcast) {
        tensors[0].shape().dims(index, dims);
        for(int i = 1; i < K; ++i)
          indices[i] = tensors[i].shape().bindex(dims);
      }

      tensors[0][index] = functional::apply(functor, tensors, indices);
    }
  }
}

template <class Functor, class ...Tensors>
void Element(Functor functor, Tensor out, Tensors ...tensors) {
  cudaSetDevice(out->getDevice().no);

  constexpr size_t K = sizeof...(tensors) + 1;
  functional::Array<functional::Tensor<float>, K> gTensors = {out, tensors...};

  int length = gTensors[0].shape().elements();
  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  bool broadcast = false;
  for(int i = 1; i < K; ++i)
    broadcast = broadcast || gTensors[0].shape() != gTensors[i].shape();

  if(broadcast)
    gElement<K, true><<<blocks, threads>>>(functor, gTensors);
  else
    gElement<K, false><<<blocks, threads>>>(functor, gTensors);
}

#include "tensors/gpu/element.inc"


}
}

