#include "tensors/gpu/element.h"

#include "functional/array.h"
#include "functional/functional.h"
#include "functional/tensor.h"
#include "functional/tmp.h"

#include "tensors/gpu/cuda_helpers.h"

namespace marian {
namespace gpu {

template <size_t K, bool broadcast, class Functor, typename T>
__global__ void gElement(
    Functor functor,
    functional::Array<functional::Tensor<T>, K> tensors) {
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

      // This performs the internal application of the functor in float32 regardless of the input type.
      // It seems there are no speed penalties but improved precision.
      tensors[0].data()[index] = (T)functional::applyWithCast<float>(functor, tensors, indices);
    }
  }
}


template <typename T, class Functor, class... Tensors>
void ElementTyped(Functor functor, Tensor out, Tensors... tensors) {
  //matchOrAbort<T>(out->type()); // @TODO: figure out undefined reference

  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();
  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  constexpr size_t K = sizeof...(tensors) + 1;
  functional::Array<functional::Tensor<T>, K> gTensors = {out, tensors...};

  bool broadcast = false;
  for(int i = 1; i < K; ++i)
    broadcast = broadcast || gTensors[0].shape() != gTensors[i].shape();
  if(broadcast)
    gElement<K, true><<<blocks, threads>>>(functor, gTensors);
  else
    gElement<K, false><<<blocks, threads>>>(functor, gTensors);
}

template <class Functor, class... Tensors>
void Element(Functor functor, Tensor out, Tensors... tensors) {
  checkCommonType(out, tensors...);

  if(out->type() == Type::float32) {
    ElementTyped<float>(functor, out, tensors...);
  } else if(out->type() == Type::float16) {
#if COMPILE_FP16
    ElementTyped<half>(functor, out, tensors...);
#else
    ABORT("FP16 not supported with chosen current hardware or CUDA version");
#endif
  } else if(out->type() == Type::float64) {
    ElementTyped<double>(functor, out, tensors...);
  } else {
    ABORT("Type {} not yet supported", out->type());
  }
}

#include "tensors/gpu/element.inc"
}  // namespace gpu
}  // namespace marian
