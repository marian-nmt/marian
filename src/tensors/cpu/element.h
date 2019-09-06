#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {

// Function in this header are supposed to execute element-wise operations
// (passed in as a Functor) on arbitrary numbers of tensors. The templates
// are required to implement correct broadcasting of operations across
// a fixed-at-compile-time but in principle arbitrary number of dimensions.

// @TODO: generalize to vector operations, possible using specializations

// single loop over outer dimension. Recursively creates nested loops
// down to inner dimension and to single elements. Since this is based
// on strides, it correctly broadcasts to all dimensions without additional
// computation.
// Compiler optimizes this to single construct with nested(?) loops.

namespace f = marian::functional;

template <size_t I = 0>
struct E {
  template <size_t numArg, class Functor, typename ElementType>
  static inline void element(
      const Functor& functor,
      f::Array<f::Tensor<ElementType>, numArg>& tensors,
      f::Array<int, numArg> indices) {
    const auto& shape = tensors[0].shape();

    // loop over outer-most dimension
    for(int i = 0; i < shape[I]; ++i) {
      // call loop for next-inner dimension
      E<I + 1>::element(functor, tensors, indices);

      // increase index for current dimension by stride or 0 if broadcasting.
      // bstride(i) is look-up value, either equal to stride if the
      // corresponding dim is larger 1 or 0 if the dim is 1.
      for(size_t k = 0; k < numArg; ++k) {
        //int stride = tensors[k].shape().stride(I);
        //indices[k] += stride == 1 ? 0 : stride;
        indices[k] += tensors[k].shape().bstride(I);
      }
    }
  }
};

// specialization for inner-most single element (recursive stopping criterion)
// using const reference for indices here to avoid copying. No loop.
template <>
struct E<f::Shape::size()> {
  template <size_t numArg, class Functor, typename ElementType>
  static inline void element(
      const Functor& functor,
      f::Array<f::Tensor<ElementType>, numArg>& tensors,
      const f::Array<int, numArg>& indices) {
    // just apply the function for all indexed elements across all tensors
    // @TODO: use converting operator[] on tensor
    tensors[0].data()[indices[0]] = f::apply(functor, tensors, indices);
  }
};

template <typename ElementType, class Functor, class... Tensors>
void element(const Functor& functor, marian::Tensor out, Tensors... tensors) {

  // Number of input tensors + 1 (output tensor)
  constexpr size_t argNum = sizeof...(tensors) + 1;
  // create and initialize indices to 0, one index per tensor
  f::Array<int, argNum> indices;
  indices.fill(0);

  // call elementwise operation going from outer-most dimension
  // to inner-most element.
  f::Array<f::Tensor<ElementType>, argNum> gTensors = {out, tensors...};
  E<0>::element(functor, gTensors, indices);
}

template <class Functor, class... Tensors>
void elementFloat(const Functor& functor, marian::Tensor out, Tensors... tensors) {
#ifndef __CUDA_ARCH__
  std::vector<marian::Tensor> ts({tensors...});
  bool div8 = true;
  bool div4 = true;

  if(out->shape()[-1] % 8 != 0)
    div8 = false;
  if(out->shape()[-1] % 4 != 0)
    div4 = false;
  for(auto t : ts) {
    if(t->shape()[-1] % 8 != 0)
      div8 = false;
    if(t->shape()[-1] % 4 != 0)
      div4 = false;
  }

  if(div8) {
    // std::cerr << "8: " << functor.to_string() << std::endl;
    element<float32x8>(functor, out, tensors...);
    return;
  }

  if(div4) {
    // std::cerr << "4: " << functor.to_string() << std::endl;
    element<float32x4>(functor, out, tensors...);
    return;
  }
#endif
  // std::cerr << "1: " << functor.to_string() << std::endl;
  element<float>(functor, out, tensors...);
}

// main call to function executing element-wise operation
template <class Functor, class... Tensors>
void Element(const Functor& functor, marian::Tensor out, Tensors... tensors) {
  switch(out->type()) {
    case Type::float32: elementFloat(functor, out, tensors...); break;
    //case Type::uint32:  element<uint32_t>(functor, out, tensors...); break;
    default: ABORT("Unsupported type for element-wise operation"); break;
  }
}

}  // namespace cpu
}  // namespace marian
