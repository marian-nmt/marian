#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {

// @TODO: generalize to vector operations, possible using specializations

// single loop over outer dimension. Recursively creates nested loops
// down to inner dimension and to single elements. Since this is based
// on strides, it correctly broadcasts to all dimensions without additional
// computation.
// Compiler optimizes this to single construct with nested(?) loops.
template <size_t I = 0> struct E {
  template <size_t K, class Functor>
  static inline void element(const Functor& functor,
                             functional::Array<functional::Tensor<float>, K>& tensors,
                             functional::Array<int, K> indices) {

    auto& shape = tensors[0].shape();

    // loop for outer-most dimension
    for(int i = 0; i < shape[I]; ++i) {

      // call loop for next-inner dimension
      E<I + 1>::element(functor, tensors, indices);

      // increase index for current dimension by stride or 0 if broadcasting. bstride(i)
      // is look-up value, either equal to stride if the corresponding dim is larger 1 or
      // 0 if the dim is 1.
      for(int k = 0; k < K; ++k)
        indices[k] += tensors[k].shape().bstride(I);
    }
  }
};

// specialization for inner-most single element (recursive stopping criterion)
// using const reference for indices here to avoid copying. No loop.
template <> struct E<functional::Shape::size()> {
  template <size_t K, class Functor>
  static inline void element(const Functor& functor,
                             functional::Array<functional::Tensor<float>, K>& tensors,
                             const functional::Array<int, K>& indices) {

    // just apply the function for all indexed elements across all tensors
    tensors[0][indices[0]] = functional::apply(functor, tensors, indices);

  }
};

template <class Functor, class... Tensors>
void Element(const Functor& functor, marian::Tensor out, Tensors... tensors) {
  constexpr size_t K = sizeof...(tensors) + 1;
  functional::Array<functional::Tensor<float>, K> gTensors = {out, tensors...};

  // create and initialize indices to 0
  functional::Array<int, K> indices;
  indices.fill(0);

  // call elementwise operation going from outer-most dimension
  // to inner-most element.
  E<>::element(functor, gTensors, indices);
}

}
}
