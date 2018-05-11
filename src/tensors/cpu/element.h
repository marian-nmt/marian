/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {

template <size_t N, size_t K, class Functor>
struct E {
  static inline void element(Functor functor,
                             functional::Array<functional::Tensor<float>, K> tensors,
                             functional::Array<int, K> indices) {

    auto& shape = tensors[0].shape();
    for(int i = 0; i < shape[functional::Shape::size() - N]; ++i) {
      E<N - 1, K, Functor>::element(functor, tensors, indices);
      for(int k = 0; k < K; ++k) {
         indices[k] += tensors[k].shape().bstride(functional::Shape::size() - N);
      }
    }
  }
};

template <size_t K, class Functor>
struct E<1, K, Functor> {
  static inline void element(Functor functor,
                             functional::Array<functional::Tensor<float>, K> tensors,
                             functional::Array<int, K> indices) {

    auto& shape = tensors[0].shape();
    for(int i = 0; i < shape[functional::Shape::size() - 1]; ++i) {
      tensors[0][indices[0]] = functional::apply(functor, tensors, indices);
      for(int k = 0; k < K; ++k) {
         indices[k] += tensors[k].shape().bstride(functional::Shape::size() - 1);
      }
    }
  }
};

template <class Functor, class... Tensors>
void Element(Functor functor, marian::Tensor out, Tensors... tensors) {
  constexpr size_t K = sizeof...(tensors) + 1;
  functional::Array<functional::Tensor<float>, K> gTensors = {out, tensors...};

  functional::Array<int, K> indices;
  indices.fill(0);
  E<functional::Shape::size(), K, Functor>::element(functor, gTensors, indices);
}

}
}
