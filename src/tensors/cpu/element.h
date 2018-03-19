/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {

template <size_t K, bool broadcast, class Functor>
void gElement(Functor functor,
              functional::Array<functional::Tensor<float>, K> tensors) {
  int length = tensors[0].shape().elements();
  functional::Array<int, functional::Shape::size()> dims;
  functional::Array<int, K> indices;

#pragma omp parallel for simd
  for(int index = 0; index < length; ++index) {
    indices.fill(index);
    if(broadcast) {
      tensors[0].shape().dims(index, dims);
      for(int i = 1; i < K; ++i)
        indices[i] = tensors[i].shape().bindex(dims);
    }
    tensors[0][index] = functional::apply(functor, tensors, indices);
  }
}

template <class Functor, class... Tensors>
void Element(Functor functor, marian::Tensor out, Tensors... tensors) {
  constexpr size_t K = sizeof...(tensors) + 1;
  functional::Array<functional::Tensor<float>, K> gTensors = {out, tensors...};

  int length = gTensors[0].shape().elements();

  bool broadcast = false;
  for(int i = 1; i < K; ++i)
    broadcast = broadcast || gTensors[0].shape() != gTensors[i].shape();

  if(broadcast)
    cpu::gElement<K, true>(functor, gTensors);
  else
    cpu::gElement<K, false>(functor, gTensors);
}
}
}
