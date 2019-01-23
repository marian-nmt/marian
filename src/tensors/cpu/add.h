/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "functional/functional.h"
#include "functional/shape.h"
#include "functional/tensor.h"
#include "functional/tmp.h"
#include "tensors/tensor.h"

namespace marian {

namespace cpu {

template <size_t K, class Functor>
void gAddGeneric(Functor functor,
                 const functional::Shape full,
                 functional::Tensor<float> out,
                 functional::Array<functional::Tensor<float>, K> ins,
                 float scale = 1.0) {
  int outLength = out.shape().elements();
  bool same = outLength == full.elements();
  for(size_t i = 0; i < K; ++i)
    same = same && outLength == ins[i].shape().elements();

  constexpr size_t N = functional::Shape::size();
  functional::Array<int, N> len;
  for(int i = 0; i < N; ++i)
    len[i] = full[i] / out.shape()[i];

  functional::Array<int, N> dims;
  for(int index = 0; index < outLength; ++index) {
    if(same) {
      out[index] += functional::apply(functor, ins, index) * scale;
    } else {
      out.shape().dims(index, dims);
      out[index] += functional::loops(functor, ins, len, dims) * scale;
    }
  }
}

template <size_t K, class Functor>
void gAddEqual(Functor functor,
               functional::Tensor<float> out,
               functional::Array<functional::Tensor<float>, K> ins,
               float scale,
               bool broadcast) {
  int length = out.shape().elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int index = 0; index < length; ++index) {
    functional::Array<int, K> indices;
    indices.fill(index);

    if(broadcast) {
      out.shape().dims(index, dims);
      for(size_t i = 0; i < K; ++i)
        indices[i] = ins[i].shape().bindex(dims);
    }

    out[index] += functional::apply(functor, ins, indices) * scale;
  }
}

template <size_t K, class Functor>
void gAddReduce(Functor functor,
                const functional::Shape full,
                functional::Tensor<float> out,
                functional::Array<functional::Tensor<float>, K> ins,
                float scale = 1.0) {
  int rows = full.elements() / full.back();
  int cols = full.back();

  bool same = true;
  for(size_t i = 0; i < K; ++i)
    same = same && ins[i].shape().elements() == full.elements();

  for(int j = 0; j < rows; ++j) {
    float sum = 0;
    if(same) {
      for(int id = 0; id < cols; ++id)
        sum += functional::apply(functor, ins, j * cols + id);
    } else {
      functional::Array<int, functional::Shape::size()> dims;
      for(int id = 0; id < cols; ++id) {
        full.dims(j * cols + id, dims);
        functional::Array<int, K> indices;
        for(size_t i = 0; i < K; ++i)
          indices[i] = ins[i].shape().bindex(dims);
        sum += functional::apply(functor, ins, indices);
      }
    }
    out[j] += sum * scale;
  }
}

template <class Functor, class... Tensors>
void Add(Functor functor, float scale, marian::Tensor out, Tensors... tensors) {
  auto full = marian::Shape::broadcast({out, tensors...});

  //int length = out->shape().elements();

  constexpr size_t K = sizeof...(Tensors);

  functional::Tensor<float> gOut = out;
  functional::Array<functional::Tensor<float>, K> gIns = {tensors...};

  if(full.back() != 1 && out->shape().back() == 1) {
    //size_t m = full.elements() / length;
    //size_t k = full.back();
    cpu::gAddReduce(functor, full, gOut, gIns, scale);
  } else if(out->shape() == full) {
    bool broadcast = false;
    for(size_t i = 0; i < K; ++i)
      broadcast = broadcast || gOut.shape() != gIns[i].shape();
    cpu::gAddEqual(functor, gOut, gIns, scale, broadcast);
  } else {
    cpu::gAddGeneric(functor, full, gOut, gIns, scale);
  }
}
}  // namespace cpu
}  // namespace marian
