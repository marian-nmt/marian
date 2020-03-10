/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <cuda.h>
#include <limits>

#include "data/types.h"
#include "tensors/tensor.h"
#include "translator/helpers.h"

#include "tensors/gpu/cuda_helpers.h"

namespace marian {

namespace gpu {

template <typename T>
__global__ void gSetColumn(T* d_in,
                           size_t n_columns,
                           size_t n_rows,
                           size_t noColumn,
                           T value) {
  size_t rowNumber = threadIdx.x + blockDim.x * blockIdx.x;
  size_t index = noColumn + rowNumber * n_columns;

  if(index < n_columns * n_rows) {
    d_in[index] = value;
  }
}

void SetColumn(Tensor in, size_t col, float value) {
  int nRows = in->shape().elements() / in->shape()[-1];
  int nColumns = in->shape()[-1];

  int nBlocks = nRows / 512 + ((nRows % 512 == 0) ? 0 : 1);
  int nThreads = std::min(512, nRows);

 if(in->type() == Type::float32) {
   gSetColumn<<<nBlocks, nThreads>>>(in->data<float>(), nColumns, nRows, col, value);
#if COMPILE_FP16
 } else if(in->type() == Type::float16) {
   gSetColumn<<<nBlocks, nThreads>>>(in->data<half>(), nColumns, nRows, col, (half)value);
#endif
 } else {
   ABORT("suppressWord not implemented for type {}", in->type());
 }
}

void suppressWord(Expr probs, WordIndex wordIndex) {
  SetColumn(probs->val(), wordIndex, NumericLimits<float>(probs->value_type()).lowest);
}
}  // namespace gpu
}  // namespace marian
