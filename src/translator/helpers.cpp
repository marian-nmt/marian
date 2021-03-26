/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <limits>

#include "data/types.h"
#include "tensors/tensor.h"
#include "translator/helpers.h"

namespace marian {

namespace cpu {

void SetColumns(Tensor in, Tensor indices, float value) {
  int nRows = in->shape().elements() / in->shape()[-1];
  int nColumns = in->shape()[-1];
  int nSuppress = indices->shape()[-1];

  for(int rowNumber = 0; rowNumber < nRows; ++rowNumber) {
    float* row = in->data() + rowNumber * nColumns;
    for(int i = 0; i < nSuppress; ++i)
      row[indices->data<WordIndex>()[i]] = value;
  }
}

void suppressWords(Expr logProbs, Expr wordIndices) {
  SetColumns(logProbs->val(), wordIndices->val(), std::numeric_limits<float>::lowest());
}
}  // namespace cpu

void suppressWords(Expr logProbs, Expr wordIndices) {
  if(logProbs->val()->getBackend()->getDeviceId().type == DeviceType::cpu) {
    cpu::suppressWords(logProbs, wordIndices);
  }
#ifdef CUDA_FOUND
  else {
    gpu::suppressWords(logProbs, wordIndices);
  }
#endif
}
}  // namespace marian
