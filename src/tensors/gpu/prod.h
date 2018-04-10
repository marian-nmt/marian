#pragma once

#include "tensors/tensor.h"

namespace marian {

namespace gpu {

void Prod(marian::Tensor C,
          const marian::Tensor A,
          const marian::Tensor B,
          bool transA,
          bool transB,
          float beta = 0,
          float scalar = 1);

void ProdWithBias(marian::Tensor C,
          const marian::Tensor A,
          const marian::Tensor B,
          const marian::Tensor bias,
          bool transA,
          bool transB,
          float beta = 0,
          float scalar = 1) {
  Prod(C, A, B, transA, transB, beta, scalar);
  Add(_1, C, bias);
}

void ProdBatched(marian::Tensor C,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta = 0,
                 float scalar = 1);
}
}
