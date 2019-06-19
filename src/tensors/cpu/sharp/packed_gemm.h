#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {
namespace variant {

void PackFp32(marian::Tensor out,
              const marian::Tensor in,
              bool transpose,
              int nrow,
              int ncol,
              int kernel_ncol_blocks,
              int brow,
              int bcol,
              int last_brow,
              int nbrow,
              int nbcol,
              uint64_t packsize);

void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const int64_t m,
                  const int64_t n,
                  const int transA = 0);

void AddBias(marian::Tensor C, const marian::Tensor Bias);

}  // namespace variant
}  // namespace cpu
}  // namespace marian
