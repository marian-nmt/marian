#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {
namespace variant { // Variants of GEMM implementations

void PackFp32(marian::Tensor out,
              const marian::Tensor in,
              const bool transpose,
              const int nrow,
              const int ncol,
              const int kernel_ncol_blocks,
              const int brow,
              const int bcol,
              const int last_brow,
              const int nbrow,
              const int nbcol,
              const uint64_t packsize);

void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const size_t m,
                  const size_t n,
                  const int transA = 0);

}  // namespace variant
}  // namespace cpu
}  // namespace marian
