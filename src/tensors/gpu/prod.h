#pragma once

#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"

namespace marian {
namespace gpu {

void BiasAdd(marian::Tensor C,
             const marian::Tensor& bias,
             bool do_relu = false);

void Affine(marian::Tensor C,
            Ptr<Allocator> allocator,
            const marian::Tensor& A,
            const marian::Tensor& B,
            const marian::Tensor& bias,
            bool transA,
            bool transB,
            float beta = 0,
            float scalar = 1,
            bool do_relu = false);

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar,
          Type computeType);

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta = 0,
          float scalar = 1);

void ProdBatched(marian::Tensor C,
                 Ptr<Allocator> allocator,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta = 0,
                 float scalar = 1);

void CSRProd(marian::Tensor C,
             Ptr<Allocator> allocator,
             const marian::Tensor& A_values,
             const marian::Tensor& A_indices,
             const marian::Tensor& A_offsets,
             const marian::Tensor& B,
             bool transA,
             bool swapOperands,
             float beta = 0);

}  // namespace gpu
}  // namespace marian
