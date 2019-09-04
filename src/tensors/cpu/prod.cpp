/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "tensors/cpu/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"

#if MKL_FOUND
#include <mkl.h>
#else
#if BLAS_FOUND
#include <cblas.h>
#endif
#endif

#include "sharp/int_gemm.h"

namespace marian {

namespace cpu {

#if BLAS_FOUND
inline void sgemm(bool transA,
                  bool transB,
                  int rows_a,
                  int rows_b,
                  int width,
                  float alpha,
                  float* a,
                  int lda,
                  float* b,
                  int ldb,
                  float beta,
                  float* c,
                  int ldc) {
  cblas_sgemm(CblasRowMajor,
              transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans,
              rows_a,
              rows_b,
              width,
              alpha,
              a,
              lda,
              b,
              ldb,
              beta,
              c,
              ldc);
}
#endif

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
#if BLAS_FOUND
  float alpha = scalar;

  int m = A->shape().elements() / A->shape()[-1];
  int k = A->shape().back();
  if(transA)
    std::swap(m, k);

  int l = B->shape().elements() / B->shape()[-1];
  int n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  int lda = A->shape()[-1];
  int ldb = B->shape()[-1];
  int ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape().elements() / B->shape()[-1];

  sgemm(transA,
        transB,
        m,
        n,
        k,
        alpha,
        A->data(),
        lda,
        B->data(),
        ldb,
        beta,
        C->data(),
        ldc);
#else
  C; A; B; transA; transB; beta; scalar;
  ABORT("You need to compile with MKL in order to use the CPU version");
#endif
}

void ProdBatched(marian::Tensor C,
                 Ptr<Allocator> /*allocator*/,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta,
                 float scalar) {
#if BLAS_FOUND
  float alpha = scalar;

  size_t batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  size_t batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);

  size_t m = A->shape()[-2];
  size_t k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  size_t l = B->shape()[-2];
  size_t n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  auto strideB = batchB == 1 ? 0 : n * k;
  auto strideA = batchA == 1 ? 0 : m * k;
  auto strideC = n * m;

  auto batchC = std::max(batchA, batchB);
  for(size_t i = 0; i < batchC; ++i) {
    sgemm(transA,
          transB,
          (int)m,
          (int)n,
          (int)k,
          alpha,
          A->data() + (i % batchA) * strideA,
          (int)lda,
          B->data() + (i % batchB) * strideB,
          (int)ldb,
          beta,
          C->data() + i * strideC,
          (int)ldc);
  }
#else
  C; A; B; transA; transB; beta; scalar;
  ABORT("You need to compile with MKL in order to use the CPU version");
#endif
}

void ProdWithBias(marian::Tensor C,
                  const marian::Tensor& A,
                  const marian::Tensor& B,
                  const marian::Tensor& bias,
                  bool transA,
                  bool transB,
                  float beta,
                  float scalar) {
  cpu::Prod(C, A, B, transA, transB, beta, scalar);
  cpu::int16::AddBias(C, bias);
}

void CSRProd(marian::Tensor C,
             Ptr<Allocator> /*allocator*/,
             const marian::Tensor& S_values,
             const marian::Tensor& S_indices,
             const marian::Tensor& S_offsets,
             const marian::Tensor& D,
             bool transS,
             bool swapOperands,
             float beta) {
  C, S_values, S_indices, S_offsets, D;

  // Note: The CPU implementation currently only implements what's needed for decoding.

  // interpret tensor dimensions as matrix dimensions
  const auto& shapeC = C->shape();
  const auto& shapeD = D->shape();
  // If swapOperands, S and D are swapped (C = D x S instead of C = S x D).
  // In that case, in the next 6 lines, please read all dimensions as if they were reversed in order.
  auto rowsC = shapeC[-(int)swapOperands];
  auto colsC = shapeC.elements() / rowsC;
  auto rowsD = shapeD[-(int)swapOperands];
  auto colsD = shapeD.elements() / rowsD;
  auto rowsS = transS ? rowsD : rowsC;
  auto colsS = transS ? rowsC : rowsD;
  ABORT_IF(colsD != colsC, "Inconsistent outer dimensions in CSR product");
  if (swapOperands) { // make rowsX actual row dimensions again, likewise colsX
    std::swap(rowsC, colsC);
    std::swap(rowsD, colsD);
    std::swap(rowsS, colsS);
  }
  // sparse arrays
  auto numOffsets = S_offsets->shape().elements() - 1; // -1 since last value is length
  ABORT_IF(numOffsets != rowsS, "Unexpected number of rows in CSR argument"); numOffsets;
  ABORT_IF(S_values->shape() != S_indices->shape(), "CSR values and indices must have the same size");
  if (!transS && !swapOperands) {
    // C = S * D, where D = CSR matrix
    const auto* offsets = S_offsets->data<IndexType>();
    const auto* indices = S_indices->data<IndexType>();
    const auto* values  = S_values->data<float>();
    const auto* dataD   = D->data<float>();
    auto*       dataC   = C->data<float>();
    ABORT_IF(beta != 0 && beta != 1, "cpu::CSRProd only supports beta = 0 or 1");
    for (size_t i = 0; i < rowsC; i++) {
      auto add = (beta == 1); // first element: overwrite or add according to beta; subsequent elements: add
      for (size_t kk = offsets[i]; kk < offsets[i + 1]; kk++) {
        auto k = indices[kk];    // fetch the non-zero row
        auto valS = values[kk]; // and the value from that row
        // This code is written with the hope for good vectorization, and the hope
        // that adding to memory will be done efficiently by the caching system.
        if (valS == 1)
          if (!add)
            for (size_t j = 0; j < colsC; j++)
              dataC[i * colsC/*==colsD*/ + j] = dataD[k * colsC/*==colsD*/ + j]; // this is a memcpy()
          else
            for (size_t j = 0; j < colsC; j++)
              dataC[i * colsC/*==colsD*/ + j] += dataD[k * colsC/*==colsD*/ + j]; // this is a contiguous-vector addition
        else
          if (!add)
            for (size_t j = 0; j < colsC; j++)
              dataC[i * colsC/*==colsD*/ + j] = valS * dataD[k * colsC/*==colsD*/ + j];
          else
            for (size_t j = 0; j < colsC; j++)
              dataC[i * colsC/*==colsD*/ + j] += valS * dataD[k * colsC/*==colsD*/ + j]; // notice the +=
        add = true; // next iteration will add to existing result
      }
    }
  }
  else
    ABORT("CSRProd for transS={}, swapOperands={} is not yet implemented for CPU", transS, swapOperands);
}

}  // namespace cpu
}  // namespace marian
