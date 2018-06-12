/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "tensors/cpu/backend.h"
#include "tensors/tensor.h"

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
inline void sgemm(bool transA, bool transB,
                  int rows_a, int rows_b, int width,
                  float alpha,
                  float *a,
                  int lda,
                  float *b,
                  int ldb,
                  float beta,
                  float *c,
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
          const marian::Tensor A,
          const marian::Tensor B,
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
  ABORT("Not implemented!");
#endif
}

void ProdBatched(marian::Tensor C,
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

  int steps = std::max(batchA, batchB);

  int offsetA = 0;
  int offsetB = 0;
  int offsetC = 0;

  for(int i = 0; i < steps; ++i) {
    sgemm(transA,
          transB,
          m,
          n,
          k,
          alpha,
          A->data() + offsetA,
          lda,
          B->data() + offsetB,
          ldb,
          beta,
          C->data() + offsetC,
          ldc);

    offsetA += strideA;
    offsetB += strideB;
    offsetC += strideC;
  }
#else
  ABORT("Not implemented!");
#endif
}

void ProdWithBias(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  bool transA,
                  bool transB,
                  float beta,
                  float scalar) {
  cpu::Prod(C, A, B, transA, transB, beta, scalar);
  cpu::int16::AddBias(C, bias);
}

}
}
