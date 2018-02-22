#include "tensors/gpu/prod.h"
#include "tensors/gpu/backend.h"

#if BLAS_FOUND
#include <cblas.h>
#endif

namespace marian {

namespace cpu {

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

  cblas_sgemm(
        CblasColMajor,
        transB ? CblasTrans : CblasNoTrans,
        transA ? CblasTrans : CblasNoTrans,
        n, m, k,
        alpha,
        B->data(),
        ldb,
        A->data(),
        lda,
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
  ABORT("Not implemented!");
}

}
}
