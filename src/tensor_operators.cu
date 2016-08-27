#include "tensor_operators.h"

namespace marian {

Tensor Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta) {
  Float alpha = 1.0;

  size_t m = A.shape()[0];
  size_t k = A.shape()[1];
  if(transA)
    std::swap(m, k);
  
  size_t l = B.shape()[0];
  size_t n = B.shape()[1];
  if(transB)
    std::swap(l, n);
  
  size_t lda = A.shape()[1];
  size_t ldb = B.shape()[1];
  size_t ldc = B.shape()[1];
  
  if(transB)
    ldc = B.shape()[0];
  
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  cublasSgemm(handle, opB, opA,
              n, m, k, &alpha, B.data(), ldb, A.data(), lda, &beta, C.data(), ldc);
  return C;
}

Tensor Prod(Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta) {

  return Prod(handles.cublasHandle, C, A, B, transA, transB, beta);
}

}