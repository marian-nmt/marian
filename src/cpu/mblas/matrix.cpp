#include "matrix.h"
#include "simd_math_prims.h"

#include "../blaze/Math.h"

namespace mblas {

//Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
//             bool transA, bool transB) {
//  Matrix::value_type alpha = 1.0;
//  Matrix::value_type beta = 0.0;
//  
//  size_t m = A.Rows();
//  size_t k = A.Cols();
//  if(transA)
//    std::swap(m, k);
//  
//  size_t l = B.Rows();
//  size_t n = B.Cols();
//  if(transB)
//    std::swap(l, n);
//  
//  size_t lda = A.Cols();
//  size_t ldb = B.Cols();
//  size_t ldc = B.Cols();
//  
//  if(transB)
//    ldc = B.Rows();
//  
//  C.Resize(m, n);
//  
//  auto opA = transA ? CblasTrans : CblasNoTrans;
//  auto opB = transB ? CblasTrans : CblasNoTrans;
//  
//  cblas_sgemm(CblasColMajor, opB, opA,
//              n, m, k, alpha, B.data(), ldb, A.data(), lda, beta, C.data(), ldc);
//  return C;
//}

}
