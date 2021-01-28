#include <cublas_v2.h>
#include <cusparse.h>

// clang-format off
#include "tensors/gpu/prod.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

namespace marian {
namespace gpu {

// primary template for specialization with different element and compute types
template <typename ElementType>
struct TypedSparseGemm { };

template <>
struct TypedSparseGemm</*ElementType=*/float> { // specialization for element type float32 and compute type float32

// bug in cuSparse: sparse matrix is limited to 65535 columns
// This function is a drop-in replacement that handles it (by slicing).
cusparseStatus_t
static cusparseSgemmiEx(cusparseHandle_t handle, int m,
  int n, // the offending number of columns of matrices B and C
  int k, int nnz, const float *alpha, const float *A, int lda,
  const float *cscValB, const int *cscColPtrB, const int *cscRowIndB, const float *beta,
  float *C, int ldc)
{
  const int nMax = 65535; // max. number of columns allowed by cuSparse 10 implementation
  for (int j0 = 0; j0 < n; j0 += 65535) { // loop over column slices, j0 = index of first column
    // Call original function on a column slice.
    // Replace all parameters that relate to the column slice.
    // nnz does not need to be corrected.
    auto n1 = std::min(n - j0, nMax);   // width of column slice is limited to max
    auto C1 = C + j0 * ldc;             // column slice into result matrix C
    auto cscColPtrB1 = cscColPtrB + j0; // column slice into sparse factor B
    auto rc = cusparseSgemmi(handle, m, n1, k, nnz, alpha, A, lda, cscValB, cscColPtrB1, cscRowIndB, beta, C1, ldc);
    if (rc != CUSPARSE_STATUS_SUCCESS)
      return rc;
  }
  return CUSPARSE_STATUS_SUCCESS;
}

// C = op(S) x D if not swapOperands else C = D x op(S)
// op(S) = S if not transA else S^T
static void CSRProd(marian::Tensor C,
                    Ptr<Allocator> allocator,
                    const marian::Tensor& S_values,
                    const marian::Tensor& S_indices,
                    const marian::Tensor& S_offsets,
                    const marian::Tensor& D,
                    bool transS,
                    bool swapOperands,
                    float beta) {
  cudaSetDevice((int)C->getDeviceId().no);
  auto cusparseHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())->getCusparseHandle();
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
  auto numValues  = S_values->shape().elements();
  auto numOffsets = S_offsets->shape().elements() - 1; // -1 since last value is length
  ABORT_IF(numOffsets != rowsS, "Unexpected number of rows in CSR argument");
  ABORT_IF(S_values->shape() != S_indices->shape(), "CSR values and indices must have the same size");
  float alpha = 1;
  MemoryPiece::PtrType St_values, St_indices, St_offsets;
  if (transS != swapOperands) {
    // Cusparse gemmi() does not support this specific version of transpose, and csrmm() is non-deterministic.
    // Hence, we transpose the matrix explicitly.
    // Note that gemmi() expects a CSC, while csrmm() a CSR; hence, the strange condition (transS != swapOperands) above.
    St_values  = allocator->alloc<float>(numValues);
    St_indices = allocator->alloc<int>(numValues);
    St_offsets = allocator->alloc<int>(colsS + 1);
    // transpose the second argument
    CUSPARSE_CHECK(cusparseScsr2csc(cusparseHandle,
        /*m=*/ rowsS, // number of rows of matrix
        /*n=*/ colsS, // number of columns of matrix
        /*nnz=*/ (int)numValues,
        /*csrcVal=*/    S_values ->data<float>(),
        /*csrcRowPtr=*/ (int*)S_offsets->data<IndexType>(),
        /*csrcColInd=*/ (int*)S_indices->data<IndexType>(),
        /*cscVal=*/    St_values ->data<float>(),  // transposed version goes here
        /*cscRowInd=*/ St_indices->data<int>(),
        /*cscColPtr=*/ St_offsets->data<int>(),
        /*copyValues=*/ CUSPARSE_ACTION_NUMERIC,
        /*idxBase=*/ CUSPARSE_INDEX_BASE_ZERO));
    std::swap(rowsS, colsS); // these variables now represent the dims of the explicitly transposed object
  }
  if (swapOperands) {
    // C = D x S for row-major matrices
    // Implemented via cusparse as C' = S' x D' ("csrmm") where C' and D' are column-major,
    // and S' is CSR (if not transS then we make a transposed copy).
    cusparseMatDescr_t descrA;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
    cusparseSetMatType     (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    CUSPARSE_CHECK(cusparseScsrmm(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, // (we explicitly transposed above)
        /*m=*/ rowsS, // #rows of first (CSR) factor (the transpose was done explicitly)
        /*n=*/ rowsC, // #cols of second (col-major) factor and (col-major) result = #rows of row-major C
        /*k=*/ colsS, // #cols of first (CSR) factor
        /*nnz=*/ (int)numValues,
        &alpha, descrA,
        /*csrValA=*/    St_values  ? St_values ->data<float>() :       S_values ->data<float>(),
        /*csrRowPtrA=*/ St_offsets ? St_offsets->data<int>()   : (int*)S_offsets->data<IndexType>(),
        /*csrColIndA=*/ St_indices ? St_indices->data<int>()   : (int*)S_indices->data<IndexType>(),
        D->data(),
        /*ldb=*/ colsD, // stride
        &beta,
        C->data(),
        /*ldc=*/ colsC)); // stride
    cusparseDestroyMatDescr(descrA);
  }
  else {
    // C = S x D for row-major matrices
    // Implemented via cusparse as C' = D' x S' ("gemmi") where C' and D' are column-major.
    CUSPARSE_CHECK(cusparseSgemmiEx(cusparseHandle,
        /*m=*/ colsD, // #rows of first (col-major) factor = #cols of row-major D
        /*n=*/ rowsC, // #cols of second (CSC) factor and (col-major) result = #rows of row-major C
        /*k=*/ rowsD, // #cols of first (col-major) factor = #rows of row-major D
        /*nnz=*/ (int)numValues,
        &alpha,
        /*A=*/ D->data(),
        /*lda=*/ colsD, // stride
        /*cscValB=*/    St_values  ? St_values ->data<float>() :       S_values ->data<float>(),
        /*cscColPtrB=*/ St_offsets ? St_offsets->data<int>()   : (int*)S_offsets->data<IndexType>(),
        /*cscRowIndB=*/ St_indices ? St_indices->data<int>()   : (int*)S_indices->data<IndexType>(),
        &beta,
        C->data(),
        /*ldc=*/ colsC)); // stride
    // Note: cuSparse 10 docs says this about cscColPtrB:
    //   "integer array of k + 1 elements that contains the start of every row and the end of the last row plus one."
    // This is wrong. It should be col instead of row, and n instead of k.
  }
  if(St_values ) allocator->free(St_values );
  if(St_indices) allocator->free(St_indices);
  if(St_offsets) allocator->free(St_offsets);
}

};

template <>
struct TypedSparseGemm</*ElementType=*/half> { // specialization for element type float32 and compute type float16
  static void CSRProd(marian::Tensor /*C*/,
                      Ptr<Allocator> /*allocator*/,
                      const marian::Tensor& /*S_values*/,
                      const marian::Tensor& /*S_indices*/,
                      const marian::Tensor& /*S_offsets*/,
                      const marian::Tensor& /*D*/,
                      bool /*transS*/,
                      bool /*swapOperands*/,
                      float /*beta*/) {
    ABORT("CSRProd with type {} not supported on CUDA 10.2 or lower. You need at least CUDA 11.0", Type::float16);
  }
};


}  // namespace gpu
}  // namespace marian
