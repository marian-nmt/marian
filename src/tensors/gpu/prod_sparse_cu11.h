#ifdef _MSC_VER
#pragma warning(disable: 4505) // warning C4505: '__float2half_rz': unreferenced local function has been removed (missing 'static inline')
#endif

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
struct TypedSparseGemm { 

static cudaDataType getCudaDataType(const float*) { return CUDA_R_32F; };
static cudaDataType getCudaDataType(const half*)  { return CUDA_R_16F; };

#if 0
static void CSRProdSwapped(marian::Tensor C,
                           Ptr<Allocator> allocator,
                           const marian::Tensor& S_values,
                           const marian::Tensor& S_indices,
                           const marian::Tensor& S_offsets,
                           const marian::Tensor& D,
                           bool transS,
                           ElementType beta) {
  cudaSetDevice((int)C->getDeviceId().no);
  auto cusparseHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())->getCusparseHandle();

  // interpret tensor dimensions as matrix dimensions
  const auto& shapeC = C->shape();
  const auto& shapeD = D->shape();
  
  auto colsC = shapeC[-1];
  auto rowsC = shapeC.elements() / colsC;

  auto colsD = shapeD[-1];
  auto rowsD = shapeD.elements() / colsD;

  auto rowsS = rowsC;
  auto colsS = rowsD;

  auto denseOrder = CUSPARSE_ORDER_COL;
  auto algorithm  = CUSPARSE_SPMM_ALG_DEFAULT; 

  std::cerr << shapeC << std::endl;
  std::cerr << shapeD << std::endl;

  if(transS)
    std::swap(rowsS, colsS);

  // sparse arrays
  auto numValues  = S_values->shape().elements();
  auto numOffsets = S_offsets->shape().elements() - 1; // -1 since last value is length
  ABORT_IF(numOffsets != rowsS, "Unexpected number of rows in CSR argument");
  ABORT_IF(S_values->shape() != S_indices->shape(), "CSR values and indices must have the same size");
  
  ElementType alpha = 1.0;

  cusparseSpMatDescr_t descS;
  cusparseDnMatDescr_t descD;
  cusparseDnMatDescr_t descC;
  CUSPARSE_CHECK(cusparseCreateCsr(&descS,
                                   rowsS, colsS, numValues,
                                   S_offsets->data<IndexType>(),
                                   S_indices->data<IndexType>(),
                                   S_values ->data<ElementType>(),
                                   CUSPARSE_INDEX_32I, 
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   getCudaDataType(S_values->data<ElementType>())));
  CUSPARSE_CHECK(cusparseCreateDnMat(&descD,
                                     rowsD, colsD, /*ld=*/colsD, 
                                     D->data<ElementType>(), 
                                     getCudaDataType(D->data<ElementType>()), 
                                     denseOrder));
  CUSPARSE_CHECK(cusparseCreateDnMat(&descC,
                                     rowsC, colsC, /*ld=*/colsC,
                                     C->data<ElementType>(), 
                                     getCudaDataType(C->data<ElementType>()), 
                                     denseOrder));

  size_t bufferSize = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparseHandle,
                                         transS ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         descS,
                                         descD,
                                         &beta,
                                         descC,
                                         getCudaDataType(C->data<ElementType>()),
                                         algorithm,
                                         &bufferSize));
  if(bufferSize > 0) {
    MemoryPiece::PtrType buffer = allocator->alloc<uint8_t>(bufferSize);
    CUSPARSE_CHECK(cusparseSpMM(cusparseHandle,
                                transS ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                descS,
                                descD,
                                &beta,
                                descC,
                                getCudaDataType(C->data<ElementType>()),
                                algorithm,
                                buffer->data<uint8_t>()));
    allocator->free(buffer);
  }

  CUSPARSE_CHECK(cusparseDestroySpMat(descS));
  CUSPARSE_CHECK(cusparseDestroyDnMat(descD));
  CUSPARSE_CHECK(cusparseDestroyDnMat(descC));
}
#endif

// C = op(S) x D if not swapOperands else C = D x op(S)
// op(S) = S if not transA else S^T
static void CSRProd(marian::Tensor C,
                    Ptr<Allocator> allocator,
                    const marian::Tensor& S_values,
                    const marian::Tensor& S_indices,
                    const marian::Tensor& S_offsets,
                    const marian::Tensor& D,
                    bool transS,
                    ElementType beta) {
  cudaSetDevice((int)C->getDeviceId().no);
  auto cusparseHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())->getCusparseHandle();

  // interpret tensor dimensions as matrix dimensions
  const auto& shapeC = C->shape();
  const auto& shapeD = D->shape();
  
  auto colsC = shapeC[-1];
  auto rowsC = shapeC.elements() / colsC;

  auto colsD = shapeD[-1];
  auto rowsD = shapeD.elements() / colsD;

  auto rowsS = rowsC;
  auto colsS = rowsD;

  auto denseOrder = CUSPARSE_ORDER_ROW;
  auto algorithm  = CUSPARSE_SPMM_CSR_ALG2; 

  if(transS)
    std::swap(rowsS, colsS);

  // sparse arrays
  auto numValues  = S_values->shape().elements();
  auto numOffsets = S_offsets->shape().elements() - 1; // -1 since last value is length
  ABORT_IF(numOffsets != rowsS, "Unexpected number of rows in CSR argument");
  ABORT_IF(S_values->shape() != S_indices->shape(), "CSR values and indices must have the same size");
  
  ElementType alpha = 1.0;

  cusparseSpMatDescr_t descS;
  cusparseDnMatDescr_t descD;
  cusparseDnMatDescr_t descC;
  CUSPARSE_CHECK(cusparseCreateCsr(&descS,
                                   rowsS, colsS, numValues,
                                   S_offsets->data<IndexType>(),
                                   S_indices->data<IndexType>(),
                                   S_values ->data<ElementType>(),
                                   CUSPARSE_INDEX_32I, 
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   getCudaDataType(S_values->data<ElementType>())));
  CUSPARSE_CHECK(cusparseCreateDnMat(&descD,
                                     rowsD, colsD, /*ld=*/colsD, 
                                     D->data<ElementType>(), 
                                     getCudaDataType(D->data<ElementType>()), 
                                     denseOrder));
  CUSPARSE_CHECK(cusparseCreateDnMat(&descC,
                                     rowsC, colsC, /*ld=*/colsC,
                                     C->data<ElementType>(), 
                                     getCudaDataType(C->data<ElementType>()), 
                                     denseOrder));

  size_t bufferSize = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparseHandle,
                                         transS ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         descS,
                                         descD,
                                         &beta,
                                         descC,
                                         getCudaDataType(C->data<ElementType>()),
                                         algorithm,
                                         &bufferSize));
  if(bufferSize > 0) {
    MemoryPiece::PtrType buffer = allocator->alloc<uint8_t>(bufferSize);
    CUSPARSE_CHECK(cusparseSpMM(cusparseHandle,
                                transS ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                descS,
                                descD,
                                &beta,
                                descC,
                                getCudaDataType(C->data<ElementType>()),
                                algorithm,
                                buffer->data<uint8_t>()));
    allocator->free(buffer);
  }

  CUSPARSE_CHECK(cusparseDestroySpMat(descS));
  CUSPARSE_CHECK(cusparseDestroyDnMat(descD));
  CUSPARSE_CHECK(cusparseDestroyDnMat(descC));
}

static void CSRProd(marian::Tensor C,
                    Ptr<Allocator> allocator,
                    const marian::Tensor& S_values,
                    const marian::Tensor& S_indices,
                    const marian::Tensor& S_offsets,
                    const marian::Tensor& D,
                    bool transS,
                    bool swapOperands,
                    ElementType beta) {
  if(swapOperands) {
    ABORT("Not implemented");
    // CSRProdSwapped(C, allocator, S_values, S_indices, S_offsets, D, transS, beta);
  }
  else {
    CSRProd(C, allocator, S_values, S_indices, S_offsets, D, transS, beta);
  }
}

};

}  // namespace gpu
}  // namespace marian
