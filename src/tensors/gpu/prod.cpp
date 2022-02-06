
#ifdef _MSC_VER
#pragma warning(disable: 4505) // warning C4505: '__float2half_rz': unreferenced local function has been removed (missing 'static inline')
#endif

#include <cublas_v2.h>

// clang-format off
#include "tensors/gpu/prod.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

#if CUDA_VERSION >= 11000
#include <cublasLt.h>
#endif

namespace marian {

namespace gpu {

// It seems that the bias must be 8 byte aligned for the cublasLt epilogue to work. Therefore,
// if the bias pointer is not 8 byte aligned, we do a normal matmul in cublasLt and invoke a 
// custom epilogue kernel.
static constexpr int REQUIRED_BIAS_ALIGNMENT = 16; // @TODO: MJD: changed this to 16 to avoid alignment error on A100. Seems to work fine.

// Used to set preferences for cublasLt to filter out algos if matrices to not meet default 256 byte alignment
int getAlignmentUpTo256(const void *ptr) {
  uintptr_t addr = (uintptr_t)ptr;
  int trailingZeros = 0;

  for(int shiftAmt = 8, mask = 0xFF; shiftAmt > 0; shiftAmt /= 2, mask >>=shiftAmt) {
    if ((addr & mask) == 0) {
      trailingZeros += shiftAmt;
      addr >>= shiftAmt;
    }
  }

  return std::min(256, 1 << trailingZeros);
}

// The explicit version of matmult like cublasGemmEx choose their math mode based on the algorithm that
// has been passed into the function call and seem to ignore setMathMode. Here we query the used math mode
// to choose the algorithm.
static bool tensorOpsEnabled(cublasHandle_t cublasHandle) {
#if CUDA_VERSION >= 9000
  cublasMath_t actual = CUBLAS_DEFAULT_MATH;
  cublasGetMathMode(cublasHandle, &actual);
  return actual == CUBLAS_TENSOR_OP_MATH;
#else
  return false;
#endif
}

static void setTensorMode(cublasHandle_t cublasHandle) {
  cublasHandle; // fool warnings
#if CUDA_VERSION >= 9000
  static int mode = 0;  // 1: use TC; -1: do not use TC; 0: not set yet
  if (mode == 0) { // multi-thread note: this is sort-of thread-safe, since multiple threads would determine the same value
    const char* var = getenv("ENABLE_CUBLAS_TENSOR_OP_MATH_FP32");
    if (!var)
      var = "1";
    switch(var[0]) {
      case '0': mode = -1; break;
      case '1': mode =  1; break;
      default: ABORT("Invalid ENABLE_CUBLAS_TENSOR_OP_MATH_FP32={}", var);
    }
    if (mode > 0) { // try whether it can be set   --@TODO: check whether this actually works
      CUBLAS_CHECK(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
      cublasMath_t actual = CUBLAS_DEFAULT_MATH;
      cublasGetMathMode(cublasHandle, &actual);
      if (actual != CUBLAS_TENSOR_OP_MATH) {
        LOG(warn, "[gpu] TensorCores requested but not available");
        mode = -1;
      }
    }
    if (mode > 0)
      LOG(info, "[gpu] 16-bit TensorCores enabled for float32 matrix operations");
  }
  CUBLAS_CHECK(cublasSetMathMode(cublasHandle, mode > 0 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
#endif
}

static void unsetTensorMode(cublasHandle_t cublasHandle) {
  cublasHandle; // fool warnings
#if CUDA_VERSION >= 9000
  CUBLAS_CHECK(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));
#endif
}

// primary template for specialization with different element and compute types
template <typename ElementType, typename ComputeType>
struct TypedGemm { };

template <>
struct TypedGemm</*ElementType=*/float, /*ComputeType=*/float> { // specialization for element type float32 and compute type float32
  static void gemm(cublasHandle_t handle,
                   CudaCompute computeCapability,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m, int n, int k,
                   const float* alpha, // has to match compute type!
                   const float* A, int lda,
                   const float* B, int ldb,
                   const float* beta,  // has to match compute type!
                   float* C, int ldc) {
  // double #if and if unfortunately required to safeguard against compilation error 
  // with CUDA 8.0 and runtime error with CUDA >9.0 on GPUs with compute capability under 5
  #if CUDA_VERSION > 9000
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    if(computeCapability.major >= 5)
      CUBLAS_CHECK(cublasGemmEx(handle, transa, transb,
                                m, n, k, alpha,
                                A, CUDA_R_32F, lda,
                                B, CUDA_R_32F, ldb, beta,
                                C, CUDA_R_32F, ldc,
                                CUDA_R_32F, algorithm));
    else // don't lose the "else"
  #endif
      CUBLAS_CHECK(cublasSgemm(handle, transa, transb,
                               m, n, k, alpha,
                               A, lda,
                               B, ldb, beta,
                               C, ldc));
  
  }

  static void batchedGemm(cublasHandle_t handle,
                          CudaCompute computeCapability,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m, int n, int k,
                          const float *alpha, // has to match compute type!
                          const float *Aarray[], int lda,
                          const float *Barray[], int ldb,
                          const float *beta,  // has to match compute type!
                          float *Carray[], int ldc,
                          int batchCount) {
  // double #if and if unfortunately required to safeguard against compilation error
  // with CUDA 8.0 and runtime error with CUDA >9.0 on GPUs with compute capability under 5
  #if CUDA_VERSION > 9000
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    if(computeCapability.major >= 5)
      CUBLAS_CHECK(cublasGemmBatchedEx(handle, transa, transb,
                                       m, n, k, alpha,
                                       (void* const*)Aarray, CUDA_R_32F, lda,
                                       (void* const*)Barray, CUDA_R_32F, ldb, beta,
                                       (void**)Carray, CUDA_R_32F, ldc, batchCount,
                                       CUDA_R_32F, algorithm));
    else // don't lose the "else"
  #endif
      CUBLAS_CHECK(cublasSgemmBatched(handle, transa, transb,
                                      m, n, k, alpha,
                                      Aarray, lda,
                                      Barray, ldb, beta,
                                      Carray, ldc, batchCount));
  }
};

#if COMPILE_FP16
template <>
struct TypedGemm</*ElementType=*/half, /*ComputeType=*/half> { // specialization for element type float16 and compute type float16
  // overload for half, contains configuration settings for float16
  static void gemm(cublasHandle_t handle,
                   CudaCompute computeCapability,
                   cublasOperation_t transa, 
                   cublasOperation_t transb,
                   int m, int n, int k,
                   const half* alpha,  // has to match compute type!
                   const half* A, int lda,
                   const half* B, int ldb,
                   const half* beta,  // has to match compute type!
                   half* C, int ldc) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmEx(handle, transa, transb,
                              m, n, k, alpha,
                              A, CUDA_R_16F, lda,
                              B, CUDA_R_16F, ldb, beta,
                              C, CUDA_R_16F, ldc,
                              CUDA_R_16F, algorithm)); // @TODO: review algorithm
  }

  static void batchedGemm(cublasHandle_t handle,
                          CudaCompute computeCapability,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m, int n, int k,
                          const half *alpha,  // has to match compute type!
                          const half *Aarray[], int lda,
                          const half *Barray[], int ldb,
                          const half *beta,   // has to match compute type!
                          half *Carray[], int ldc,
                          int batchCount) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmBatchedEx(handle, transa, transb,
                                     m, n, k, alpha,
                                     (void* const*)Aarray, CUDA_R_16F, lda,
                                     (void* const*)Barray, CUDA_R_16F, ldb, beta,
                                     (void**)Carray, CUDA_R_16F, ldc, batchCount,
                                     CUDA_R_16F, algorithm));
  }
};

template <>
struct TypedGemm</*ElementType=*/half, /*ComputeType=*/float> { // specialization for element type float16 and compute type float32
// overload for half, contains configuration settings for float16 and accumulation in float32
  static void gemm(cublasHandle_t handle,
                   CudaCompute computeCapability,
                   cublasOperation_t transa, 
                   cublasOperation_t transb,
                   int m, int n, int k,
                   const float* alpha, // has to match compute type!
                   const half* A, int lda,
                   const half* B, int ldb,
                   const float* beta, // has to match compute type!
                   half* C, int ldc) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmEx(handle, transa, transb, 
                              m, n, k, alpha,
                              A, CUDA_R_16F, lda,
                              B, CUDA_R_16F, ldb, beta,
                              C, CUDA_R_16F, ldc,
                              CUDA_R_32F, algorithm)); // use 32-bit compute type for accumulation
  }

  static void batchedGemm(cublasHandle_t handle,
                          CudaCompute computeCapability,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m, int n, int k,
                          const float *alpha, // has to match compute type!
                          const half *Aarray[], int lda,
                          const half *Barray[], int ldb,
                          const float *beta,  // has to match compute type!
                          half *Carray[], int ldc,
                          int batchCount) {
    ABORT_IF(computeCapability.major < 6, "Compute capability {} below 6 should not happen for FP16", computeCapability.major);
    // query math mode and set algorithm accordingly
    auto algorithm = tensorOpsEnabled(handle) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    CUBLAS_CHECK(cublasGemmBatchedEx(handle, transa, transb,
                                     m, n, k, alpha,
                                     (void* const*)Aarray, CUDA_R_16F, lda,
                                     (void* const*)Barray, CUDA_R_16F, ldb, beta,
                                     (void**)Carray, CUDA_R_16F, ldc, batchCount,
                                     CUDA_R_32F, algorithm)); // use 32-bit compute type for accumulation
  }
};
#endif


// overload for float, contains configuration settings for float32
template <typename ElementType, typename ComputeType>
void ProdTyped(marian::Tensor C,
               const marian::Tensor& A,
               const marian::Tensor& B,
               bool transA,
               bool transB,
               ComputeType beta,
               ComputeType scalar) {
  CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
  ComputeType alpha = scalar;

  int m = A->shape().elements() / A->shape().back();
  int k = A->shape().back();
  if(transA)
    std::swap(m, k);

  int l = B->shape().elements() / B->shape().back();
  int n = B->shape().back();
  if(transB)
    std::swap(l, n);

  int lda = A->shape().back();
  int ldb = B->shape().back();
  int ldc = B->shape().back();

  if(transB)
    ldc = B->shape().elements() / B->shape().back();

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
  auto cublasHandle = backend->getCublasHandle();
  auto computeCapability = backend->getCudaComputeCapability();

  setTensorMode(cublasHandle);
  TypedGemm<ElementType, ComputeType>::gemm(cublasHandle, computeCapability,
                                            opB, opA,
                                            n, m, k,
                                            &alpha,
                                            B->data<ElementType>(), ldb,
                                            A->data<ElementType>(), lda,
                                            &beta,
                                            C->data<ElementType>(), ldc);
  unsetTensorMode(cublasHandle);
}

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
  gpu::Prod(C, A, B, transA, transB, beta, scalar, C->type());
}

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar,
          Type computeType) {
  if(C->type() == Type::float32 && computeType == Type::float32) {
    ProdTyped</*ElementType=*/float, /*ComputeType=*/float>(C, A, B, transA, transB, beta, scalar);
#if COMPILE_FP16
  } else if(C->type() == Type::float16 && computeType == Type::float16) {
    ProdTyped</*ElementType=*/half, /*ComputeType=*/half>(C, A, B, transA, transB, __float2half(beta), __float2half(scalar));
  } else if(C->type() == Type::float16 && computeType == Type::float32) {
    ProdTyped</*ElementType=*/half, /*ComputeType=*/float>(C, A, B, transA, transB, beta, scalar);
#endif
  } else {
    ABORT("Prod not implemented for element type {} and compute type {}", C->type(), computeType);
  }
}

template <typename ElementType, typename ComputeType>
void ProdBatchedTyped(marian::Tensor C,                 
                      Ptr<Allocator> allocator,
                      const marian::Tensor A,
                      const marian::Tensor B,
                      bool transA,
                      bool transB,
                      ComputeType beta,
                      ComputeType scalar) {
  CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
  ComputeType alpha = scalar;

  // determine meta-shape of bdot operation. Essentially treat the last two dimensions as single elements
  // such that (..., m, k) x (..., k, n) -> (..., m, n) where ... is a broadcastable shape as in element-wise kernels.

  auto aShape = A->shape();
  auto bShape = B->shape();

  // make sure both shape have the same number of dimensions via broadcasting
  size_t maxLength = std::max(aShape.size(), bShape.size());
  if(aShape.size() != bShape.size()) {
    Shape ones(std::vector<int>(maxLength, 1));
    aShape = Shape::broadcast({aShape, ones});
    bShape = Shape::broadcast({bShape, ones});
  }

  // Create meta-shapes without last 2 dimensions
  Shape aShapeMeta, bShapeMeta, cShapeMeta;
  aShapeMeta.resize(maxLength - 2);
  bShapeMeta.resize(maxLength - 2);
  for(size_t i = 0; i < maxLength - 2; ++i) {
    aShapeMeta.set(i, aShape[i]);
    bShapeMeta.set(i, bShape[i]);
  }
  cShapeMeta = Shape::broadcast({aShapeMeta, bShapeMeta});

  int m = aShape[-2];
  int k = aShape[-1];
  if(transA)
    std::swap(m, k);

  int l = bShape[-2];
  int n = bShape[-1];
  if(transB)
    std::swap(l, n);

  int lda = aShape[-1];
  int ldb = bShape[-1];
  int ldc = bShape[-1];

  if(transB)
    ldc = bShape[-2];

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
  auto cublasHandle = backend->getCublasHandle();
  auto compute = backend->getCudaComputeCapability();

  int strideA = m * k;
  int strideB = n * k;
  int strideC = n * m;

  int batchC = cShapeMeta.elements();

  // Convert to functional shapes to be able to map dimensions. @TODO merge this
  functional::Shape aShapeMetaF = aShapeMeta;
  functional::Shape bShapeMetaF = bShapeMeta;
  functional::Shape cShapeMetaF = cShapeMeta;

  std::vector<const ElementType*> aptr;
  std::vector<const ElementType*> bptr;
  std::vector<ElementType*> cptr;

  functional::Array<int, functional::Shape::size()> dims;
  for(int i = 0; i < batchC; i++) {
    cShapeMetaF.dims(i, dims);
    auto aIndex = aShapeMetaF.bindex(dims);
    auto bIndex = bShapeMetaF.bindex(dims);

    aptr.push_back(A->data<ElementType>() + aIndex * strideA);
    bptr.push_back(B->data<ElementType>() + bIndex * strideB);
    cptr.push_back(C->data<ElementType>() + i * strideC);
  }

  // auto fails here from weird reason
  IPtr<MemoryPiece> mp_aptr = allocator->alloc<const ElementType*>(aptr.size());
  CudaCopy(aptr.data(), aptr.data() + aptr.size(), mp_aptr->data<const ElementType*>());

  IPtr<MemoryPiece> mp_bptr = allocator->alloc<const ElementType*>(bptr.size());
  CudaCopy(bptr.data(), bptr.data() + bptr.size(), mp_bptr->data<const ElementType*>());

  IPtr<MemoryPiece> mp_cptr = allocator->alloc<ElementType*>(cptr.size());
  CudaCopy(cptr.data(), cptr.data() + cptr.size(), mp_cptr->data<ElementType*>());

  setTensorMode(cublasHandle);
  TypedGemm<ElementType, ComputeType>::batchedGemm(cublasHandle, compute,
                                                   opB, opA,
                                                   n, m, k,
                                                   &alpha,
                                                   mp_bptr->data<const ElementType*>(), ldb,
                                                   mp_aptr->data<const ElementType*>(), lda,
                                                   &beta,
                                                   mp_cptr->data<ElementType*>(), ldc,
                                                   batchC);
  unsetTensorMode(cublasHandle);

  allocator->free(mp_aptr);
  allocator->free(mp_bptr);
  allocator->free(mp_cptr);
}

// @TODO: add version with compute type for completeness
void ProdBatched(marian::Tensor C,
                 Ptr<Allocator> allocator,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta,
                 float scalar) {
  if(C->type() == Type::float32) {
    ProdBatchedTyped<float, float>(C, allocator, A, B, transA, transB, beta, scalar);
#if COMPILE_FP16
  } else if(C->type() == Type::float16) { // not a *.cu file
    ProdBatchedTyped<half, half>(C, allocator, A, B, transA, transB, __float2half(beta), __float2half(scalar));
#endif
  } else {
    ABORT("ProdBatched not implemented for element type {}", C->type());
  }
}

template <typename ElementType, typename ComputeType>
void ProdBatchedTypedLegacy(marian::Tensor C,                 
                            Ptr<Allocator> allocator,
                            const marian::Tensor A,
                            const marian::Tensor B,
                            bool transA,
                            bool transB,
                            ComputeType beta,
                            ComputeType scalar) {
  CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
  ComputeType alpha = scalar;

  int batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  int batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);

  int m = A->shape()[-2];
  int k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  int l = B->shape()[-2];
  int n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  int lda = A->shape()[-1];
  int ldb = B->shape()[-1];
  int ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
  auto cublasHandle = backend->getCublasHandle();
  auto compute = backend->getCudaComputeCapability();

  auto strideA = batchA == 1 ? 0 : m * k;
  auto strideB = batchB == 1 ? 0 : n * k;
  auto strideC = n * m;
  auto batchC = std::max(batchA, batchB);

  std::vector<const ElementType*> aptr;
  std::vector<const ElementType*> bptr;
  std::vector<ElementType*> cptr;

  for(int i = 0; i < batchC; i++) {
    aptr.push_back(A->data<ElementType>() + (i % batchA) * strideA);
    bptr.push_back(B->data<ElementType>() + (i % batchB) * strideB);
    cptr.push_back(C->data<ElementType>() + i * strideC);
  }

  // auto fails here from weird reason
  IPtr<MemoryPiece> mp_aptr = allocator->alloc<const ElementType*>(aptr.size());
  CudaCopy(aptr.data(), aptr.data() + aptr.size(), mp_aptr->data<const ElementType*>());

  IPtr<MemoryPiece> mp_bptr = allocator->alloc<const ElementType*>(bptr.size());
  CudaCopy(bptr.data(), bptr.data() + bptr.size(), mp_bptr->data<const ElementType*>());

  IPtr<MemoryPiece> mp_cptr = allocator->alloc<ElementType*>(cptr.size());
  CudaCopy(cptr.data(), cptr.data() + cptr.size(), mp_cptr->data<ElementType*>());

  setTensorMode(cublasHandle);
  TypedGemm<ElementType, ComputeType>::batchedGemm(cublasHandle, compute,
                                                   opB, opA,
                                                   n, m, k,
                                                   &alpha,
                                                   mp_bptr->data<const ElementType*>(), ldb,
                                                   mp_aptr->data<const ElementType*>(), lda,
                                                   &beta,
                                                   mp_cptr->data<ElementType*>(), ldc,
                                                   batchC);
  unsetTensorMode(cublasHandle);

  allocator->free(mp_aptr);
  allocator->free(mp_bptr);
  allocator->free(mp_cptr);
}

// @TODO: add version with compute type for completeness
void ProdBatchedLegacy(marian::Tensor C,
                       Ptr<Allocator> allocator,
                       const marian::Tensor A,
                       const marian::Tensor B,
                       bool transA,
                       bool transB,
                       float beta,
                       float scalar) {
  if(C->type() == Type::float32) {
    ProdBatchedTypedLegacy<float, float>(C, allocator, A, B, transA, transB, beta, scalar);
#if COMPILE_FP16
  } else if(C->type() == Type::float16) { // not a *.cu file
    // we use computeType=float here for fp16 training as this seems more stable and roughly as fast
    ProdBatchedTypedLegacy<half, float>(C, allocator, A, B, transA, transB, beta, scalar);

    // original for reference:
    // ProdBatchedTypedLegacy<half, half>(C, allocator, A, B, transA, transB, __float2half(beta), __float2half(scalar));
#endif
  } else {
    ABORT("ProdBatchedLegacy not implemented for element type {}", C->type());
  }
}


#if CUDA_VERSION >= 11000 // Earlier versions of cublasLT do not support bias addition for fp32 and fp16.

static cublasStatus_t cublasLtAffineHelper(cublasLtHandle_t ltHandle, cublasOperation_t transA, cublasOperation_t transB,
                                           cudaDataType matrixType,
                                           int m, int n, int k, const void *alpha, const void *A, int lda, const void *B,
                                           int ldb, const void *beta, void *C, int ldc, const void* bias, 
                                           void* workspace, size_t workspaceSize, bool do_relu, cudaStream_t stream)  {

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  cublasLtEpilogue_t epilogue = do_relu? CUBLASLT_EPILOGUE_RELU_BIAS: CUBLASLT_EPILOGUE_BIAS;
  cublasComputeType_t computeType = matrixType == CUDA_R_32F? CUBLAS_COMPUTE_32F_FAST_16F: CUBLAS_COMPUTE_16F;

  // If the bias is not aligned, just matmul and invoke custom epilogue later. 
  // cublas fails with a misalignment error if this condition is not true.
  if((uintptr_t)bias % REQUIRED_BIAS_ALIGNMENT != 0) {
    epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  }

  CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, computeType, matrixType));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, matrixType, transA == CUBLAS_OP_N ? m : k, transA == CUBLAS_OP_N ? k : m, lda));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, matrixType, transB == CUBLAS_OP_N ? k : n, transB == CUBLAS_OP_N ? n : k, ldb));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, matrixType, m, n, ldc));

  // I think we need to do this since we can slice matrices...
  // The allocator always allocates on 256 byte boundaries but we have no guarantees about the alignment of a matrix slice so we filter out
  // algorithms that would not work with matrices not aligned to 256 bytes.
  int alignmentA = getAlignmentUpTo256(A);
  int alignmentB = getAlignmentUpTo256(B);
  int alignmentC = getAlignmentUpTo256(C);

  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignmentA, sizeof(alignmentA)));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignmentB, sizeof(alignmentB)));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &alignmentC, sizeof(alignmentC)));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alignmentC, sizeof(alignmentC)));
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  cublasStatus_t opStatus = cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, 
                                           &heuristicResult.algo, workspace, workspaceSize, stream);
  
  if (preference) CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));

  return opStatus;
}

static cublasStatus_t cublasLtAffineTyped(cublasLtHandle_t ltHandle, cublasOperation_t transA, cublasOperation_t transB,
                                          int m, int n, int k, const half *alpha, const half *A, int lda, const half *B,
                                          int ldb, const half *beta, half *C, int ldc, const half* bias, 
                                          half* workspace, size_t workspaceSizeBytes, bool do_relu, cudaStream_t stream) {
  return cublasLtAffineHelper(ltHandle, transA, transB, CUDA_R_16F, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bias, 
                              workspace, workspaceSizeBytes, do_relu, stream);
}

static cublasStatus_t cublasLtAffineTyped(cublasLtHandle_t ltHandle, cublasOperation_t transA, cublasOperation_t transB,
                                          int m, int n, int k, const float *alpha, const float *A, int lda, const float *B,
                                          int ldb, const float *beta, float *C, int ldc, const float* bias, 
                                          float* workspace, size_t workspaceSizeBytes,bool do_relu, cudaStream_t stream) {
  
  return cublasLtAffineHelper(ltHandle, transA, transB, CUDA_R_32F, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bias, 
                              workspace, workspaceSizeBytes, do_relu, stream);
}

template <typename T>
void affineTyped(marian::Tensor C, Ptr<Allocator> allocator, const marian::Tensor& A, const marian::Tensor& B, const marian::Tensor& bias,
                  bool transA, bool transB, T beta, T scalar, bool do_relu) {

  CUDA_CHECK(cudaSetDevice((int)C->getDeviceId().no));
  T alpha = scalar;
    
  int m = A->shape().elements() / A->shape().back();
  int k = A->shape().back();
  if(transA)
    std::swap(m, k);

  int l = B->shape().elements() / B->shape().back();
  int n = B->shape().back();
  if(transB)
    std::swap(l, n);

  int lda = A->shape().back();
  int ldb = B->shape().back();
  int ldc = B->shape().back();

  size_t bias_size = bias->shape().elements();
  ABORT_IF(n != bias_size, "The number of elements in the bias must match the number of columns in C");

  if(transB)
    ldc = B->shape().elements() / B->shape().back();

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto backend = std::static_pointer_cast<gpu::Backend>(C->getBackend());
  auto cublasHandle = backend->getCublasHandle();
  auto ltHandle = (cublasLtHandle_t)backend->getCublasHandle(); // A cublas handle encapsulates an lt handle

  size_t numWorkSpaceElts = 8192; // Allows for cublasLt to perform split-K gemms. This is chosen to be at least
                                  // 16 KiB for float16 which is large enough to prevent alloc failed errors
  size_t workspaceSizeBytes = numWorkSpaceElts * sizeof(T);
  IPtr<MemoryPiece> workspace = allocator->alloc<T>(numWorkSpaceElts);  

  cudaStream_t stream = 0;
  CUBLAS_CHECK(cublasGetStream(cublasHandle, &stream));


  CUBLAS_CHECK(cublasLtAffineTyped(ltHandle, 
                                   opB, 
                                   opA, 
                                   n, 
                                   m, 
                                   k, 
                                   &alpha, 
                                   B->data<T>(),
                                   ldb,
                                   A->data<T>(),
                                   lda,
                                   &beta,
                                   C->data<T>(),
                                   ldc,
                                   bias->data<T>(),
                                   workspace->data<T>(),
                                   workspaceSizeBytes,
                                   do_relu,
                                   stream));
  
  allocator->free(workspace);
}

// This version is needed so that Windows doesn't complain when compiling CUDA < 11. Otherwise, the ifdef could be inside of one
// definition of Affine.
void Affine(marian::Tensor C, 
            Ptr<Allocator> allocator, 
            const marian::Tensor& A, 
            const marian::Tensor& B, 
            const marian::Tensor& bias,
            bool transA, bool transB, float beta, float scalar, bool do_relu) {
  // There is a bug in CUDA 11 where the bias pointer needs to be 8 byte aligned. This bug will be fix in a subsequent release. For now,
  // we launch a custom epilogue if the bias does not meet the alignment requirement.           
  if(C->type() == Type::float32) {
    affineTyped<float>(C, allocator, A, B, bias, transA, transB, beta, scalar, do_relu);
    if((uintptr_t)bias->data<float>() % REQUIRED_BIAS_ALIGNMENT != 0) {
      BiasAdd(C, bias, do_relu);              
    }
#if COMPILE_FP16
  } else if(C->type() == Type::float16) {
    affineTyped<half>(C, allocator, A, B, bias, transA, transB, __float2half(beta), __float2half(scalar), do_relu);
    if((uintptr_t)bias->data<half>() % REQUIRED_BIAS_ALIGNMENT != 0) {
      BiasAdd(C, bias, do_relu);              
    }
#endif
  } else {
    ABORT("Affine not implemented for type {}", C->type());
  }
}

#else

void Affine(marian::Tensor C, 
            Ptr<Allocator> /*allocator*/, 
            const marian::Tensor& A, 
            const marian::Tensor& B, 
            const marian::Tensor& bias,
            bool transA, bool transB, float beta, float scalar, bool do_relu) {
             
  if(C->type() == Type::float32) {
    ProdTyped<float>(C, A, B, transA, transB, beta, scalar);
#if COMPILE_FP16
  } else if(C->type() == Type::float16) {
    ProdTyped<half>(C, A, B, transA, transB, __float2half(beta), __float2half(scalar));
#endif
  } else {
    ABORT("Prod not implemented for type {}", C->type());
  }
  BiasAdd(C, bias, do_relu);              
}
#endif

}  // namespace gpu
}  // namespace marian
