
#include <cublas_v2.h>

// clang-format off
#include "tensors/gpu/prod.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

namespace marian {

namespace gpu {

void Prod(marian::Tensor C,
          marian::Tensor A,
          marian::Tensor B,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
  cudaSetDevice(C->getDevice().no);
  float alpha = scalar;

  size_t m = A->shape().elements() / A->shape().back();
  size_t k = A->shape().back();
  if(transA)
    std::swap(m, k);

  size_t l = B->shape().elements() / B->shape().back();
  size_t n = B->shape().back();
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape().back();
  size_t ldb = B->shape().back();
  size_t ldc = B->shape().back();

  if(transB)
    ldc = B->shape().elements() / B->shape().back();

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())
                          ->getCublasHandle();

#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif

  cublasSgemm(cublasHandle,
              opB,
              opA,
              n,
              m,
              k,
              &alpha,
              B->data(),
              ldb,
              A->data(),
              lda,
              &beta,
              C->data(),
              ldc);
#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif
}

__global__ void gAddBias(float* out, const float* bias, size_t length, size_t cols) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      size_t index2 = index % cols;
      out[index] += bias[index2];
    }
  }
}

void AddBias(marian::Tensor C, const marian::Tensor bias) {
  cudaSetDevice(C->getDevice().no);

  int length = C->shape().elements();
  int cols = bias->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gAddBias<<<blocks, threads>>>(C->data(), bias->data(), length, cols);

  cudaStreamSynchronize(0);
}

void ProdWithBias(marian::Tensor C,
          const marian::Tensor A,
          const marian::Tensor B,
          const marian::Tensor bias,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
  marian::gpu::Prod(C, A, B, transA, transB, beta, scalar);
  marian::gpu::AddBias(C, bias);
}

//
//void ProdBatched2(marian::Tensor C,
//                 const marian::Tensor A,
//                 const marian::Tensor B,
//                 bool transA,
//                 bool transB,
//                 float beta,
//                 float scalar) {
//  cudaSetDevice(C->getDevice().no);
//  float alpha = scalar;
//
//  size_t batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
//  size_t batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);
//
//  size_t m = A->shape()[-2];
//  size_t k = A->shape()[-1];
//  if(transA)
//    std::swap(m, k);
//
//  size_t l = B->shape()[-2];
//  size_t n = B->shape()[-1];
//  if(transB)
//    std::swap(l, n);
//
//  size_t lda = A->shape()[-1];
//  size_t ldb = B->shape()[-1];
//  size_t ldc = B->shape()[-1];
//
//  if(transB)
//    ldc = B->shape()[-2];
//
//  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
//  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
//
//  auto cublasHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())
//                          ->getCublasHandle();
//
//#if CUDA_VERSION >= 9000
//  cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
//#endif
//  cublasSgemmStridedBatched(cublasHandle,
//                            opB,
//                            opA,
//                            n,
//                            m,
//                            k,
//                            &alpha,
//                            B->data(),
//                            ldb,
//                            batchB == 1 ? 0 : n * k,
//                            A->data(),
//                            lda,
//                            batchA == 1 ? 0 : m * k,
//                            &beta,
//                            C->data(),
//                            ldc,
//                            n * m,
//                            std::max(batchA, batchB));
//#if CUDA_VERSION >= 9000
//  cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
//#endif
//}

void ProdBatched(marian::Tensor C,
                 Ptr<Allocator> allocator,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta,
                 float scalar) {
  cudaSetDevice(C->getDevice().no);
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

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  auto cublasHandle = std::static_pointer_cast<gpu::Backend>(C->getBackend())
                          ->getCublasHandle();


  int strideA = batchA == 1 ? 0 : m * k;
  int strideB = batchB == 1 ? 0 : n * k;
  int strideC = n * m;
  int batchC = std::max(batchA, batchB);

  std::vector<const float*> aptr;
  std::vector<const float*> bptr;
  std::vector<float*> cptr;

  for(int i = 0; i < batchC; i++) {
    aptr.push_back(A->data() + (i % batchA) * strideA);
    bptr.push_back(B->data() + (i % batchB) * strideB);
    cptr.push_back(C->data() + i * strideC);
  }

  auto mp_aptr = allocator->alloc<const float*>(aptr.size());
  CudaCopy(aptr.data(), aptr.data() + aptr.size(), mp_aptr->data<const float*>());

  auto mp_bptr = allocator->alloc<const float*>(bptr.size());
  CudaCopy(bptr.data(), bptr.data() + bptr.size(), mp_bptr->data<const float*>());

  auto mp_cptr = allocator->alloc<float*>(cptr.size());
  CudaCopy(cptr.data(), cptr.data() + cptr.size(), mp_cptr->data<float*>());

#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
#endif
  cublasSgemmBatched(cublasHandle,
                      opB,
                      opA,
                      n,
                      m,
                      k,
                      &alpha,
                      mp_bptr->data<const float*>(),
                      ldb,
                      mp_aptr->data<const float*>(),
                      lda,
                      &beta,
                      mp_cptr->data<float*>(),
                      ldc,
                      batchC);
#if CUDA_VERSION >= 9000
  cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
#endif

  allocator->free(mp_aptr);
  allocator->free(mp_bptr);
  allocator->free(mp_cptr);
}

}
}
