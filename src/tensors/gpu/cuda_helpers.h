#pragma once
#include "common/logging.h"
#include "cuda_runtime.h"
#include "nccl.h"

// fixes a missing constant in CUDA device code (specific to MSVC compiler)
static __constant__ float CUDA_FLT_MAX = 1.70141e+38;
const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

#define CUDA_CHECK(expr) do {                                                                      \
  cudaError_t rc = (expr);                                                                         \
  ABORT_IF(rc != cudaSuccess,                                                                      \
        "CUDA error {} '{}' - {}:{}: {}", rc, cudaGetErrorString(rc),  __FILE__, __LINE__, #expr); \
} while(0)

#define CUSPARSE_CHECK(expr) do {                                              \
  cusparseStatus_t rc = (expr);                                                \
  ABORT_IF(rc != CUSPARSE_STATUS_SUCCESS,                                      \
           "Cusparse Error: {} - {}:{}: {}", rc, __FILE__, __LINE__, #expr);   \
} while(0)

#define NCCL_CHECK(expr) do {                                                                      \
  ncclResult_t rc = (expr);                                                                        \
  ABORT_IF(rc != ncclSuccess,                                                                      \
        "NCCL error {} '{}' - {}:{}: {}", rc, ncclGetErrorString(rc),  __FILE__, __LINE__, #expr); \
} while(0)

#define CURAND_CHECK(expr) do {                                          \
  curandStatus_t rc = (expr);                                            \
  ABORT_IF(rc != CURAND_STATUS_SUCCESS,                                  \
          "Curand error {} - {}:{}: {}", rc, __FILE__, __LINE__, #expr); \
} while(0)

// @TODO: remove this if no longer used
inline void gpuAssert(cudaError_t code, const char* exprString,
                      const char* file,
                      int line) {
  ABORT_IF(code != cudaSuccess,
           "CUDA Error {}: {} - {}:{}: {}", code, cudaGetErrorString(code), file, line, exprString);
}

// @TODO: is this used anywhere?
template <typename T>
void CudaCopy(const T* start, const T* end, T* dest) {
  CUDA_CHECK(cudaMemcpy(dest, start, (end - start) * sizeof(T), cudaMemcpyDefault));
}

// void cusparseStatus(cusparseStatus_t status){
//  switch(status){
//    case CUSPARSE_STATUS_INVALID_VALUE:
//      printf("invalid value");
//      break;
//    case CUSPARSE_STATUS_NOT_INITIALIZED:
//      printf("not initialized");
//      break;
//    case CUSPARSE_STATUS_ARCH_MISMATCH:
//      printf("arch mismatch");
//      break;
//    case CUSPARSE_STATUS_EXECUTION_FAILED:
//      printf("exe failed");
//      break;
//    case CUSPARSE_STATUS_INTERNAL_ERROR:
//      printf("internal error");
//      break;
//    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
//      printf("not supported");
//      break;
//    case CUSPARSE_STATUS_ALLOC_FAILED:
//      printf("alloc failed");
//      break;
//    case CUSPARSE_STATUS_MAPPING_ERROR :
//      printf("map error");
//      break;
//    case CUSPARSE_STATUS_SUCCESS:
//      printf("success\n");
//      break;
//    default:
//        printf("unknown status\n");
//      break;
//  }
//}
