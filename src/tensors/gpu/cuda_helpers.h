#pragma once
#include "common/logging.h"
#include "common/types.h"

#include <cuda_runtime.h>

// template <> inline bool matchType<__half>(Type type)  { return type == Type::float16; }
// template <> inline std::string request<__half>()  { return "float16"; }

// fixes a missing constant in CUDA device code
#define CUDA_FLT_MAX 1.70141e+38; // note: 'static __constant__' causes a warning on gcc; non-static fails CUDA, so #define instead
const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

#define CUDA_CHECK(expr) do {                                                                      \
  cudaError_t rc = (expr);                                                                         \
  ABORT_IF(rc != cudaSuccess,                                                                      \
        "CUDA error {} '{}' - {}:{}: {}", rc, cudaGetErrorString(rc),  __FILE__, __LINE__, #expr); \
} while(0)

#define CUBLAS_CHECK(expr) do {                                              \
  cublasStatus_t rc = (expr);                                                \
  ABORT_IF(rc != CUBLAS_STATUS_SUCCESS,                                      \
           "Cublas Error: {} - {}:{}: {}", rc, __FILE__, __LINE__, #expr);   \
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
