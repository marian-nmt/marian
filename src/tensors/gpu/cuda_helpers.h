#pragma once
#include <cstdlib>

#include "3rd_party/exception.h"
#include "common/logging.h"

const float CUDA_FLT_MAX = 1.70141e+38;
const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true) {
  if(code != cudaSuccess) {
    LOG(critical, "Error: {} - {}:{}", cudaGetErrorString(code), file, line);
    std::abort();
  }
}

template <typename T>
void CudaCopy(const T* start, const T* end, T* dest) {
  CUDA_CHECK(cudaMemcpy((void*)dest, (void*)start, (end - start) * sizeof(T),
             cudaMemcpyDefault));
}

#define CUSPARSE_CHECK(x)                               \
  {                                                     \
    cusparseStatus_t _c = x;                            \
    if(_c != CUSPARSE_STATUS_SUCCESS) {                 \
      printf("cusparse fail: %d, file: %s, line: %d\n", \
             (int)_c,                                   \
             __FILE__,                                  \
             __LINE__);                                 \
      exit(-1);                                         \
    }                                                   \
  }

// void cusparseStatus(cusparseStatus_t status){
//	switch(status){
//		case CUSPARSE_STATUS_INVALID_VALUE:
//			printf("invalid value");
//			break;
//		case CUSPARSE_STATUS_NOT_INITIALIZED:
//			printf("not initialized");
//			break;
//		case CUSPARSE_STATUS_ARCH_MISMATCH:
//			printf("arch mismatch");
//			break;
//		case CUSPARSE_STATUS_EXECUTION_FAILED:
//			printf("exe failed");
//			break;
//		case CUSPARSE_STATUS_INTERNAL_ERROR:
//			printf("internal error");
//			break;
//		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
//			printf("not supported");
//			break;
//		case CUSPARSE_STATUS_ALLOC_FAILED:
//			printf("alloc failed");
//			break;
//		case CUSPARSE_STATUS_MAPPING_ERROR :
//			printf("map error");
//			break;
//		case CUSPARSE_STATUS_SUCCESS:
//			printf("success\n");
//			break;
//		default:
//				printf("unknown status\n");
//			break;
//	}
//}
