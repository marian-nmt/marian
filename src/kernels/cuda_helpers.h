#pragma once
#include "3rd_party/exception.h"

#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true) {
  if(code != cudaSuccess) {
    UTIL_THROW2("GPUassert: " << cudaGetErrorString(code) << " " << file << " "
                              << line);
  }
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