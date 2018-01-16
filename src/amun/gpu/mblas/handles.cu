#include "handles.h"
#include "gpu/types-gpu.h"

namespace amunmt {
namespace GPU {
namespace mblas {

CudaStreamHandler::CudaStreamHandler()
{
  HANDLE_ERROR( cudaStreamCreate(&stream_));
  // cudaStreamCreateWithFlags(stream_.get(), cudaStreamNonBlocking);
}

CudaStreamHandler::~CudaStreamHandler()
{
  HANDLE_ERROR(cudaStreamDestroy(stream_));
}

/////////////////////////////////////////////////////////////////////////////////////////

CublasHandler::CublasHandler()
{
  cublasStatus_t stat;
  stat = cublasCreate(&handle_);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("cublasCreate initialization failed\n");
    abort();
  }

#if CUDA_VERSION >= 9000
  /*
    stat = cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("cublasSetMathMode failed\n");
      abort();
    }
  */
#endif
		  
  stat = cublasSetStream(handle_, CudaStreamHandler::GetStream());
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("cublasSetStream initialization failed\n");
    abort();
  }
}

CublasHandler::~CublasHandler() {
  cublasDestroy(handle_);
}


} // namespace
}
}
