#pragma once

#include <cuda.h>
#include <cublas_v2.h>

namespace amunmt {
namespace GPU {
namespace mblas {

class CudaStreamHandler
{
public:
  static const cudaStream_t& GetStream() {
    return instance_.stream_;
  }

protected:
    static thread_local CudaStreamHandler instance_;
    cudaStream_t stream_;

    CudaStreamHandler()
    {
      HANDLE_ERROR( cudaStreamCreate(&stream_));
      // cudaStreamCreateWithFlags(stream_.get(), cudaStreamNonBlocking);
    }

    CudaStreamHandler(const CudaStreamHandler&) = delete;

    virtual ~CudaStreamHandler() {
      HANDLE_ERROR(cudaStreamDestroy(stream_));
    }
};


class CublasHandler
{
  public:
    static cublasHandle_t &GetHandle() {
        return instance_.handle_;
    }

  private:
    CublasHandler()
    {
      cublasStatus_t stat;
      stat = cublasCreate(&handle_);
      if (stat != CUBLAS_STATUS_SUCCESS) {
		  printf ("cublasCreate initialization failed\n");
		  abort();
      }

      stat = cublasSetStream(handle_, CudaStreamHandler::GetStream());
      if (stat != CUBLAS_STATUS_SUCCESS) {
		  printf ("cublasSetStream initialization failed\n");
		  abort();
      }
    }

    ~CublasHandler() {
      cublasDestroy(handle_);
    }

    static thread_local CublasHandler instance_;
    cublasHandle_t handle_;
};

} // namespace mblas
} // namespace GPU
}
