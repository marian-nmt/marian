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

    CudaStreamHandler();
    CudaStreamHandler(const CudaStreamHandler&) = delete;
    virtual ~CudaStreamHandler();

};

/////////////////////////////////////////////////////////////////////////////////////////

class CublasHandler
{
  public:
    static cublasHandle_t &GetHandle() {
        return instance_.handle_;
    }

  private:
    CublasHandler();
    ~CublasHandler();

    static thread_local CublasHandler instance_;
    cublasHandle_t handle_;
};

} // namespace mblas
} // namespace GPU
}
