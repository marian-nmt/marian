#pragma once

#include <cuda.h>
#include <cublas_v2.h>

namespace GPU {
namespace mblas {

class CudaStreamHandler {
    CudaStreamHandler()
    : stream_(new cudaStream_t()) {
      cudaStreamCreate(stream_.get());
      // cudaStreamCreateWithFlags(stream_.get(), cudaStreamNonBlocking);
    }

  protected:
    static thread_local CudaStreamHandler *instance_;
    std::unique_ptr<cudaStream_t> stream_;

  public:
    static cudaStream_t& GetStream() {
      if (instance_ == nullptr) {
        instance_ = new CudaStreamHandler();
      }
      return *(instance_->stream_);
    }

    virtual ~CudaStreamHandler() {
        cudaStreamDestroy(*stream_);
    }
};


class CublasHandler {
  public:
    static cublasHandle_t GetHandle() {
#ifdef __APPLE__
      cublasHandle_t *handle = handle_.get();
      if (handle == nullptr) {
        handle = new cublasHandle_t;
        handle_.reset(handle);
      }
      return *handle;
#else
      if(handle_ == nullptr) {
        assert(handle_ == nullptr);
        handle_ = new cublasHandle_t;
        cublasCreate(handle_);
        cublasSetStream(*handle_, CudaStreamHandler::GetStream());
      }
      return *handle_;
#endif
    }

  private:
    ~CublasHandler() {
      cublasDestroy(*handle_);
      if (handle_) {
        delete handle_;
      }
    }

#ifdef __APPLE__
    static boost::thread_specific_ptr<cublasHandle_t> handle_;
#else
    static thread_local cublasHandle_t* handle_;
#endif
};

} // namespace mblas
} // namespace GPU
