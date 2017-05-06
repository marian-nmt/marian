#pragma once

#include <vector>
#include <cuda.h>

#include "gpu/mblas/handles.h"


namespace amunmt {
namespace GPU {
namespace mblas {


// void HandleError(cudaError_t err, const char *file, int line );

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

template<typename T>
__global__ void gSetValue(T* data, size_t n, T val) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += gridDim.x * blockDim.x) {
    data[idx] = val;
  }
}

template<typename T>
void SetValue(T* data, int n, T val) {
  int threadNum = std::min(n, 512);
  int blockNum = (n / 512) + ( (n % 512) != 0 );

  gSetValue<<<blockNum, threadNum, 0, CudaStreamHandler::GetStream()>>>
    (data, n, val);
}


template<typename T>
class device_vector
{
  public:
    device_vector()
      : data_(nullptr),
        size_(0),
        realSize_(0)
    {}

    device_vector(size_t size)
      : size_(size),
        realSize_(size)
    {
      HANDLE_ERROR( cudaMalloc((void**)&data_, size_ * sizeof(T)) );
    }

    device_vector(size_t size, T val)
      : device_vector(size)
    {
      SetValue(data_, size, val);
    }

    device_vector(const std::vector<T>& hostVector)
      : device_vector(hostVector.size())
    {
      HANDLE_ERROR( cudaMemcpyAsync(
          data_,
          hostVector.data(),
          hostVector.size() * sizeof(T),
          cudaMemcpyHostToDevice,
          CudaStreamHandler::GetStream()) );
    }

    void resize(size_t newSize) {
      if (newSize > realSize_) {
        if (data_ == nullptr) {
          HANDLE_ERROR( cudaMalloc((void**)&data_, newSize * sizeof(T)) );
          realSize_ = newSize;
          size_ = newSize;
        } else {
          T* newData;
          HANDLE_ERROR( cudaMalloc((void**)&newData, newSize * sizeof(T)) );
          HANDLE_ERROR( cudaMemcpyAsync(
                newData,
                data_,
                size_ * sizeof(T),
                cudaMemcpyDeviceToDevice,
                CudaStreamHandler::GetStream())
          );
          HANDLE_ERROR( cudaFree(data_) );
          data_ = newData;
          realSize_ = newSize;
          size_ = newSize;
        }
      }
      size_ = newSize;
    }

    size_t size() const {
      return size_;
    }

    T* data() {
      return data_;
    }

    T* data() const {
      return data_;
    }

    ~device_vector() {
      if (data_) {
        HANDLE_ERROR( cudaFree(data_) );
      }
    }

  protected:
    T* data_;
    size_t size_;
    size_t realSize_;

};

}  // namespace mblas
}  // namespace GPU
}  // namespace amunmt
