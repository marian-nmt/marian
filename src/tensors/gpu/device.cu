#include <cuda.h>
#include <iostream>

#include "tensors/device.h"
#include "tensors/gpu/cuda_helpers.h"

namespace marian {
namespace gpu {

Device::~Device() {
  // No CUDA error checking as this is a destructor and we cannot do anything about errors anyway.
  cudaSetDevice(deviceId_.no);
  if(data_) {
    cudaFree(data_);
  }
  cudaDeviceSynchronize();
}

void Device::reserve(size_t size) {
  size = align(size);
  CUDA_CHECK(cudaSetDevice(deviceId_.no));

  ABORT_IF(size < size_ || size == 0,
           "New size must be larger than old size and larger than 0");

  if(data_) {
    // Allocate memory while temporarily parking original content in host memory
    std::vector<uint8_t> temp(size_);
    CUDA_CHECK(cudaMemcpy(temp.data(), data_, size_, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(data_));
    LOG(debug, "[memory] Re-allocating from {} to {} bytes on device {}", size_, size, deviceId_.no);
    CUDA_CHECK(cudaMalloc(&data_, size));
    CUDA_CHECK(cudaMemcpy(data_, temp.data(), size_, cudaMemcpyHostToDevice));
    //logCallStack(0);
  } else {
    // No data_ yet: Just alloc.
    LOG(debug, "[memory] Allocating {} bytes in device {}", size, deviceId_.no);
    CUDA_CHECK(cudaMalloc(&data_, size));
  }

  size_ = size;
}
}  // namespace gpu
}  // namespace marian
