#include <cuda.h>
#include <iostream>

#include "tensors/device.h"
#include "tensors/gpu/cuda_helpers.h"

namespace marian {
namespace gpu {

Device::~Device() {
  // Note: The CUDA_CHECKs here are not throwing, but will terminate the program.
  CUDA_CHECK(cudaSetDevice(deviceId_.no));
  if(data_) {
    CUDA_CHECK(cudaFree(data_));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

void Device::reserve(size_t size) {
  size = align(size);
  CUDA_CHECK(cudaSetDevice(deviceId_.no));

  ABORT_IF(size < size_ || size == 0,
           "New size must be larger than old size and larger than 0");

  if(data_) {
    // Allocate memory while temporarily parking original content in host memory
    uint8_t *temp = new uint8_t[size_]; // @TODO: use std::vector
    CUDA_CHECK(cudaMemcpy(temp, data_, size_, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(data_));
    LOG(info, "[memory] Re-allocating {} bytes on device {}", size, deviceId_.no);
    CUDA_CHECK(cudaMalloc(&data_, size));
    CUDA_CHECK(cudaMemcpy(data_, temp, size_, cudaMemcpyHostToDevice));
    delete[] temp;
  } else {
    // No data_ yet: Just alloc.
    LOG(info, "[memory] Allocating {} bytes in device {}", size, deviceId_.no);
    CUDA_CHECK(cudaMalloc(&data_, size));
  }

  size_ = size;
}
}  // namespace gpu
}  // namespace marian
