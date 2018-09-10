#include <cuda.h>
#include <iostream>

#include "tensors/device.h"
#include "tensors/gpu/cuda_helpers.h"

namespace marian {
namespace gpu {

Device::~Device() {
  cudaSetDevice(deviceId_.no);
  if(data_) {
    CUDA_CHECK(cudaFree(data_));
  }
  cudaDeviceSynchronize();
}

void Device::reserve(size_t size) {
  size = align(size);
  cudaSetDevice(deviceId_.no);

  ABORT_IF(size < size_ || size == 0,
           "New size must be larger than old size and larger than 0");

  if(data_) {
    // Allocate memory by going through host memory
    uint8_t *temp = new uint8_t[size_]; // TODO: use std::vector
    CUDA_CHECK(cudaMemcpy(temp, data_, size_, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(data_));
    LOG(info, "re-allocating {} bytes on device {} (with CPU detour)", size, deviceId_.no);
    CUDA_CHECK(cudaMalloc(&data_, size));
    CUDA_CHECK(cudaMemcpy(data_, temp, size_, cudaMemcpyHostToDevice));
    delete[] temp;
  } else {
    LOG(info, "re-allocating {} bytes in device {} (without CPU detour)", size, deviceId_.no);
    CUDA_CHECK(cudaMalloc(&data_, size));
  }

  size_ = size;
}
}  // namespace gpu
}  // namespace marian
