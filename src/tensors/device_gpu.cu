#include "tensors/device_gpu.h"

#include <cuda.h>
#include <iostream>
#include "kernels/cuda_helpers.h"

namespace marian {

DeviceGPU::~DeviceGPU() {
  cudaSetDevice(device_);
  if(data_) {
    CUDA_CHECK(cudaFree(data_));
  }
  cudaDeviceSynchronize();
}

void DeviceGPU::reserve(size_t size) {
  size = align(size);
  cudaSetDevice(device_);

  UTIL_THROW_IF2(size < size_, "New size must be larger than old size");

  if(data_) {
    // Allocate memory by going through host memory
    uint8_t *temp = new uint8_t[size_];
    CUDA_CHECK(cudaMemcpy(temp, data_, size_, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(data_));
    CUDA_CHECK(cudaMalloc(&data_, size));
    CUDA_CHECK(cudaMemcpy(data_, temp, size_, cudaMemcpyHostToDevice));
    delete[] temp;
  } else {
    CUDA_CHECK(cudaMalloc(&data_, size));
  }

  size_ = size;
}

}