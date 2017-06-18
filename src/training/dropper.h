#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <memory>

#include "kernels/cuda_helpers.h"
#include "kernels/tensor_operators.h"
#include "training/sparse_tensor.h"

namespace marian {

__global__ void grad_drop(
    float* data, float* tmp, float* errors, float cut_off, int max_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= max_size)
    return;
  if(std::abs(data[idx]) <= cut_off) {
    errors[idx] = data[idx];
    data[idx] = 0;
    tmp[idx] = 0;
  } else {
    errors[idx] = 0;
    tmp[idx] = 1;
  }
}

__global__ void grad_add_error(float* data, float* errors, int max_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= max_size)
    return;
  data[idx] += errors[idx];
}

__global__ void full_abs(float* data, int max_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= max_size)
    return;
  data[idx] = abs(data[idx]);
}

__global__ void buildIndices(float* denseData,
                             float* denseSum,
                             float* sparseData,
                             int* sparseIndices,
                             int denseSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= denseSize)
    return;
  int t_id = round(denseSum[idx]);
  if(t_id <= 0) {
    return;
  }

  if(idx == 0 && t_id > 0) {
    sparseIndices[t_id - 1] = idx;
    sparseData[t_id - 1] = denseData[idx];
  } else if(idx > 0 && t_id > round(denseSum[idx - 1])) {
    sparseIndices[t_id - 1] = idx;
    sparseData[t_id - 1] = denseData[idx];
  }
}

__global__ void randomSampling(
    float* originalData, float* data, int size, int scale, int fullSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  data[idx] = abs(originalData[idx * scale]);
}

class GradientDropBase {
  float* feedback;
  float* temp_d;
  float cut_off;
  int step;
  int _device;

  void grad_drop_do(
      float* data, float* errors, float* tmp, int len, float rate) {
    int threads = 512;
    int blocks = 1 + len / threads;
    cudaSetDevice(_device);

    grad_add_error<<<blocks, threads>>>(data, errors, len);
    // full sort
    // int sortSize = len;
    int sortSize = min(100000, len);
    int blocksSample = 1 + sortSize / threads;
    randomSampling<<<blocksSample, threads>>>(
        data, tmp, sortSize, len / sortSize, len);
    // dont update the cut threshold every step

    thrust::device_ptr<float> dev_data_ptr(tmp);
    thrust::sort(dev_data_ptr, dev_data_ptr + sortSize);

    int cut_index = std::max(0, (int)(sortSize * rate) - 1);
    cudaMemcpy(
        &cut_off, tmp + cut_index, sizeof(float), cudaMemcpyDeviceToHost);

    grad_drop<<<blocks, threads>>>(data, tmp, errors, cut_off, len);
  }

public:
  void dropGraph(Tensor t, SparseTensor destination, double rate = 0.99) {
    cudaSetDevice(t->getDevice());
    if(!feedback) {
      _device = t->getDevice();
      cudaMalloc(&feedback, sizeof(float) * t->size());
      cudaMalloc(&temp_d, sizeof(float) * t->size());
      cudaMemset(feedback, 0, sizeof(float) * t->size());
      cudaMemset(temp_d, 0, sizeof(float) * t->size());

      step = 0;
    }

    grad_drop_do(t->data(), feedback, temp_d, t->size(), rate);

    thrust::device_ptr<float> mask_ptr(temp_d);
    int denseSize = t->size();
    thrust::inclusive_scan(mask_ptr, mask_ptr + denseSize, mask_ptr);
    float sparseSize;

    cudaMemcpy(&sparseSize,
               temp_d + denseSize - 1,
               sizeof(float),
               cudaMemcpyDeviceToHost);

    // convert result of exscan to indices.
    int threads = 512;
    int blocks = 1 + denseSize / threads;
    cudaSetDevice(t->getDevice());
    buildIndices<<<blocks, threads>>>(t->data(),
                                      temp_d,
                                      destination->data(),
                                      destination->indices(),
                                      denseSize);
    destination->setSize(sparseSize);

    cudaStreamSynchronize(0);

    step++;
  }
};

typedef Ptr<GradientDropBase> GradientDrop;
}
