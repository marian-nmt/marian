#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <memory>

#include "kernels/cuda_helpers.h"
#include "kernels/tensor_operators.h"
#include "training/dropper.h"
#include "training/sparse_tensor.h"

namespace marian {

__global__ void grad_drop(float* data,
                          float* tmp,
                          float* residual,
                          float* velocity,
                          float cut_off,
                          int max_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= max_size)
    return;

  bool mask = std::abs(data[idx]) > cut_off;

  residual[idx] = data[idx] * !mask; //store residual
  velocity[idx] = velocity[idx] * !mask; //momentum factor masking
  data[idx] = data[idx] * mask; //send
  tmp[idx] = 1 * mask;
  
}

__global__ void grad_add_error(float* data, float* residual, float* velocity, float m, int max_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= max_size)
    return;
  // momentum correction
  velocity[idx] = m * velocity[idx] + data[idx];
  data[idx] = velocity[idx] + residual[idx];
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

__global__ void randomSampling(float* originalData,
                               float* data,
                               int size,
                               int scale,
                               int fullSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  data[idx] = abs(originalData[idx * scale]);
}

void GradientDropBase::grad_drop_do(float* grads,
                                    float* residual,
                                    float* velocity,
                                    float* tmp,
                                    int len,
                                    float rate,
                                    float m) {
  int threads = 512;
  int blocks = 1 + len / threads;
  cudaSetDevice(_deviceId.no);

  grad_add_error<<<blocks, threads>>>(grads, residual, velocity, m, len);
  // full sort
  // int sortSize = len;
  int sortSize = min(100000, len);
  int blocksSample = 1 + sortSize / threads;
  randomSampling<<<blocksSample, threads>>>(
      grads, tmp, sortSize, len / sortSize, len);

  thrust::device_ptr<float> dev_data_ptr(tmp);
  thrust::sort(dev_data_ptr, dev_data_ptr + sortSize);

  int cut_index = std::max(0, (int)(sortSize * rate) - 1);
  cudaMemcpy(&cut_off, tmp + cut_index, sizeof(float), cudaMemcpyDeviceToHost);

  grad_drop<<<blocks, threads>>>(grads, tmp, residual, velocity, cut_off, len);
}

void GradientDropBase::dropGraph(Tensor t,
                                 SparseTensor destination,
                                 double rate,
                                 double momentum) {
  cudaSetDevice(t->getDevice().no);
  if(!residual) {
    _deviceId = t->getDevice();
    CUDA_CHECK(cudaMalloc(&residual, sizeof(float) * t->size()));
    CUDA_CHECK(cudaMalloc(&temp_d, sizeof(float) * t->size()));
    CUDA_CHECK(cudaMalloc(&velocity, sizeof(float) * t->size()));

    cudaMemset(residual, 0, sizeof(float) * t->size());
    cudaMemset(temp_d, 0, sizeof(float) * t->size());
    cudaMemset(velocity, 0, sizeof(float) * t->size());
    step = 0;
  }

  // drop the gradients in t->data(). Also fills in feedback with the
  // propagated error fills temp_d with binary flag. 0 means that gradient in
  // that position is dropped, 1 otherwise
  grad_drop_do(t->data(), residual, velocity, temp_d, t->size(), rate, momentum);

  // do inclusive sum on temp_d, to obtain the sparse matrix location of
  // non-dropped gradients
  thrust::device_ptr<float> mask_ptr(temp_d);
  int denseSize = t->size();
  thrust::inclusive_scan(mask_ptr, mask_ptr + denseSize, mask_ptr);
  float sparseSize;

  cudaMemcpy(&sparseSize,
             temp_d + denseSize - 1,
             sizeof(float),
             cudaMemcpyDeviceToHost);

  int threads = 512;
  int blocks = 1 + denseSize / threads;
  cudaSetDevice(t->getDevice().no);
  buildIndices<<<blocks, threads>>>(t->data(),
                                    temp_d,
                                    destination->data(),
                                    destination->indices(),
                                    denseSize);
  destination->setSize(sparseSize);

  cudaStreamSynchronize(0);

  step++;
}
}
