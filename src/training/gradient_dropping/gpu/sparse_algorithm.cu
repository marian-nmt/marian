#include "training/gradient_dropping/gpu/sparse_algorithm.h"

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include "tensors/gpu/algorithm.h"
#include "tensors/gpu/cuda_helpers.h"

namespace marian {
namespace gpu {
struct non_zero {
  __host__ __device__ bool operator()(const float x) { return x != 0; }
};

__global__ void copy_id(float* data, int* indices, float* out, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  out[idx] = data[indices[idx]];
}

__global__ void gScatterAdd(float* denseData,
                            float* sparseData,
                            int* sparseIndices,
                            int denseSize,
                            int sparseSize,
                            int offset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= sparseSize)
    return;
  if(sparseIndices[idx] >= -offset && sparseIndices[idx] + offset < denseSize)
    denseData[sparseIndices[idx] + offset] += sparseData[idx];
}

__global__ void gScatterUpdate(float* denseData,
                               float* sparseData,
                               int* sparseIndices,
                               int denseSize,
                               int sparseSize,
                               int offset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= sparseSize)
    return;
  if(sparseIndices[idx] >= -offset && sparseIndices[idx] + offset < denseSize)
    denseData[sparseIndices[idx] + offset] = sparseData[idx];
}

__global__ void gGather(float* denseData,
                        float* sparseData,
                        int* sparseIndices,
                        int denseSize,
                        int sparseSize,
                        int offset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= sparseSize)
    return;
  if(sparseIndices[idx] >= -offset && sparseIndices[idx] + offset < denseSize)
    sparseData[idx] = denseData[sparseIndices[idx] + offset];
}

std::vector<int> lower_bounds(int* data,
                              std::vector<int> values,
                              int size,
                              DeviceId device) {
  cudaSetDevice(device.no);

  thrust::device_ptr<int> data_ptr(data);
  thrust::device_vector<int> d_values(values);
  thrust::device_vector<int> d_output(values.size());

  thrust::lower_bound(data_ptr,
                      data_ptr + size,
                      d_values.begin(),
                      d_values.end(),
                      d_output.begin());

  std::vector<int> output(values.size());
  thrust::copy(d_output.begin(), d_output.end(), output.begin());

  return output;
}

int buildSparse(Tensor t, float* data, int* indices) {
  cudaSetDevice(t->getDevice().no);
  using namespace thrust;

  device_ptr<float> grad_ptr(t->data());
  device_ptr<float> sparse_grad_ptr(data);
  device_ptr<int> indices_ptr(indices);

  int sparse_size = copy_if(make_counting_iterator<int>(0),
                            make_counting_iterator<int>(t->size()),
                            grad_ptr,
                            indices_ptr,
                            non_zero())
                    - indices_ptr;

  int threads = 512;
  int blocks = 1 + t->size() / threads;
  copy_id<<<blocks, threads>>>(t->data(), indices, data, sparse_size);

  return sparse_size;
}

void scatterAdd(Tensor t, float* data, int* indices, int size, int offset) {
  cudaSetDevice(t->getDevice().no);

  int threads = 512;
  int blocks = 1 + size / threads;
  gScatterAdd<<<blocks, threads>>>(
      t->data(), data, indices, t->size(), size, offset);
  cudaStreamSynchronize(0);
}

void scatterUpdate(Tensor t, float* data, int* indices, int size, int offset) {
  cudaSetDevice(t->getDevice().no);

  int threads = 512;
  int blocks = 1 + size / threads;
  gScatterUpdate<<<blocks, threads>>>(
      t->data(), data, indices, t->size(), size, offset);
  cudaStreamSynchronize(0);
}

void gather(Tensor t, float* data, int* indices, int size, int offset) {
  cudaSetDevice(t->getDevice().no);

  int threads = 512;
  int blocks = 1 + size / threads;
  gGather<<<blocks, threads>>>(
      t->data(), data, indices, t->size(), size, offset);
  cudaStreamSynchronize(0);
}
}
}
