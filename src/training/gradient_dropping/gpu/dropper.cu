#include <curand.h>
#include <curand_kernel.h>

#include <memory>

#include "tensors/gpu/cuda_helpers.h"
#include "tensors/tensor_operators.h"
#include "training/gradient_dropping/dropper.h"
#include "training/gradient_dropping/sparse_tensor.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>

namespace marian {

namespace gpu {

__global__ void sampling(float* originalData,
                               float* data,
                               int size,
                               int scale,
                               int fullSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  data[idx] = abs(originalData[idx * scale]);
}

float GradientDropBase::find_threshold(Tensor grads, float rate) { 
  cudaSetDevice(grads->getDevice().no);

  int size = grads->size();

  int threads = 512;
  int sortSize = min(100000, size);
  int blocksSample = 1 + sortSize / threads;


  if (!tmp) {
    tmp = newTensor(sortSize, grads->getBackend());
  }

  sampling<<<blocksSample, threads>>>(
      grads->data(), tmp->data(), sortSize, size / sortSize, size);
  thrust::device_ptr<float> dev_data_ptr(tmp->data());
  thrust::sort(dev_data_ptr, dev_data_ptr + sortSize);

  int cut_index = std::max(0, (int)(sortSize * rate) - 1);
  float t;
  cudaMemcpy(&t, tmp->data() + cut_index, sizeof(float), cudaMemcpyDeviceToHost);

  return t;
}

void GradientDropBase::dropGraph(Tensor grads,
                                 SparseTensor destination,
                                 float rate,
                                 float momentum) {
  // init
  if(!residual) {
    residual = newTensor(grads->size(), grads->getBackend());
    step = 0;
  }

  if(!velocity && momentum > 0.0) {
    velocity = newTensor(grads->size(), grads->getBackend());
  }

  // Step 1: add residual to the current gradient
  {
    using namespace functional;
    marian::gpu::Element(_1 = _1 + _2, grads, residual);
  }

  // step 2: find threshold 
  float t = find_threshold(grads, rate);

  // step 3: drop gradients lower than threshold
  //         store gradients lower than threshold into the residual
  {
    using namespace functional;
    marian::gpu::Element(_1 = if_then_else(abs(_2) > t, 0, _2), residual, grads);
    marian::gpu::Element(_1 = if_then_else(abs(_1) <= t, 0, _1), grads);
  }

  destination->fromDense(grads);

  step++;
}

}
}
