#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include "tensor_operators.h"

namespace marian {

__global__ void gInitCurandStates(curandState* states, unsigned int seed);

class Bernoulli {
  private:
    float p_;
    curandState* states_;
    static unsigned seed;
    Shape shape_;
    
  public:
    Bernoulli(float p, const Shape& shape)
    : p_(p), shape_(shape) {}
    
    void InitStates(curandState* states) {
      states_ = states;
      int blocks = std::min(MAX_BLOCKS, shape_[0]);
      int threads = std::min(MAX_THREADS, shape_[1]);
      int n = blocks * threads;
      cudaMalloc((void**) &states_, n * sizeof(curandState));
      gInitCurandStates<<<blocks, threads>>>(states_, seed++);
      cudaStreamSynchronize(0);
    }
    
    void FreeStates(curandState* states) {
      cudaFree(states);
    }
    
    __device__ float operator()(int i, int j) const {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      float dist = curand_uniform(&states_[tid]);	
      float zeroOne = dist > p_;
      return zeroOne / (1 - p_);
    }
    
    __device__ int rows() const {
      return shape_[0];
    }
    
    __device__ int cols() const {
      return shape_[1];
    }
    
    Bernoulli& gpu() {
      return *this;
    }
};

}