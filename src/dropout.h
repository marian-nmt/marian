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

template <class T1, class T2>
__global__ void gDropout(T1 out, T2 drop) {
  int rows = out.rows();
  int cols = out.cols();
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int i = bid + blockIdx.x;
    if(i < rows) {
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int j = tid + threadIdx.x;
        if(j < cols)
          out(i, j) = drop(i, j);
      }
    }
  }
}

template <class T1, class T2>
void Dropout(T1 out, T2 drop) {

  int m = out.shape()[0];
  int n = out.shape()[1];

  int blocks  = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, n);
  gDropout<<<blocks, threads>>>(out.gpu(), drop.gpu());
  cudaStreamSynchronize(0);
}

}
