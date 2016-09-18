#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tensor.h"

namespace marian {

using namespace thrust::placeholders;
#define MAX_THREADS 512
#define MAX_BLOCKS 65535

template <class Functor>
__global__ void gElement(Functor functor, Float* out,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      Float* rowOut = out + j * cols;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i]);;
      }
    }
  }
}

template <class Functor>
__global__ void gElement(Functor functor,
                         Float* out, const Float* in,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      Float* rowOut = out + j * cols;
      const Float* rowIn = in + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn[i]);;
      }
    }
  }
}

template <class Functor>
__global__ void gElement(Functor functor,
                         Float* out, const Float* in1, const Float* in2,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      Float* rowOut = out + j * cols;
      const Float* rowIn1 = in1 + j * cols;
      const Float* rowIn2 = in2 + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn1[i], rowIn2[i]);
      }
    }
  }
}

template <class Functor>
__global__ void gElement(Functor functor,
                         Float* out, const Float* in1,
                         const Float* in2, const Float* in3,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      Float* rowOut = out + j * cols;
      const Float* rowIn1 = in1 + j * cols;
      const Float* rowIn2 = in2 + j * cols;
      const Float* rowIn3 = in3 + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn1[i], rowIn2[i], rowIn3[i]);
      }
    }
  }
}

// @TODO add broadcasting

template <class Functor>
void Element(Functor functor, Tensor Out) {
  Float* d_out = Out.data();
  int blocks  = std::min(MAX_BLOCKS, (int)Out.shape()[0]);
  int threads = std::min(MAX_THREADS, (int)Out.shape()[1]);
  gElement<<<blocks, threads>>>(functor, d_out,
                                Out.shape()[0], Out.shape()[1]);
  cudaStreamSynchronize(0);
}

template <class Functor>
void Element(Functor functor,
             Tensor Out, const Tensor In) {
  Float* d_out = Out.data();
  const Float* d_in = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)Out.shape()[0]);
  int threads = std::min(MAX_THREADS, (int)Out.shape()[1]);
  gElement<<<blocks, threads>>>(functor, d_out, d_in,
                                Out.shape()[0], Out.shape()[1]);
  cudaStreamSynchronize(0);
}

template <class Functor>
void Element(Functor functor,
             Tensor Out, const Tensor In1, const Tensor In2) {
  
  Float* d_out = Out.data();
  const Float* d_in1 = In1.data();
  const Float* d_in2 = In2.data();
  
  int blocks  = std::min(MAX_BLOCKS, (int)Out.shape()[0]);
  int threads = std::min(MAX_THREADS, (int)Out.shape()[1]);
  gElement<<<blocks, threads>>>(functor, d_out, d_in1, d_in2,
                                Out.shape()[0], Out.shape()[1]);
  cudaStreamSynchronize(0);
}

template <class Functor>
void Element(Functor functor,
             Tensor Out, const Tensor In1,
             const Tensor In2, const Tensor In3) {
  
  Float* d_out = Out.data();
  const Float* d_in1 = In1.data();
  const Float* d_in2 = In2.data();
  const Float* d_in3 = In3.data();
  
  int blocks  = std::min(MAX_BLOCKS, (int)Out.shape()[0]);
  int threads = std::min(MAX_THREADS, (int)Out.shape()[1]);
  gElement<<<blocks, threads>>>(functor, d_out, d_in1, d_in2, d_in3,
                                Out.shape()[0], Out.shape()[1]);
  cudaStreamSynchronize(0);
}

__global__ void gSubtractMean(float* out, float* weights,
                              size_t rows, size_t cols);

void SubtractMean(Tensor* Out, Tensor &Weights);

__global__ void gSubtractMax(float* out, size_t rows, size_t cols);

void SubtractMax(Tensor* Out);

__global__ void gSoftMax(float* softMaxP, size_t rows, size_t cols);

void Softmax(Tensor* Out);

__global__ void gArgMax(float *out, const float *data, size_t rows, size_t cols);

void Argmax(Tensor* Out, const Tensor* In);

Tensor Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta);

Tensor Prod(Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta = 0);

}
