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

#include <cublas_v2.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include "tensors/tensor_gpu.h"

namespace marian {

using namespace thrust::placeholders;
#define MAX_THREADS 512
#define MAX_BLOCKS 65535

class TensorGPU;

template <class Functor>
__global__ void gElementVec(Functor functor,
                            float* out, const float* in,
                            int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int noColumn = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (noColumn < length) {
      out[noColumn] = functor(out[noColumn], in[noColumn]);
    }
  }
}

template <class Functor, class T1, class T2>
void ElementVec(Functor functor,
                T1 out, T2 in) {

  int rows = out->shape()[0];
  int cols = out->shape()[1];

  int length = rows * cols;

  float* d_out = out->data();
  float* d_in  = in->data();

  int threads = std::min(MAX_THREADS, length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gElementVec<<<blocks, threads>>>(functor, d_out, d_in, length);
  cudaStreamSynchronize(0);
}

template <class Functor>
__global__ void gElementVec(Functor functor,
                            float* out,
                            const float* in1,
                            const float* in2,
                            int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int noColumn = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if (noColumn < length) {
      out[noColumn] = functor(out[noColumn],
                              in1[noColumn],
                              in2[noColumn]);
    }
  }
}

template <class Functor, class T1, class T2, class T3>
void ElementVec(Functor functor,
                T1 out, T2 in1, T3 in2) {

  int rows = out->shape()[0];
  int cols = out->shape()[1];

  int length = rows * cols;

  float* d_out = out->data();
  float* d_in1  = in1->data();
  float* d_in2  = in2->data();

  int threads = std::min(MAX_THREADS, (int)length);
  int blocks  = std::min(MAX_BLOCKS, length / threads  + (length % threads != 0));

  gElementVec<<<blocks, threads>>>(functor, d_out, d_in1, d_in2, length);
  cudaStreamSynchronize(0);
}

template <class Functor, class T>
__global__ void gElement(Functor functor,
                         T out) {
  int rows = out.shape()[0];
  int cols = out.shape()[1];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int i = bid + blockIdx.x;
    if(i < rows) {
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int j = tid + threadIdx.x;
        if(j < cols)
          out(i, j) = functor(out(i, j));
      }
    }
  }
}

template <class Functor, class T>
void Element(Functor functor, T& out) {

  int m = out->shape()[0];
  int n = out->shape()[1];

  int blocks  = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, n);

  auto outGpu = static_cast<TensorGPU*>(out.get());

  gElement<<<blocks, threads>>>(functor, outGpu->access());
  cudaStreamSynchronize(0);
}


template <class Functor, class T1, class T2>
__global__ void gElement(Functor functor,
                         T1 out, T2 in) {
  int rows = out.shape()[0];
  int cols = out.shape()[1];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int i = bid + blockIdx.x;
    if(i < rows) {
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int j = tid + threadIdx.x;
        if(j < cols) {
          out(i, j) = functor(out(i, j), in(i, j));
        }
      }
    }
  }
}

template <class Functor, class T1, class T2>
void Element(Functor functor,
             T1& out, T2& in) {

  if(out->shape() == in->shape()) {
    ElementVec(functor, out, in);
  }
  else {
    int m = out->shape()[0];
    int n = out->shape()[1];

    int blocks  = std::min(MAX_BLOCKS, m);
    int threads = std::min(MAX_THREADS, n);

    auto inGpu = static_cast<TensorGPU*>(in.get());
    auto outGpu = static_cast<TensorGPU*>(out.get());

    gElement<<<blocks, threads>>>(functor,
                                  outGpu->access(), inGpu->access());
    cudaStreamSynchronize(0);
  }
}

template <class Functor, class T1, class T2, class T3>
__global__ void gElement(Functor functor,
                         T1 out, T2 in1, T3 in2) {
  int rows = out.shape()[0];
  int cols = out.shape()[1];
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int i = bid + blockIdx.x;
    if(i < rows) {
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int j = tid + threadIdx.x;
        if(j < cols)
          out(i, j) = functor(out(i, j), in1(i, j), in2(i, j));
      }
    }
  }
}

template <class Functor, class T1, class T2, class T3>
void Element(Functor functor,
             T1& out, T2& in1, T3& in2) {

  if(out->shape() == in1->shape() && in1->shape() == in2->shape()) {
    ElementVec(functor, out, in1, in2);
  }
  else {
    auto in1Gpu = static_cast<TensorGPU*>(in1.get());
    auto in2Gpu = static_cast<TensorGPU*>(in2.get());
    auto outGpu = static_cast<TensorGPU*>(out.get());

    int m = out->shape()[0];
    int n = out->shape()[1];

    int blocks  = std::min(MAX_BLOCKS, m);
    int threads = std::min(MAX_THREADS, n);
    gElement<<<blocks, threads>>>(functor,
                                  outGpu->access(),
                                  in1Gpu->access(),
                                  in2Gpu->access());
    cudaStreamSynchronize(0);
  }
}

template <class Functor, class T1, class T2, class T3, class T4>
__global__ void gElement(Functor functor,
                         T1 out, T2 in1, T3 in2, T4 in3) {
  int rows = out.shape()[0];
  int cols = out.shape()[1];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int i = bid + blockIdx.x;
    if(i < rows) {
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int j = tid + threadIdx.x;
        if(j < cols)
          out(i, j) = functor(out(i, j), in1(i, j), in2(i, j), in3(i, j));
      }
    }
  }
}

template <class Functor, class T1, class T2, class T3, class T4>
void Element(Functor functor,
             T1& out, T2& in1, T3& in2, T4& in3) {

  auto in1Gpu = static_cast<TensorGPU*>(in1.get());
  auto in2Gpu = static_cast<TensorGPU*>(in2.get());
  auto in3Gpu = static_cast<TensorGPU*>(in3.get());
  auto outGpu = static_cast<TensorGPU*>(out.get());

  int m = outGpu->shape()[0];
  int n = outGpu->shape()[1];

  int blocks  = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, n);
  gElement<<<blocks, threads>>>(functor,
                                outGpu->access(),
                                in1Gpu->access(),
                                in2Gpu->access(),
                                in3Gpu->access());
  cudaStreamSynchronize(0);
}

void ClipNorm(Tensor out, float threshold);

void SubtractMax(Tensor out, Tensor in);

void Softmax(Tensor out, Tensor in);

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val);
void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnSoftmax(Tensor out, Tensor in);
void CudnnSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnLogSoftmax(Tensor out, Tensor in);
void CudnnLogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void Argmax(Tensor Out, const Tensor In);

void Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta);

void Prod(Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta = 0);

void Sum(Tensor out, const Tensor in, int axis=-1, bool mean=false);
void SumBackward(Tensor out, const Tensor in, int axis=-1, bool mean=false);

void CopyRowsByIndex(Tensor out, const Tensor in,
                     thrust::pair<size_t, size_t>* ipair, size_t length);

void CopyRows(Tensor out, const Tensor in, const DeviceVector<size_t>& indeces);

void PasteRows(Tensor out, const Tensor in, const DeviceVector<size_t>& indeces);

void CudnnDropoutPrepare(Tensor in, float p,
                         cudnnDropoutDescriptor_t* dropDesc,
                         void** space, size_t* spaceSize,
                         void** states, size_t seed);

void CudnnDropoutDestroy(cudnnDropoutDescriptor_t dropDesc,
                         void* space, void* states);

void CudnnDropoutForward(cudnnDropoutDescriptor_t dropoutDesc,
                  void* space, size_t spaceSize,
                  Tensor out, Tensor in);

void CudnnDropoutBackward(cudnnDropoutDescriptor_t dropoutDesc,
                          void* space, size_t spaceSize,
                          Tensor out, Tensor in);

void Transpose(Tensor out, const Tensor in);

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax);

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax);

void GRUFastForward(Tensor out, const std::vector<Tensor>& inputs);

void GRUFastBackward(std::vector<Tensor>& outputs,
                     const std::vector<Tensor>& inputs,
                     const Tensor adj);

}
