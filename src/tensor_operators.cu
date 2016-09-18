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

#include "tensor_operators.h"

using namespace std;

namespace marian {

__global__ void gSubtractMean(float* out, float* weights,
                              size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;
      float* sp = out + j * cols;
      float* w = weights + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += w[id] * sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x){
        int id = tid + threadIdx.x;
        if(id < cols)
          sp[id] -= _sum[0];
      }
    }
  }
}

void SubtractMean(Tensor* Out, Tensor &Weights) {
  // Out and Weights are both m-by-k matrices, passed as input.
  // A weighted average of each row of Out (according to the weights
  // specified in Weights) is computed and subtracted from Out.
  // Out is both input and output.
  size_t m = Out->shape()[0];
  size_t k = Out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;
  gSubtractMean<<<blocks, threads, shared>>>(Out->data(), Weights.data(),
                                             m, k);
  cudaStreamSynchronize(0);
}

__global__ void gSubtractMax(float* out, size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if (j < rows) {
      extern __shared__ float _share[];
      float* _max = _share + blockDim.x;
      float* sp = out + j * cols;
      _max[threadIdx.x] = sp[threadIdx.x];
      for(int tid = 1; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          if (sp[id] > _max[threadIdx.x]) _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if (threadIdx.x < (len >> 1)) {
          if (_max[threadIdx.x + skip] > _max[threadIdx.x]) {
             _max[threadIdx.x] = _max[threadIdx.x + skip];
          }
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x){
        int id = tid + threadIdx.x;
        if(id < cols)
          sp[id] -= _max[0];
      }
    }
  }
}

void SubtractMax(Tensor* Out) {
  // Out is a m-by-k matrix, passed as input.
  // The max element of each row of Out is computed and subtracted from Out.
  // Out is both input and output.
  size_t m = Out->shape()[0];
  size_t k = Out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;
  gSubtractMax<<<blocks, threads, shared>>>(Out->data(), m, k);
  cudaStreamSynchronize(0);
}

///////////////////////////////////////////////////////
__global__ void gSoftMax(float* softMaxP, size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;
      float* sp = softMaxP + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sp[id] = __expf(sp[id]);
          _sum[threadIdx.x] += sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x){
        int id = tid + threadIdx.x;
        if(id < cols)
          sp[id] /= _sum[0];
      }
    }
  }
}

void Softmax(Tensor* Out) {
  size_t m = Out->shape()[0];
  size_t k = Out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;
  gSoftMax<<<blocks, threads, shared>>>(Out->data(), m, k);
  cudaStreamSynchronize(0);
}
///////////////////////////////////////////////////////
__global__ void gArgMax(float *out, const float *data, size_t rows, size_t cols) {
  size_t row = blockIdx.x;
    size_t startInd = row * cols;
    float maxScore = -99999;
    size_t maxInd;
    for (size_t col = 0; col < cols; ++col) {
      size_t ind = startInd + col;
      float score = data[ind];
      if (score > maxScore) {
        maxScore = score;
        maxInd = col;
      }
    }
    out[row] = maxInd;
}

void Argmax(Tensor* Out, const Tensor* In) {
  size_t m = In->shape()[0];
  size_t k = In->shape()[1];

  int blocks = m; //std::min(MAX_BLOCKS, (int) m);
  int threads = k; //std::min(MAX_THREADS, (int) k);
  //int shared = sizeof(float) * threads * 2;
  gArgMax<<<blocks, threads>>>(Out->data(), In->data(), m, k);
  cudaStreamSynchronize(0);
}

///////////////////////////////////////////////////////

Tensor Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta) {
  Float alpha = 1.0;

  size_t m = A.shape()[0];
  size_t k = A.shape()[1];
  if(transA)
    std::swap(m, k);
  
  size_t l = B.shape()[0];
  size_t n = B.shape()[1];
  if(transB)
    std::swap(l, n);
  
  size_t lda = A.shape()[1];
  size_t ldb = B.shape()[1];
  size_t ldc = B.shape()[1];
  
  if(transB)
    ldc = B.shape()[0];
  
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  cublasSgemm(handle, opB, opA,
              n, m, k, &alpha, B.data(), ldb, A.data(), lda, &beta, C.data(), ldc);
  return C;
}

Tensor Prod(Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta) {

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);  
  Tensor temp = Prod(cublasHandle, C, A, B, transA, transB, beta);
  cublasDestroy(cublasHandle);
  return temp;
}

}
