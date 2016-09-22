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

namespace marian {

// @TODO: handle this better, maybe per thread?
static cublasHandle_t create_handle() {
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  return cublasHandle;
}

static cudnnHandle_t create_handle_dnn() {
  cudnnHandle_t cudnnHandle;
  cudnnCreate(&cudnnHandle);
  return cudnnHandle;
}

cublasHandle_t cublasHandle = create_handle();
cudnnHandle_t cudnnHandle = create_handle_dnn();

void CudnnSoftmax(Tensor out, Tensor in) {
    float alpha = 1, beta = 0;
    cudnnSoftmaxForward(cudnnHandle,
                        CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        in.cudnn(),
                        in.data(),
                        &beta,
                        out.cudnn(),
                        out.data());
    cudaDeviceSynchronize();
}

void CudnnLogSoftmax(Tensor out, Tensor in) {
    float alpha = 1, beta = 0;
    cudnnSoftmaxForward(cudnnHandle,
                        CUDNN_SOFTMAX_LOG,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        in.cudnn(),
                        in.data(),
                        &beta,
                        out.cudnn(),
                        out.data());
    cudaDeviceSynchronize();
}

void CudnnSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
    float alpha = 1, beta = 0;
    cudnnSoftmaxBackward(cudnnHandle,
                        CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        val.cudnn(),
                        val.data(),
                        adj.cudnn(),
                        adj.data(),
                        &beta,
                        grad.cudnn(),
                        grad.data());
    cudaDeviceSynchronize();
}

void CudnnLogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
    float alpha = 1, beta = 0;
    cudnnSoftmaxBackward(cudnnHandle,
                        CUDNN_SOFTMAX_LOG,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        val.cudnn(),
                        val.data(),
                        adj.cudnn(),
                        adj.data(),
                        &beta,
                        grad.cudnn(),
                        grad.data());
    cudaDeviceSynchronize();
}

__global__ void gSubtractMax(float* out, const float* in,
                             size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if (j < rows) {
      extern __shared__ float _share[];
      float* _max = _share + blockDim.x;
      const float* inRow = in + j * cols;
      float* outRow = out + j * cols;
      _max[threadIdx.x] = inRow[threadIdx.x];
      for(int tid = 1; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          if (in[id] > _max[threadIdx.x]) _max[threadIdx.x] = inRow[id];
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
          outRow[id] = inRow[id] - _max[0];
      }
    }
  }
}

void SubtractMax(Tensor out, Tensor in) {
  // Out is a m-by-k matrix, passed as input.
  // The max element of each row of Out is computed and subtracted from Out.
  // Out is both input and output.
  size_t m = out.shape()[0];
  size_t k = out.shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;
  gSubtractMax<<<blocks, threads, shared>>>(out.data(),
                                            in.data(), m, k);
  cudaStreamSynchronize(0);
}

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

void Softmax(Tensor out, Tensor in) {
  size_t m = out.shape()[0];
  size_t k = out.shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;
  // Subtract the max rowwise for numerical stability (safe softmax).
  gSubtractMax<<<blocks, threads, shared>>>(out.data(),
                                            in.data(), m, k);
  cudaStreamSynchronize(0);
  gSoftMax<<<blocks, threads, shared>>>(out.data(), m, k);
  cudaStreamSynchronize(0);
}

///////////////////////////////////////////////////////

__global__ void gSoftmaxGrad(float* grad, const float* adj, const float* val,
                             const int rows, const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      float* gradRow = grad + j * cols;
      const float* adjRow = adj + j * cols;
      const float* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += valRow[id] * adjRow[id];
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
          gradRow[id] += valRow[id] * (adjRow[id] - _sum[0]);
      }
    }
  }
}

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad.shape()[0];
  int k = grad.shape()[1];

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gSoftmaxGrad<<<blocks, threads, shared>>>(grad.data(), adj.data(), val.data(),
                                            m, k);
  cudaStreamSynchronize(0);
}

__global__ void gLogSoftmaxGrad(float* grad, const float* adj, const float* val,
                                const int rows, const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      float* gradRow = grad + j * cols;
      const float* adjRow = adj + j * cols;
      const float* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += adjRow[id];
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
          gradRow[id] += adjRow[id] - (expf(valRow[id]) * _sum[0]);
      }
    }
  }
}

void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad.shape()[0];
  int k = grad.shape()[1];

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gLogSoftmaxGrad<<<blocks, threads, shared>>>(grad.data(),
                                               adj.data(), val.data(),
                                               m, k);
  cudaStreamSynchronize(0);
}

///////////////////////////////////////////////////////
__global__ void gArgmax(float *out, const float *data, size_t rows, size_t cols) {
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
  gArgmax<<<blocks, threads>>>(Out->data(), In->data(), m, k);
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

  Tensor temp = Prod(cublasHandle, C, A, B, transA, transB, beta);
  return temp;
}

Tensor SumRowwise(cublasHandle_t handle, const Tensor A, Tensor result) {
  size_t rows = A.shape()[0];
  size_t cols = A.shape()[1];
  thrust::device_vector<float> d_ones(cols, 1.f);
  Float alpha = 1.f;
  Float beta  = 0.f;
  cublasSgemv(handle, CUBLAS_OP_T, cols, rows, &alpha,
              A.data(), cols,
              thrust::raw_pointer_cast(d_ones.data()), 1, &beta,
              result.data(), 1);
  return result;
}

Tensor SumRowwise(const Tensor A, Tensor result) {
  Tensor temp = SumRowwise(cublasHandle, A, result);
  return temp;
}

void CudnnDropoutPrepare(Tensor in, float p,
                         cudnnDropoutDescriptor_t* dropDesc,
                         void** space, size_t* spaceSize,
                         void** states, size_t seed) {
  size_t statesSize;
  cudnnDropoutGetStatesSize(cudnnHandle, &statesSize);
  cudnnDropoutGetReserveSpaceSize(in.cudnn(), spaceSize);

  cudaMalloc((void**)states, statesSize);
  cudaMalloc((void**)space, *spaceSize);

  cudnnCreateDropoutDescriptor(dropDesc);
  cudnnSetDropoutDescriptor(*dropDesc,
                            cudnnHandle,
                            p,
                            (void*)*states,
                            statesSize,
                            seed);
}

void CudnnDropoutDestroy(cudnnDropoutDescriptor_t dropDesc,
                         void* space, void* states) {
  cudnnDestroyDropoutDescriptor(dropDesc);
  cudaFree(space);
  cudaFree(states);
}

void CudnnDropoutForward(cudnnDropoutDescriptor_t dropoutDesc,
                  void* space, size_t spaceSize,
                  Tensor out, Tensor in) {
  cudnnDropoutForward(cudnnHandle,
                      dropoutDesc,
                      in.cudnn(),
                      in.data(),
                      out.cudnn(),
                      out.data(),
                      space,
                      spaceSize);
}

void CudnnDropoutBackward(cudnnDropoutDescriptor_t dropoutDesc,
                          void* space, size_t spaceSize,
                          Tensor out, Tensor in) {
  cudnnDropoutBackward(cudnnHandle,
                      dropoutDesc,
                      in.cudnn(),
                      in.data(),
                      out.cudnn(),
                      out.data(),
                      space,
                      spaceSize);
}

}
