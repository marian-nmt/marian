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
    auto inGpu = static_cast<TensorGPU*>(in.get());
    auto outGpu = static_cast<TensorGPU*>(out.get());
    cudnnSoftmaxForward(cudnnHandle,
                        CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        inGpu->cudnn(),
                        inGpu->data(),
                        &beta,
                        outGpu->cudnn(),
                        outGpu->data());
    cudaDeviceSynchronize();
}

void CudnnLogSoftmax(Tensor out, Tensor in) {
    float alpha = 1, beta = 0;
    auto inGpu = static_cast<TensorGPU*>(in.get());
    auto outGpu = static_cast<TensorGPU*>(out.get());
    cudnnSoftmaxForward(cudnnHandle,
                        CUDNN_SOFTMAX_LOG,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        inGpu->cudnn(),
                        inGpu->data(),
                        &beta,
                        outGpu->cudnn(),
                        outGpu->data());
    cudaDeviceSynchronize();
}

void CudnnSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
    float alpha = 1, beta = 0;
    auto valGpu = static_cast<TensorGPU*>(val.get());
    auto adjGpu = static_cast<TensorGPU*>(adj.get());
    auto gradGpu = static_cast<TensorGPU*>(grad.get());
    cudnnSoftmaxBackward(cudnnHandle,
                        CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        valGpu->cudnn(),
                        valGpu->data(),
                        adjGpu->cudnn(),
                        adjGpu->data(),
                        &beta,
                        gradGpu->cudnn(),
                        gradGpu->data());
    cudaDeviceSynchronize();
}

void CudnnLogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
    float alpha = 1, beta = 0;
    auto valGpu = static_cast<TensorGPU*>(val.get());
    auto adjGpu = static_cast<TensorGPU*>(adj.get());
    auto gradGpu = static_cast<TensorGPU*>(grad.get());
    cudnnSoftmaxBackward(cudnnHandle,
                        CUDNN_SOFTMAX_LOG,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha,
                        valGpu->cudnn(),
                        valGpu->data(),
                        adjGpu->cudnn(),
                        adjGpu->data(),
                        &beta,
                        gradGpu->cudnn(),
                        gradGpu->data());
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
  size_t m = out->shape()[0];
  size_t k = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;
  gSubtractMax<<<blocks, threads, shared>>>(out->data(),
                                            in->data(), m, k);
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
  size_t m = out->shape()[0];
  size_t k = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;
  // Subtract the max rowwise for numerical stability (safe softmax).
  gSubtractMax<<<blocks, threads, shared>>>(out->data(),
                                            in->data(), m, k);
  cudaStreamSynchronize(0);
  gSoftMax<<<blocks, threads, shared>>>(out->data(), m, k);
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
  int m = grad->shape()[0];
  int k = grad->shape()[1];

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gSoftmaxGrad<<<blocks, threads, shared>>>(grad->data(), adj->data(), val->data(),
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
  int m = grad->shape()[0];
  int k = grad->shape()[1];

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gLogSoftmaxGrad<<<blocks, threads, shared>>>(grad->data(),
                                               adj->data(), val->data(),
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

//void Argmax(Tensor* Out, const Tensor* In) {
//  size_t m = In->shape()[0];
//  size_t k = In->shape()[1];
//
//  int blocks = m; //std::min(MAX_BLOCKS, (int) m);
//  int threads = k; //std::min(MAX_THREADS, (int) k);
//  //int shared = sizeof(float) * threads * 2;
//  gArgmax<<<blocks, threads>>>(Out->data(), In->data(), m, k);
//  cudaStreamSynchronize(0);
//}

///////////////////////////////////////////////////////

void Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta) {
  Float alpha = 1.0;

  size_t m = A->shape()[0];
  size_t k = A->shape()[1];
  if(transA)
    std::swap(m, k);

  size_t l = B->shape()[0];
  size_t n = B->shape()[1];
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape()[1];
  size_t ldb = B->shape()[1];
  size_t ldc = B->shape()[1];

  if(transB)
    ldc = B->shape()[0];

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemm(handle, opB, opA,
              n, m, k, &alpha, B->data(), ldb, A->data(), lda, &beta, C->data(), ldc);
}

void Prod(Tensor C, const Tensor A, const Tensor B,
            bool transA, bool transB, Float beta) {

  Prod(cublasHandle, C, A, B, transA, transB, beta);
}

void Sum(Tensor out, const Tensor in, int axis, bool mean) {
  int rows = in->shape()[0];
  int cols = in->shape()[1];

  if(axis == 0) {
    float scale = 1.f;
    if(mean)
      scale = 1.f / rows;

    thrust::device_vector<float> d_ones(rows, scale);
    Tensor ones(new TensorGPU(thrust::raw_pointer_cast(d_ones.data()),
                              {1, rows}));
    Prod(out, ones, in, false, false);
  }
  else if(axis == 1) {
    float scale = 1.f;
    if(mean)
      scale = 1.f / cols;

    thrust::device_vector<float> d_ones(cols, scale);
    Tensor ones(new TensorGPU(thrust::raw_pointer_cast(d_ones.data()),
                              {cols, 1}));
    Prod(out, in, ones, false, false);
  }
  else {
    float scale1 = 1.f;
    float scale2 = 1.f;
    if(mean) {
      scale1 = 1.f / rows;
      scale2 = 1.f / cols;
    }
    thrust::device_vector<float> d_ones1(rows, scale1);
    Tensor ones1(new TensorGPU(thrust::raw_pointer_cast(d_ones1.data()),
                               {1, rows}));
    thrust::device_vector<float> d_ones2(cols, scale2);
    Tensor ones2(new TensorGPU(thrust::raw_pointer_cast(d_ones2.data()),
                               {cols, 1}));
    thrust::device_vector<float> d_temp(cols, 0.f);
    Tensor temp(new TensorGPU(thrust::raw_pointer_cast(d_temp.data()),
                               {1, cols}));

    Prod(temp, ones1, in, false, false);
    Prod(out, temp, ones2, false, false);
  }
}

void SumBackward(Tensor out, const Tensor in, int axis, bool mean) {
  int rows = out->shape()[0];
  int cols = out->shape()[1];

  if(axis == 0) {
    float scale = 1.f;
    if(mean)
      scale = 1.f / rows;

    thrust::device_vector<float> d_ones(rows, scale);
    Tensor ones(new TensorGPU(thrust::raw_pointer_cast(d_ones.data()),
                              {rows, 1}));
    Prod(out, ones, in, false, false);
  }
  else if(axis == 1) {
    float scale = 1.f;
    if(mean)
      scale = 1.f / cols;

    thrust::device_vector<float> d_ones(cols, scale);
    Tensor ones(new TensorGPU(thrust::raw_pointer_cast(d_ones.data()),
                              {1, cols}));
    Prod(out, in, ones, false, false);
  }
  else {
    float scale1 = 1.f;
    float scale2 = 1.f;
    if(mean) {
      scale1 = 1.f / rows;
      scale2 = 1.f / cols;
    }
    thrust::device_vector<float> d_ones1(rows, scale1);
    Tensor ones1(new TensorGPU(thrust::raw_pointer_cast(d_ones1.data()),
                               {rows, 1}));
    thrust::device_vector<float> d_ones2(cols, scale2);
    Tensor ones2(new TensorGPU(thrust::raw_pointer_cast(d_ones2.data()),
                               {1, cols}));
    thrust::device_vector<float> d_temp(rows, 0.f);
    Tensor temp(new TensorGPU(thrust::raw_pointer_cast(d_temp.data()),
                               {rows, 1}));

    Prod(temp, ones1, in, false, false);
    Prod(out, temp, ones2, false, false);
  }
}

void CudnnDropoutPrepare(Tensor in, float p,
                         cudnnDropoutDescriptor_t* dropDesc,
                         void** space, size_t* spaceSize,
                         void** states, size_t seed) {
  size_t statesSize;
  cudnnDropoutGetStatesSize(cudnnHandle, &statesSize);
  auto inGpu = static_cast<TensorGPU*>(in.get());
  cudnnDropoutGetReserveSpaceSize(inGpu->cudnn(), spaceSize);

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
  auto inGpu = static_cast<TensorGPU*>(in.get());
  auto outGpu = static_cast<TensorGPU*>(out.get());
  cudnnDropoutForward(cudnnHandle,
                      dropoutDesc,
                      inGpu->cudnn(),
                      inGpu->data(),
                      outGpu->cudnn(),
                      outGpu->data(),
                      space,
                      spaceSize);
}

void CudnnDropoutBackward(cudnnDropoutDescriptor_t dropoutDesc,
                          void* space, size_t spaceSize,
                          Tensor out, Tensor in) {
  auto inGpu = static_cast<TensorGPU*>(in.get());
  auto outGpu = static_cast<TensorGPU*>(out.get());
  cudnnDropoutBackward(cudnnHandle,
                      dropoutDesc,
                      inGpu->cudnn(),
                      inGpu->data(),
                      outGpu->cudnn(),
                      outGpu->data(),
                      space,
                      spaceSize);
}

__global__ void gCopyRows(float* out, const float* in, size_t cols,
                          const size_t* sourceRowIdx, size_t rows) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      size_t dstId = j;
      size_t srcId = sourceRowIdx[j];

      float* rowOut = out + dstId * cols;
      const float* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

void CopyRows(Tensor out, const Tensor in, const DeviceVector<size_t>& indeces) {
  size_t cols = in->shape()[1];
  size_t rowsToCopy = indeces.size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  gCopyRows<<<blocks, threads>>>(out->data(), in->data(), cols,
                                 thrust::raw_pointer_cast(indeces.data()),
                                 rowsToCopy);
  cudaStreamSynchronize(0);
}

__global__ void gPasteRows(float* out, const float* in, size_t cols,
                          const size_t* targetRowIdx, size_t rows) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      size_t dstId = targetRowIdx[j];
      size_t srcId = j;

      float* rowOut = out + dstId * cols;
      const float* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

void PasteRows(Tensor out, const Tensor in, const DeviceVector<size_t>& indeces) {
  size_t cols = in->shape()[1];
  size_t rowsToCopy = indeces.size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  gPasteRows<<<blocks, threads>>>(out->data(), in->data(), cols,
                                  thrust::raw_pointer_cast(indeces.data()),
                                  rowsToCopy);

  cudaStreamSynchronize(0);
}

void Transpose(Tensor out, const Tensor in) {
  size_t m = in->shape()[0];
  size_t n = in->shape()[1];
  float alpha = 1.0;
  float beta  = 0.0;

  cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, in->data(), n,
              &beta, in->data(), n, out->data(), m);
}

void Concatenate(Tensor out, const std::vector<Tensor>& inputs) {
  size_t offset = 0;
  for(auto in : inputs) {
    cudaMemcpy(out->data() + offset,
               in->data(),
               in->size() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    offset += in->size();
  }
}

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in) {
  size_t offset = 0;
  for(auto out: outputs) {
    cudaMemcpy(out->data(),
               in->data() + offset,
               out->size() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    offset += out->size();
  }
}

__global__ void gGRUFastForward(float* out,
                                const float* state,
                                const float* xW,
                                const float* sU,
                                const float* b,
                                size_t rows, size_t cols) {

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;

      const float* rowState = state + j * cols;
      const float* xWrow = xW + j * cols * 3;
      const float* sUrow = sU + j * cols * 3;


      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float ev1 = expf(-(xWrow[i] + sUrow[i] + b[i]));
          float r = 1.0f / (1.0f + ev1);

          int k = i + cols;
          float ev2 = expf(-(xWrow[k] + sUrow[k] + b[k]));
          float z = 1.0f / (1.0f + ev2);

          int l = i + 2 * cols;
          float h = tanhf(xWrow[l] + sUrow[l] * r + b[l]);

          rowOut[i] = (1.0f - z) * h + z * rowState[i];
        }
      }
    }
  }
}

void GRUFastForward(Tensor out, const std::vector<Tensor>& inputs){
  int rows = out->shape()[0];
  int cols = out->shape()[1];

  int blocks  = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastForward<<<blocks, threads>>>(
    out->data(),
    inputs[0]->data(),
    inputs[1]->data(),
    inputs[2]->data(),
    inputs[3]->data(),
    rows, cols);
}

void GRUFastBackward(std::vector<Tensor>& output, const Tensor in) {
  
}

}
