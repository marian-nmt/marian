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

__global__ void gSoftmax(float* out,
                         const Shape outShape,
                         const float* in,
                         const float* mask) {
  int rows = outShape[0];
  int cols = outShape[1];
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;
      const float* mp = mask ? (mask + j * cols) : 0;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
      _max[threadIdx.x] = sp[threadIdx.x]; // mask
      for(int tid = 1; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          if (sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
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
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = 0;
          if(!mask || mp[id])
            ex = __expf(sp[id] - max);
          so[id] = ex;
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
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
          so[id] /= _sum[0];
      }
    }
  }
}

void Softmax(Tensor out, Tensor in, Tensor mask) {
  size_t m = out->shape()[0] * out->shape()[2] * out->shape()[3];
  size_t k = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;

  if(mask)
    gSoftmax<<<blocks, threads, shared>>>(out->data(),
                                          out->shape(),
                                          in->data(),
                                          mask->data());
  else
    gSoftmax<<<blocks, threads, shared>>>(out->data(),
                                          out->shape(),
                                          in->data(),
                                          0);
  cudaStreamSynchronize(0);
}

__global__ void gLogSoftmax(float* out,
                            const Shape outShape,
                            const float* in) {
  int rows = outShape[0];
  int cols = outShape[1];
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
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
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float sm = sp[id] - max;
          float ex = __expf(sm);
          so[id] = sm;
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
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
          so[id] -= __logf(_sum[0]);
      }
    }
  }
}

void LogSoftmax(Tensor out, Tensor in) {
  size_t m = out->shape()[0];
  size_t k = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;

  gLogSoftmax<<<blocks, threads, shared>>>(out->data(),
                                           out->shape(),
                                           in->data());

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
  int m = grad->shape()[0] * grad->shape()[2] * grad->shape()[3];
  int k = grad->shape()[1];

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gSoftmaxGrad<<<blocks, threads, shared>>>(grad->data(),
                                            adj->data(),
                                            val->data(),
                                            m, k);

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
//
//}

///////////////////////////////////////////////////////

void Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, Float beta) {
  Float alpha = 1.0;

  size_t m = A->shape()[0] * A->shape()[2] * A->shape()[3];
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
          atomicAdd(rowOut + i, rowIn[i]);
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


}

void Transpose(Tensor out, const Tensor in) {
  size_t m = in->shape()[0];
  size_t n = in->shape()[1];
  float alpha = 1.0;
  float beta  = 0.0;

  cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, in->data(), n,
              &beta, in->data(), n, out->data(), m);
}

void Concatenate0(Tensor out, const std::vector<Tensor>& inputs) {
  size_t offset = 0;
  for(auto in : inputs) {
    UTIL_THROW_IF2(out->shape()[1] != in->shape()[1],
                   "Second dimension must be equal");
    cudaMemcpy(out->data() + offset,
               in->data(),
               in->size() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    offset += in->size();
  }
}

__global__ void gInsertCols(float* out, const float* in,
                            size_t rows, size_t cols,
                            size_t cols_out, size_t cols_in,
                            size_t offset_out, size_t offset_in) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols_out + offset_out;
      const float* rowIn = in + j * cols_in + offset_in;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

// this probably does not work for tensors with more than 2
// dimensions, verify this!
void Concatenate1(Tensor out, const std::vector<Tensor>& inputs) {
  size_t offset = 0;
  int rows = out->shape()[0];
  int cols_out = out->shape()[1];

  for(auto in : inputs) {
    UTIL_THROW_IF2(out->shape()[0] != in->shape()[0],
                   "First dimension must be equal");
    int cols_in = in->shape()[1];

    int blocks  = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_in);

    gInsertCols<<<blocks, threads>>>(
      out->data(),
      in->data(),
      rows, cols_in,
      cols_out, cols_in,
      offset, 0);
    offset += cols_in;
  }
}

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax) {
  if(ax == 1)
    Concatenate1(out, inputs);
  else
    Concatenate0(out, inputs);
}

void Deconcatenate0(std::vector<Tensor>& outputs, const Tensor in) {
  size_t offset = 0;
  for(auto out : outputs) {
    cudaMemcpy(out->data(),
               in->data() + offset,
               out->size() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    offset += out->size();
  }
}

void Deconcatenate1(std::vector<Tensor>& outputs, const Tensor in) {
  size_t offset = 0;
  int rows = in->shape()[0];
  int cols_in = in->shape()[1];
  for(auto out : outputs) {
    UTIL_THROW_IF2(out->shape()[0] != in->shape()[0],
                   "First dimension must be equal");
    int cols_out = out->shape()[1];

    int blocks  = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_out);

    gInsertCols<<<blocks, threads>>>(
      out->data(),
      in->data(),
      rows, cols_out,
      cols_out, cols_in,
      0, offset);
    offset += cols_out;
  }
}

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax) {
  if(ax == 1)
    Deconcatenate1(outputs, in);
  else
    Deconcatenate0(outputs, in);
}

__global__ void gGRUFastForward(float* out,
                                const float* state,
                                const float* xW,
                                const float* sU,
                                const float* b,
                                const float* mask,
                                size_t rows, size_t cols,
                                bool final) {

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];
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
          float h;
          if(final)
            h = tanhf(xWrow[l] + (sUrow[l] + b[l]) * r);
          else
            h = tanhf(xWrow[l] + sUrow[l] * r + b[l]);

          float out = (1.0f - z) * h + z * rowState[i];
          rowOut[i] = m * out + (1 - m) * rowState[i];
        }
      }
    }
  }
}

void GRUFastForward(Tensor out, const std::vector<Tensor>& inputs, bool final){
  int rows = out->shape()[0];
  int cols = out->shape()[1];

  int blocks  = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastForward<<<blocks, threads>>>(
    out->data(), // output
    inputs[0]->data(), // state
    inputs[1]->data(), // xW
    inputs[2]->data(), // sU
    inputs[3]->data(), // b
    inputs.size() > 4 ? inputs[4]->data() : 0, // mask
    rows, cols, final);
}

__global__ void gGRUFastBackward(float* outState,
                                 float* outXW,
                                 float* outSU,
                                 float* outB,
                                 const float* state,
                                 const float* xW,
                                 const float* sU,
                                 const float* b,
                                 const float* mask,
                                 const float* adj,
                                 size_t rows, size_t cols,
                                 bool final) {

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOutState = outState + j * cols;
      float* rowOutXW = outXW + j * cols * 3;
      float* rowOutSU = outSU + j * cols * 3;

      const float* rowState = state + j * cols;
      const float* rowXW = xW + j * cols * 3;
      const float* rowSU = sU + j * cols * 3;
      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + cols;
          int l = i + 2 * cols;

          float ev1 = expf(-(rowXW[i] + rowSU[i] + b[i]));
          float r = 1.0f / (1.0f + ev1);

          float ev2 = expf(-(rowXW[k] + rowSU[k] + b[k]));
          float z = 1.0f / (1.0f + ev2);

          float h;
          if(final)
            h = tanhf(rowXW[l] + (rowSU[l] + b[l]) * r);
          else
            h = tanhf(rowXW[l] + rowSU[l] * r + b[l]);

          float adj = rowAdj[i];

          float t = (1-z)*(1-h*h);

          // df/ds
          if(outState) rowOutState[i] += m * z * adj - m + 1;

          // df/d(xW_r) ...
          float dfdxW_r = r * (1-r) * t * adj;
          if(final)
            dfdxW_r *= rowSU[l] + b[l];
          else
            dfdxW_r *= rowSU[l];
          if(outXW) rowOutXW[i] += m * dfdxW_r;
          if(outSU) rowOutSU[i] += m * dfdxW_r;
          if(outB)  atomicAdd(outB + i, m * dfdxW_r);

          // df/d(xW_z) ...
          float dfdxW_z = (1-z)*z*(rowState[i]-h) * adj;
          if(outXW) rowOutXW[k] += m * dfdxW_z;
          if(outSU) rowOutSU[k] += m * dfdxW_z;
          if(outB)  atomicAdd(outB + k, m * dfdxW_z);

          // df/d(xW_x) ...
          float dfdxW_x = t * adj;
          if(outXW) rowOutXW[l] += m * dfdxW_x;
          if(outSU) rowOutSU[l] += m * dfdxW_x * r;
          if(final)
            if(outB) atomicAdd(outB + l, m * dfdxW_x * r);
          else
            if(outB) atomicAdd(outB + l, m * dfdxW_x);
        }
      }
    }
  }
}

void GRUFastBackward(std::vector<Tensor>& outputs,
                     const std::vector<Tensor>& inputs,
                     const Tensor adj, bool final) {
  int rows = adj->shape()[0];
  int cols = adj->shape()[1];

  int blocks  = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastBackward<<<blocks, threads>>>(
    outputs[0] ? outputs[0]->data() : nullptr, // state - adj
    outputs[1] ? outputs[1]->data() : nullptr, // xW - adj
    outputs[2] ? outputs[2]->data() : nullptr, // sU - adj
    outputs[3] ? outputs[3]->data() : nullptr, // b - adj
    inputs[0]->data(), // state
    inputs[1]->data(), // xW
    inputs[2]->data(), // sU
    inputs[3]->data(), // b
    inputs.size() > 4 ? inputs[4]->data() : 0, // mask
    adj->data(),
    rows, cols, final);
}

__global__ void gCrossEntropyPick(float* out,
                                  const Shape outShape,
                                  const float* in,
                                  const Shape inShape,
                                  const float* pick) {

  int rows = inShape[0];
  int cols = inShape[1];
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* sp = in + j * cols;

      extern __shared__ float _share[];
      float* _max = _share + blockDim.x;

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
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += __expf(sp[id] - max);
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();

      // cross-entropy
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id == (int)pick[j]) {
          out[j] = __logf(_sum[0]) - sp[id] + max;
        }
      }
    }
  }
}

void CrossEntropyPick(Tensor out, Tensor in, Tensor pick) {
  size_t m = in->shape()[0];
  size_t k = in->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;

  gCrossEntropyPick<<<blocks, threads, shared>>>(out->data(),
                                                 out->shape(),
                                                 in->data(),
                                                 in->shape(),
                                                 pick->data());

}

__global__ void gCrossEntropyPickBackward(float* out,
                                          const Shape outShape,
                                          const float* adj,
                                          const float* in,
                                          const float* pick) {

  int rows = outShape[0];
  int cols = outShape[1];
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* sp = in + j * cols;
      float* so = out + j * cols;

      extern __shared__ float _share[];
      float* _max = _share + blockDim.x;

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
      float max = _max[0];
      __syncthreads();

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = __expf(sp[id] - max);
          _sum[threadIdx.x] += ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();

      // cross-entropy
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float sub = (float)(id == (int)pick[j]);
          so[id] += adj[j] * (__expf(sp[id] - max) / _sum[0] - sub);
        }
      }
    }
  }
}

void CrossEntropyPickBackward(Tensor out, Tensor adj, Tensor a, Tensor pick) {
  size_t m = out->shape()[0];
  size_t k = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int) m);
  int threads = std::min(MAX_THREADS, (int) k);
  int shared = sizeof(float) * threads * 2;

  gCrossEntropyPickBackward<<<blocks, threads, shared>>>(out->data(),
                                                         out->shape(),
                                                         adj->data(),
                                                         a->data(),
                                                         pick->data());
}


}
