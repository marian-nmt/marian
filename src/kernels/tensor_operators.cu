
#include "kernels/cuda_helpers.h"
#include "kernels/tensor_operators.h"

#include "3rd_party/reduce_all.h"

namespace marian {

// static cudnnHandle_t create_handle_dnn() {
//  cudnnHandle_t cudnnHandle;
//  cudnnCreate(&cudnnHandle);
//  return cudnnHandle;
//}

// cudnnHandle_t cudnnHandle = create_handle_dnn();

// void CudnnSoftmax(Tensor out, Tensor in) {
//    float alpha = 1, beta = 0;
//    auto inGpu = static_cast<TensorGPU*>(in.get());
//    auto outGpu = static_cast<TensorGPU*>(out.get());
//    cudnnSoftmaxForward(cudnnHandle,
//                        CUDNN_SOFTMAX_ACCURATE,
//                        CUDNN_SOFTMAX_MODE_CHANNEL,
//                        &alpha,
//                        inGpu->cudnn(),
//                        inGpu->data(),
//                        &beta,
//                        outGpu->cudnn(),
//                        outGpu->data());
//    cudaDeviceSynchronize();
//}
//
// void CudnnLogSoftmax(Tensor out, Tensor in) {
//    float alpha = 1, beta = 0;
//    auto inGpu = static_cast<TensorGPU*>(in.get());
//    auto outGpu = static_cast<TensorGPU*>(out.get());
//    cudnnSoftmaxForward(cudnnHandle,
//                        CUDNN_SOFTMAX_LOG,
//                        CUDNN_SOFTMAX_MODE_CHANNEL,
//                        &alpha,
//                        inGpu->cudnn(),
//                        inGpu->data(),
//                        &beta,
//                        outGpu->cudnn(),
//                        outGpu->data());
//    cudaDeviceSynchronize();
//}
//
// void CudnnSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
//    float alpha = 1, beta = 0;
//    auto valGpu = static_cast<TensorGPU*>(val.get());
//    auto adjGpu = static_cast<TensorGPU*>(adj.get());
//    auto gradGpu = static_cast<TensorGPU*>(grad.get());
//    cudnnSoftmaxBackward(cudnnHandle,
//                        CUDNN_SOFTMAX_ACCURATE,
//                        CUDNN_SOFTMAX_MODE_CHANNEL,
//                        &alpha,
//                        valGpu->cudnn(),
//                        valGpu->data(),
//                        adjGpu->cudnn(),
//                        adjGpu->data(),
//                        &beta,
//                        gradGpu->cudnn(),
//                        gradGpu->data());
//    cudaDeviceSynchronize();
//}
//
// void CudnnLogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
//    float alpha = 1, beta = 0;
//    auto valGpu = static_cast<TensorGPU*>(val.get());
//    auto adjGpu = static_cast<TensorGPU*>(adj.get());
//    auto gradGpu = static_cast<TensorGPU*>(grad.get());
//    cudnnSoftmaxBackward(cudnnHandle,
//                        CUDNN_SOFTMAX_LOG,
//                        CUDNN_SOFTMAX_MODE_CHANNEL,
//                        &alpha,
//                        valGpu->cudnn(),
//                        valGpu->data(),
//                        adjGpu->cudnn(),
//                        adjGpu->data(),
//                        &beta,
//                        gradGpu->cudnn(),
//                        gradGpu->data());
//    cudaDeviceSynchronize();
//}

__global__ void gSoftmax(float* out,
                         const ShapeGPU outShape,
                         const float* in,
                         const float* mask) {
  int rows = outShape[0] * outShape[2] * outShape[3];
  int cols = outShape[1];
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;
      const float* mp = mask ? (mask + j * cols) : 0;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
      _max[threadIdx.x] = sp[threadIdx.x];  // mask
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
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
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          so[id] /= _sum[0];
      }
    }
  }
}

void Softmax(Tensor out, Tensor in, Tensor mask) {
  cudaSetDevice(out->getDevice());

  size_t m = out->shape()[0] * out->shape()[2] * out->shape()[3];
  size_t k = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  if(mask)
    gSoftmax<<<blocks, threads, shared>>>(
        out->data(), out->shape(), in->data(), mask->data());
  else
    gSoftmax<<<blocks, threads, shared>>>(
        out->data(), out->shape(), in->data(), 0);
  // cudaStreamSynchronize(0);
}

__global__ void gLogSoftmax(float* out,
                            const ShapeGPU outShape,
                            const float* in) {
  int rows = outShape[0] * outShape[2] * outShape[3];
  int cols = outShape[1];
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      extern __shared__ float _share[];

      float* _max = _share + blockDim.x;
      _max[threadIdx.x] = sp[threadIdx.x];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
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
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          so[id] -= __logf(_sum[0]);
      }
    }
  }
}

void LogSoftmax(Tensor out, Tensor in) {
  cudaSetDevice(out->getDevice());

  size_t m = out->shape()[0] * out->shape()[2] * out->shape()[3];
  size_t k = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  gLogSoftmax<<<blocks, threads, shared>>>(
      out->data(), out->shape(), in->data());
}

///////////////////////////////////////////////////////

__global__ void gSoftmaxGrad(float* grad,
                             const float* adj,
                             const float* val,
                             const int rows,
                             const int cols) {
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
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float val = valRow[id] * (adjRow[id] - _sum[0]);
          if(val)
            gradRow[id] += val;
        }
      }
    }
  }
}

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  cudaSetDevice(adj->getDevice());
  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad->shape()[0] * grad->shape()[2] * grad->shape()[3];
  int k = grad->shape()[1];

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gSoftmaxGrad<<<blocks, threads, shared>>>(
      grad->data(), adj->data(), val->data(), m, k);
}

__global__ void gLogSoftmaxGrad(float* grad,
                                const float* adj,
                                const float* val,
                                const int rows,
                                const int cols) {
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
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          gradRow[id] += adjRow[id] - (expf(valRow[id]) * _sum[0]);
      }
    }
  }
}

void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  cudaSetDevice(adj->getDevice());

  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad->shape()[0] * grad->shape()[2] * grad->shape()[3];
  int k = grad->shape()[1];

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads * 2;
  gLogSoftmaxGrad<<<blocks, threads, shared>>>(
      grad->data(), adj->data(), val->data(), m, k);
}

///////////////////////////////////////////////////////
__global__ void gArgmax(float* out,
                        const float* data,
                        size_t rows,
                        size_t cols) {
  size_t row = blockIdx.x;
  size_t startInd = row * cols;
  float maxScore = -99999;
  size_t maxInd;
  for(size_t col = 0; col < cols; ++col) {
    size_t ind = startInd + col;
    float score = data[ind];
    if(score > maxScore) {
      maxScore = score;
      maxInd = col;
    }
  }
  out[row] = maxInd;
}

// void Argmax(Tensor* Out, const Tensor* In) {
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

void Prod(cublasHandle_t handle,
          Tensor C,
          const Tensor A,
          const Tensor B,
          bool transA,
          bool transB,
          float beta) {
  cudaSetDevice(C->getDevice());
  float alpha = 1.0;

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

  cublasSgemm(handle,
              opB,
              opA,
              n,
              m,
              k,
              &alpha,
              B->data(),
              ldb,
              A->data(),
              lda,
              &beta,
              C->data(),
              ldc);
}

__global__ void gCopyRows(float* out,
                          const float* in,
                          size_t cols,
                          const size_t* sourceRowIdx,
                          size_t rows) {
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

void CopyRows(Tensor out, const Tensor in, const std::vector<size_t>& indeces) {
  cudaSetDevice(out->getDevice());

  size_t cols = in->shape()[1];
  size_t rowsToCopy = indeces.size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  size_t* d_indeces;
  CUDA_CHECK(cudaMalloc(&d_indeces, rowsToCopy * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indeces,
                        indeces.data(),
                        rowsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  gCopyRows<<<blocks, threads>>>(
      out->data(), in->data(), cols, d_indeces, rowsToCopy);

  CUDA_CHECK(cudaFree(d_indeces));
}

__global__ void gPasteRows(float* out,
                           const float* in,
                           size_t cols,
                           const size_t* targetRowIdx,
                           size_t rows) {
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

void PasteRows(Tensor out,
               const Tensor in,
               const std::vector<size_t>& indeces) {
  cudaSetDevice(out->getDevice());

  size_t cols = in->shape()[1];
  size_t rowsToCopy = indeces.size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  // @TODO: turn into tensor
  size_t* d_indeces;
  CUDA_CHECK(cudaMalloc(&d_indeces, rowsToCopy * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indeces,
                        indeces.data(),
                        rowsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  gPasteRows<<<blocks, threads>>>(
      out->data(), in->data(), cols, d_indeces, rowsToCopy);
  CUDA_CHECK(cudaFree(d_indeces));
}

/////////////

__global__ void gCopyCols(float* out,
                          const float* in,
                          size_t rows,
                          size_t colsIn,
                          const size_t* sourceColIdx,
                          size_t colsOut) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* rowIn = in + j * colsIn;
      float* rowOut = out + j * colsOut;

      for(int tid = 0; tid < colsOut; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsOut)
          rowOut[i] = rowIn[sourceColIdx[i]];
      }
    }
  }
}

void CopyCols(Tensor out, const Tensor in, const std::vector<size_t>& indeces) {
  cudaSetDevice(out->getDevice());

  size_t rows = in->shape()[0] * in->shape()[2] * in->shape()[3];
  size_t cols = in->shape()[1];
  size_t colsToCopy = indeces.size();

  int threads = std::min(MAX_THREADS, (int)colsToCopy);
  int blocks = std::min(MAX_BLOCKS, (int)rows);

  size_t* d_indeces;
  CUDA_CHECK(cudaMalloc(&d_indeces, colsToCopy * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indeces,
                        indeces.data(),
                        colsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  gCopyCols<<<blocks, threads>>>(
      out->data(), in->data(), rows, cols, d_indeces, colsToCopy);

  CUDA_CHECK(cudaFree(d_indeces));
}

__global__ void gPasteCols(float* out,
                           const float* in,
                           size_t rows,
                           size_t colsOut,
                           const size_t* targetColIdx,
                           size_t colsIn) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* rowIn = in + j * colsIn;
      float* rowOut = out + j * colsOut;

      for(int tid = 0; tid < colsIn; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsIn)
          rowOut[targetColIdx[i]] = rowIn[i];
      }
    }
  }
}

void PasteCols(Tensor out,
               const Tensor in,
               const std::vector<size_t>& indeces) {
  cudaSetDevice(out->getDevice());

  size_t rows = out->shape()[0] * out->shape()[2] * out->shape()[3];
  size_t cols = out->shape()[1];
  size_t colsToCopy = indeces.size();

  int threads = std::min(MAX_THREADS, (int)colsToCopy);
  int blocks = std::min(MAX_BLOCKS, (int)rows);

  size_t* d_indeces;
  CUDA_CHECK(cudaMalloc(&d_indeces, colsToCopy * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indeces,
                        indeces.data(),
                        colsToCopy * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  gPasteCols<<<blocks, threads>>>(
      out->data(), in->data(), rows, cols, d_indeces, colsToCopy);

  CUDA_CHECK(cudaFree(d_indeces));
}
//////////////

void Transpose(cublasHandle_t cublasHandle, Tensor out, const Tensor in) {
  cudaSetDevice(out->getDevice());
  size_t steps = in->shape()[2] * in->shape()[3];
  for(int i = 0; i < steps; i++) {
    size_t m = in->shape()[0];
    size_t n = in->shape()[1];
    float alpha = 1.0;
    float beta = 0.0;

    size_t offset = i * m * n;

    cublasSgeam(cublasHandle,
                CUBLAS_OP_T,
                CUBLAS_OP_T,
                m,
                n,
                &alpha,
                in->data() + offset,
                n,
                &beta,
                in->data() + offset,
                n,
                out->data() + offset,
                m);
  }
}

void Concatenate0(Tensor out, const std::vector<Tensor>& inputs) {
  cudaSetDevice(out->getDevice());

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

__global__ void gInsertCols(float* out,
                            const float* in,
                            size_t rows,
                            size_t cols,
                            size_t cols_out,
                            size_t cols_in,
                            size_t offset_out,
                            size_t offset_in) {
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

__global__ void gConcatenateAx1(float* out,
                                size_t rows,
                                const float* in1,
                                const float* in2,
                                size_t colsIn1,
                                size_t colsIn2) {
  size_t cols = colsIn1 + colsIn2;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn1 = in1 + j * colsIn1;
      const float* rowIn2 = in2 + j * colsIn2;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsIn1)
          rowOut[i] = rowIn1[i];
        else if(i >= colsIn1 && i < colsIn1 + colsIn2)
          rowOut[i] = rowIn2[i - colsIn1];
      }
    }
  }
}

__global__ void gConcatenateAx1(float* out,
                                size_t rows,
                                const float* in1,
                                const float* in2,
                                const float* in3,
                                size_t colsIn1,
                                size_t colsIn2,
                                size_t colsIn3) {
  size_t cols = colsIn1 + colsIn2 + colsIn3;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn1 = in1 + j * colsIn1;
      const float* rowIn2 = in2 + j * colsIn2;
      const float* rowIn3 = in3 + j * colsIn3;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsIn1)
          rowOut[i] = rowIn1[i];
        else if(i >= colsIn1 && i < colsIn1 + colsIn2)
          rowOut[i] = rowIn2[i - colsIn1];
        else if(i >= colsIn1 + colsIn2 && i < colsIn1 + colsIn2 + colsIn3)
          rowOut[i] = rowIn3[i - colsIn1 - colsIn2];
      }
    }
  }
}

void Concatenate1(Tensor out, const std::vector<Tensor>& inputs) {
  cudaSetDevice(out->getDevice());

  int rows = out->shape()[0] * out->shape()[2] * out->shape()[3];
  if(inputs.size() == 2) {
    int cols = out->shape()[1];
    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols);
    gConcatenateAx1<<<blocks, threads>>>(out->data(),
                                         rows,
                                         inputs[0]->data(),
                                         inputs[1]->data(),
                                         inputs[0]->shape()[1],
                                         inputs[1]->shape()[1]);
    return;
  }
  if(inputs.size() == 3) {
    int cols = out->shape()[1];
    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols);
    gConcatenateAx1<<<blocks, threads>>>(out->data(),
                                         rows,
                                         inputs[0]->data(),
                                         inputs[1]->data(),
                                         inputs[2]->data(),
                                         inputs[0]->shape()[1],
                                         inputs[1]->shape()[1],
                                         inputs[2]->shape()[1]);
    return;
  }

  size_t offset = 0;
  int cols_out = out->shape()[1];

  for(auto in : inputs) {
    UTIL_THROW_IF2(out->shape()[0] != in->shape()[0],
                   "First dimension must be equal");
    int cols_in = in->shape()[1];

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_in);

    gInsertCols<<<blocks, threads>>>(
        out->data(), in->data(), rows, cols_in, cols_out, cols_in, offset, 0);
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
  cudaSetDevice(in->getDevice());

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
  cudaSetDevice(in->getDevice());

  size_t offset = 0;
  int rows = in->shape()[0] * in->shape()[2] * in->shape()[3];
  int cols_in = in->shape()[1];
  for(auto out : outputs) {
    UTIL_THROW_IF2(out->shape()[0] != in->shape()[0],
                   "First dimension must be equal");
    int cols_out = out->shape()[1];

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_out);

    gInsertCols<<<blocks, threads>>>(
        out->data(), in->data(), rows, cols_out, cols_out, cols_in, 0, offset);
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
                                size_t rows,
                                size_t cols,
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

void GRUFastForward(Tensor out, std::vector<Tensor> inputs, bool final) {
  cudaSetDevice(out->getDevice());

  int rows = out->shape()[0] * out->shape()[2] * out->shape()[3];
  int cols = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastForward<<<blocks, threads>>>(
      out->data(),                                // output
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      rows,
      cols,
      final);
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
                                 size_t rows,
                                 size_t cols,
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

          float t = (1 - z) * (1 - h * h);

          // df/ds
          if(outState)
            rowOutState[i] += (m * z - m + 1) * adj;

          // df/d(xW_r) ...
          float dfdxW_r = m * r * (1 - r) * t * adj;
          if(final)
            dfdxW_r *= rowSU[l] + b[l];
          else
            dfdxW_r *= rowSU[l];
          if(outXW)
            rowOutXW[i] += dfdxW_r;
          if(outSU)
            rowOutSU[i] += dfdxW_r;
          if(outB)
            atomicAdd(outB + i, dfdxW_r);

          // df/d(xW_z) ...
          float dfdxW_z = m * (1 - z) * z * (rowState[i] - h) * adj;
          if(outXW)
            rowOutXW[k] += dfdxW_z;
          if(outSU)
            rowOutSU[k] += dfdxW_z;
          if(outB)
            atomicAdd(outB + k, dfdxW_z);

          // df/d(xW_x) ...
          float dfdxW_x = m * t * adj;
          if(outXW)
            rowOutXW[l] += dfdxW_x;
          if(outSU)
            rowOutSU[l] += dfdxW_x * r;
          if(outB)
            if(final)
              atomicAdd(outB + l, dfdxW_x * r);
            else
              atomicAdd(outB + l, dfdxW_x);
        }
      }
    }
  }
}

void GRUFastBackward(std::vector<Tensor> outputs,
                     std::vector<Tensor> inputs,
                     Tensor adj,
                     bool final) {
  cudaSetDevice(adj->getDevice());

  int rows = adj->shape()[0] * adj->shape()[2] * adj->shape()[3];
  int cols = adj->shape()[1];

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gGRUFastBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,        // state - adj
      outputs[1] ? outputs[1]->data() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data() : 0,        // b - adj
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      adj->data(),
      rows,
      cols,
      final);
}

__global__ void gCrossEntropyPick(float* out,
                                  const ShapeGPU outShape,
                                  const float* in,
                                  const ShapeGPU inShape,
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
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
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
  cudaSetDevice(out->getDevice());

  size_t m = in->shape()[0] * in->shape()[2] * in->shape()[3];
  size_t k = in->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  gCrossEntropyPick<<<blocks, threads, shared>>>(
      out->data(), out->shape(), in->data(), in->shape(), pick->data());
}

__global__ void gCrossEntropyPickBackward(float* out,
                                          const ShapeGPU outShape,
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
        if(id < cols) {
          if(sp[id] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          if(_max[threadIdx.x + skip] > _max[threadIdx.x]) {
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
  cudaSetDevice(out->getDevice());

  size_t m = out->shape()[0] * out->shape()[2] * out->shape()[3];
  size_t k = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  gCrossEntropyPickBackward<<<blocks, threads, shared>>>(
      out->data(), out->shape(), adj->data(), a->data(), pick->data());
}

float L2Norm(Tensor in) {
  cudaSetDevice(in->getDevice());

  uint8_t* data;
  cudaMalloc(&data, sizeof(float));
  Tensor out(new TensorBase(New<MemoryPiece>(data, sizeof(float)), {1, 1}, in->getDevice()));
  ReduceAll(_1 * _1, out, in);
  float dataCpu = sqrtf(out->get(0));
  out.reset();
  cudaFree(data);
  return dataCpu;
}

__global__ void gAtt(float* out,
                     const float* va,
                     const float* ctx,
                     const float* state,
                     const float* cov,
                     int m,  // total rows (batch x time x beam)
                     int k,  // depth
                     int b,  // batch size
                     int t   // time of ctx
                     ) {
  int rows = m;
  int cols = k;
  for(int bid = 0; bid < m; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const float* vaRow = va;
      const float* ctxRow = ctx + (j % (b * t)) * cols;
      const float* stateRow = state + (j / (b * t) + j % b) * cols;
      const float* covRow = cov ? cov + (j % (b * t)) * cols : nullptr;

      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float z = ctxRow[id] + stateRow[id];
          if(cov)
            z += covRow[id];
          float ex = tanhf(z) * vaRow[id];
          _sum[threadIdx.x] += ex;
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
      out[j] = _sum[0];
    }
  }
}

void Att(Tensor out, Tensor va, Tensor context, Tensor state, Tensor coverage) {
  cudaSetDevice(out->getDevice());

  size_t m = out->shape()[0] * out->shape()[2] * out->shape()[3];

  size_t b = context->shape()[0];
  size_t k = context->shape()[1];
  size_t t = context->shape()[2];

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  gAtt<<<blocks, threads, shared>>>(out->data(),
                                    va->data(),
                                    context->data(),
                                    state->data(),
                                    coverage ? coverage->data() : nullptr,
                                    m,
                                    k,
                                    b,
                                    t);
}

__global__ void gAttBack(float* gVa,
                         float* gContext,
                         float* gState,
                         float* gCoverage,
                         const float* va,
                         const float* context,
                         const float* state,
                         const float* coverage,
                         const float* adj,
                         int m,  // rows
                         int k,  // cols
                         int n   // batch size
                         ) {
  int rows = m;
  int cols = k;
  for(int bid = 0; bid < m; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* gcRow = gContext + j * cols;
      float* gsRow = gState + (j % n) * cols;
      float* gcovRow = gCoverage ? gCoverage + j * cols : nullptr;

      const float* cRow = context + j * cols;
      const float* sRow = state + (j % n) * cols;
      const float* covRow = coverage ? coverage + j * cols : nullptr;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float z = cRow[id] + sRow[id];
          if(coverage)
            z += covRow[id];

          float t = tanhf(z);
          float r = va[id] * (1.f - t * t);

          gcRow[id] += r * adj[j];
          gsRow[id] += r * adj[j];
          if(gCoverage)
            gcovRow[id] += r * adj[j];
          atomicAdd(gVa + id, t * adj[j]);
        }
      }
    }
  }
}

void AttBack(Tensor gVa,
             Tensor gContext,
             Tensor gState,
             Tensor gCoverage,
             Tensor va,
             Tensor context,
             Tensor state,
             Tensor coverage,
             Tensor adj) {
  cudaSetDevice(adj->getDevice());

  size_t m = context->shape()[0] * context->shape()[2] * context->shape()[3];
  size_t k = context->shape()[1];

  size_t n = context->shape()[0];

  int blocks = std::min(MAX_BLOCKS, (int)n);
  int threads = std::min(MAX_THREADS, (int)k);

  gAttBack<<<blocks, threads>>>(gVa->data(),
                                gContext->data(),
                                gState->data(),
                                gCoverage ? gCoverage->data() : nullptr,

                                va->data(),
                                context->data(),
                                state->data(),
                                coverage ? coverage->data() : nullptr,

                                adj->data(),
                                m,
                                k,
                                n);
}

__global__ void gLNormalization(float* out,
                                const float* in,
                                const float* alpha,
                                const float* beta,
                                int rows,
                                int cols,
                                float eps = 1e-9) {
  extern __shared__ float _share[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = 0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share + blockDim.x;

      _sqSum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = sp[id] - mean;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float t = alpha[id] * ((sp[id] - mean) / sigma);
          if(beta != nullptr)
            t += beta[id];
          so[id] = t;
        }
      }
    }
  }
}

void LayerNormalization(
    Tensor out, Tensor in, Tensor gamma, Tensor beta, float eps) {
  cudaSetDevice(out->getDevice());

  int rows = in->shape()[0] * in->shape()[2] * in->shape()[3];
  int cols = in->shape()[1];

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = 2 * threads * sizeof(float);

  gLNormalization<<<blocks, threads, shared>>>(out->data(),
                                               in->data(),
                                               gamma->data(),
                                               beta ? beta->data() : nullptr,
                                               rows,
                                               cols,
                                               eps);
}

__global__ void gLayerNormalizationGrad(float* gradX,
                                        float* gradGamma,
                                        float* gradBeta,
                                        float* adj,
                                        float* y,
                                        float* x,
                                        float* gamma,
                                        float* beta,
                                        int rows,
                                        int cols,
                                        float eps = 1e-9) {
  extern __shared__ float shared[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* sum_adj = shared;
      float* sum_adj_x = shared + blockDim.x;
      float* sum_x = shared + 2 * blockDim.x;
      float* sum_sqr = shared + 3 * blockDim.x;

      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      sum_x[threadIdx.x] = 0.0f;
      sum_adj[threadIdx.x] = 0.0f;
      sum_adj_x[threadIdx.x] = 0.0f;
      sum_sqr[threadIdx.x] = 0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          sum_x[threadIdx.x] += xRow[id];
          sum_adj_x[threadIdx.x]
              += adjRow[id] * (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          sum_adj[threadIdx.x] += adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_x[threadIdx.x] += sum_x[threadIdx.x + skip];
          sum_adj[threadIdx.x] += sum_adj[threadIdx.x + skip];
          sum_adj_x[threadIdx.x] += sum_adj_x[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = sum_x[0] / cols;
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = xRow[id] - mean;
          sum_sqr[threadIdx.x] += ex * ex;
        }
      }

      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (sum_sqr[0] / cols));
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float grad_x = 0.0f;
          float x_hat = (yRow[id] - ((beta) ? beta[id] : 0)) / gamma[id];
          grad_x += cols * adjRow[id];
          grad_x -= sum_adj[0];
          grad_x -= sum_adj_x[0] * x_hat;
          grad_x /= (cols * sigma);

          gradXRow[id] += gamma[id] * grad_x;
          atomicAdd(gradGamma + id, adjRow[id] * x_hat);
          if(beta) {
            atomicAdd(gradBeta + id, adjRow[id]);
          }
        }
      }
    }
  }
}

void LayerNormalizationGrad(Tensor gradX,
                            Tensor gradGamma,
                            Tensor gradBeta,
                            Tensor adj,
                            Tensor y,
                            Tensor x,
                            Tensor gamma,
                            Tensor beta) {
  cudaSetDevice(adj->getDevice());
  int rows = y->shape()[0] * y->shape()[2] * y->shape()[3];
  int cols = y->shape()[1];

  int threads = std::min(MAX_THREADS, cols);
  int blocks = std::min(MAX_BLOCKS, rows);
  int shared = sizeof(float) * threads * 4;

  gLayerNormalizationGrad<<<blocks, threads, shared>>>(
      gradX->data(),
      gradGamma->data(),
      (gradBeta) ? gradBeta->data() : nullptr,
      adj->data(),
      y->data(),
      x->data(),
      gamma->data(),
      (beta) ? beta->data() : nullptr,
      rows,
      cols);
}

__global__ void gShift(float* out, const float* in, int length, int offset) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      if(index - offset < 0 || index - offset >= length)
        out[index] = 0;
      else
        out[index] = in[index - offset];
    }
  }
}

void Shift(Tensor out, Tensor in, ShapeGPU shift, bool invert) {
  int offset
      = in->shape().stride(0) * shift[0] + in->shape().stride(1) * shift[1]
        + in->shape().stride(2) * shift[2] + in->shape().stride(3) * shift[3];

  if(invert)
    offset = -offset;

  cudaSetDevice(out->getDevice());

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gShift<<<blocks, threads>>>(out->data(), in->data(), length, offset);
}

__global__ void gSetSparse(float* out,
                           const size_t* indeces,
                           const float* values,
                           int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[indeces[index]] = values[index];
    }
  }
}

void SetSparse(float* out,
               const std::vector<size_t>& indeces,
               const std::vector<float>& values) {
  int length = indeces.size();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  size_t* d_indeces;
  CUDA_CHECK(cudaMalloc(&d_indeces, length * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indeces,
                        indeces.data(),
                        length * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  float* d_values;
  CUDA_CHECK(cudaMalloc(&d_values, length * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(
      d_values, values.data(), length * sizeof(float), cudaMemcpyHostToDevice));

  gSetSparse<<<blocks, threads>>>(out, d_indeces, d_values, length);

  cudaFree(d_indeces);
  cudaFree(d_values);
}

/******************************************************************************/

__device__ inline float logit(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void gLSTMCellForward(float* out,
                                 const float* cell,
                                 const float* xW,
                                 const float* sU,
                                 const float* b,
                                 const float* mask,
                                 size_t rows,
                                 size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOut = out + j * cols;
      const float* rowCell = cell + j * cols;

      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float gf = logit(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;
          float gi = logit(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          float gc = tanhf(xWrow[l] + sUrow[l] + b[l]);

          float cout = gf * rowCell[i] + gi * gc;
          rowOut[i] = m * cout + (1 - m) * rowCell[i];
        }
      }
    }
  }
}

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs) {
  cudaSetDevice(out->getDevice());

  int rows = out->shape()[0] * out->shape()[2] * out->shape()[3];
  int cols = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMCellForward<<<blocks, threads>>>(
      out->data(),                                // output
      inputs[0]->data(),                          // cell state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      rows,
      cols);
}

__global__ void gLSTMOutputForward(float* out,
                                 const float* cell,
                                 const float* xW,
                                 const float* sU,
                                 const float* b,
                                 size_t rows,
                                 size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowCell = cell + j * cols;

      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {

          int k = i + 3 * cols ;
          float go = logit(xWrow[k] + sUrow[k] + b[k]);

          rowOut[i] = go * tanhf(rowCell[i]);
        }
      }
    }
  }
}

void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs) {
  cudaSetDevice(out->getDevice());

  int rows = out->shape()[0] * out->shape()[2] * out->shape()[3];
  int cols = out->shape()[1];

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMOutputForward<<<blocks, threads>>>(
      out->data(),                                // output
      inputs[0]->data(),                          // cell state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      rows,
      cols);
}

__global__ void gLSTMCellBackward(float* outCell,
                                  float* outXW,
                                  float* outSU,
                                  float* outB,
                                  const float* cell,
                                  const float* xW,
                                  const float* sU,
                                  const float* b,
                                  const float* mask,
                                  const float* adj,
                                  size_t rows,
                                  size_t cols) {

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      float* rowOutCell = outCell + j * cols;
      float* rowOutXW = outXW + j * cols * 4;
      float* rowOutSU = outSU + j * cols * 4;

      const float* rowCell = cell + j * cols;
      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {

          float gf = logit(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;
          float gi = logit(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          float gc = tanhf(xWrow[l] + sUrow[l] + b[l]);

          float adj = rowAdj[i];

          // dc/dc_{t-1}
          if(outCell)
            rowOutCell[i] += (m * gf - m + 1) * adj;

          // dc/d(b_f) = dc/d(xW_f) ...
          float dcdxf = m * rowCell[i] * gf * (1 - gf) * adj;
          if(outXW)
            rowOutXW[i] += dcdxf;
          if(outSU)
            rowOutSU[i] += dcdxf;
          if(outB)
            atomicAdd(outB + i, dcdxf);

          // dc/d(b_i) ...
          float dcdb_i = m * gc * gi * (1 - gi) * adj;
          if(outXW)
            rowOutXW[k] += dcdb_i;
          if(outSU)
            rowOutSU[k] += dcdb_i;
          if(outB)
            atomicAdd(outB + k, dcdb_i);

          // dc/d(b_c) ...
          float dcdxc = m * gi * (1 - gc * gc) * adj;
          if(outXW)
            rowOutXW[l] += dcdxc;
          if(outSU)
            rowOutSU[l] += dcdxc;
          if(outB)
            atomicAdd(outB + l, dcdxc);
        }
      }
    }
  }
}

void LSTMCellBackward(std::vector<Tensor> outputs,
                      std::vector<Tensor> inputs,
                      Tensor adj) {
  cudaSetDevice(adj->getDevice());

  int rows = adj->shape()[0] * adj->shape()[2] * adj->shape()[3];
  int cols = adj->shape()[1];

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMCellBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,        // state - adj
      outputs[1] ? outputs[1]->data() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data() : 0,        // b - adj
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      inputs.size() > 4 ? inputs[4]->data() : 0,  // mask
      adj->data(),
      rows,
      cols);
}

__global__ void gLSTMOutputBackward(float* outCell,
                                  float* outXW,
                                  float* outSU,
                                  float* outB,
                                  const float* cell,
                                  const float* xW,
                                  const float* sU,
                                  const float* b,
                                  const float* adj,
                                  size_t rows,
                                  size_t cols) {

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOutCell = outCell + j * cols;
      float* rowOutXW = outXW + j * cols * 4;
      float* rowOutSU = outSU + j * cols * 4;

      const float* rowCell = cell + j * cols;
      const float* xWrow = xW + j * cols * 4;
      const float* sUrow = sU + j * cols * 4;

      const float* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {

          int k = i + 3 * cols;
          float go = logit(xWrow[k] + sUrow[k] + b[k]);

          float t = tanhf(rowCell[i]);

          float adj = rowAdj[i];

          // dc/dc_{t-1}
          if(outCell)
            rowOutCell[i] += go * (1 - t * t) * adj;

          // dc/d(b_o) = dc/d(xW_f) ...
          float dcdxo = t * go * (1 - go) * adj;
          if(outXW)
            rowOutXW[k] += dcdxo;
          if(outSU)
            rowOutSU[k] += dcdxo;
          if(outB)
            atomicAdd(outB + k, dcdxo);

        }
      }
    }
  }
}

void LSTMOutputBackward(std::vector<Tensor> outputs,
                      std::vector<Tensor> inputs,
                      Tensor adj) {
  cudaSetDevice(adj->getDevice());

  int rows = adj->shape()[0] * adj->shape()[2] * adj->shape()[3];
  int cols = adj->shape()[1];

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  gLSTMOutputBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data() : 0,        // state - adj
      outputs[1] ? outputs[1]->data() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data() : 0,        // b - adj
      inputs[0]->data(),                          // state
      inputs[1]->data(),                          // xW
      inputs[2]->data(),                          // sU
      inputs[3]->data(),                          // b
      adj->data(),
      rows,
      cols);
}

}  // namespace marian
