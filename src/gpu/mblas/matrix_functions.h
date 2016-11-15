#pragma once

#define MAX_THREADS 512
#define MAX_BLOCKS 65535


#include <cmath>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "gpu/mblas/thrust_functions.h"
#include "gpu/mblas/matrix.h"
#include "gpu/mblas/handles.h"

namespace GPU {
namespace mblas {

template <class M>
void Debug(const M& m, size_t pos = 0, size_t l = 8) {
  std::cerr << m.Rows() << " " << m.Cols() << std::endl;
  for(size_t i = 0; i < m.Rows(); ++i) {
    for(size_t j = pos; j < m.Cols() && j < pos + l; ++j) {
      std::cerr << m.GetVec()[i * m.Cols() + j] << " ";
    }
    std::cerr << std::endl;
    // if(i == 4)
      // break;
  }
}

template<class IteratorT1, class IteratorT2>
void copy(IteratorT1 inBegin, IteratorT1 inEnd, IteratorT2 outBegin) {
  thrust::copy(thrust::cuda::par.on(CudaStreamHandler::GetStream()), inBegin, inEnd, outBegin);
}

template<class IteratorT1, class IteratorT2>
void copy_n(IteratorT1 inBegin, size_t size, IteratorT2 outBegin) {
  thrust::copy_n(thrust::cuda::par.on(CudaStreamHandler::GetStream()), inBegin, size, outBegin);
}

void Fill(Matrix& In, float value=0.0f);

Matrix& Swap(Matrix& Out, Matrix& In);

void Mean(Matrix& Out, const Matrix& In, const DeviceVector<int>& mapping);

Matrix& Transpose(Matrix& Out, const Matrix& In);

Matrix& Transpose(Matrix& Out);

Matrix& Copy(Matrix& Out, const Matrix& In);

Matrix& PasteRow(Matrix& Out,
                 const Matrix& In,
                 const size_t r = 0,
                 const size_t c = 0);
void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo=0, size_t sparse=1);

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r = 0,
                const size_t c = 0);

Matrix& Concat(Matrix& Out, const Matrix& In);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const size_t* devPairs,
                 size_t numPairs);

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<size_t>& indeces);

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim);

Matrix& Prod(cublasHandle_t handle, Matrix& C, const Matrix& A, const Matrix& B,
             bool transA=false, bool transB=false);

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& Softmax(Matrix& Out);

Matrix& LogSoftmax(Matrix& Out);

template <class Functor>
__global__ void gBroadcast(Functor functor,
                           float* out, const float* in1, const float* in2,
                           size_t rows, size_t rows1, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;

      const float* rowIn1 = in1 + (j % rows1) * cols;
      const float* rowIn2 = in2 + (j / rows1) * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowIn1[i], rowIn2[i]);
      }
    }
  }
}

template <class Functor>
Matrix& Broadcast(Functor functor, Matrix& Out, const Matrix& In) {
  size_t rows1 = Out.Rows();
  size_t rows2 = In.Rows();

  size_t rows = rows1 * rows2;
  size_t cols  = Out.Cols();

  thread_local static Matrix Temp;
  Temp.Resize(rows, cols);
  mblas::Fill(Temp, 1.0f);

  float* d_out = Temp.data();
  const float* d_in1 = Out.data();
  const float* d_in2 = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);

  gBroadcast<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (functor, d_out, d_in1, d_in2, rows, rows1, cols);

  Swap(Out, Temp);
  return Out;
}

template <class Functor>
Matrix& BroadcastColumn(Functor functor, Matrix& Out, const Matrix& In) {
  // @TODO: Make this efficient with special kernel!
  Matrix InTemp;
  Transpose(InTemp, In);

  Transpose(Out);
  Broadcast(functor, Out, InTemp);
  Transpose(Out);
  return Out;
}

template <class Functor>
__global__ void gBroadcastVecColumn(Functor functor,
                                    float* out, const float* in, size_t rows, size_t cols) {
  extern __shared__ float sdata[];
  if (threadIdx.x < rows) {
      sdata[threadIdx.x] = in[threadIdx.x];
  }
  __syncthreads();

  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    int index = noColumn;
    for (int noRow = 0; noRow < rows; ++noRow) {
        out[index] = functor(out[index], sdata[noRow]);
        index += cols;
    }
  }
}

template <class Functor>
Matrix& BroadcastVecColumn(Functor functor, Matrix& Out, const DeviceVector<float>& In) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  float* d_out = Out.data();
  const float* d_in = thrust::raw_pointer_cast(In.data());

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + (cols % threads != 0);

  gBroadcastVecColumn<<<blocks, threads, rows * sizeof(float), CudaStreamHandler::GetStream()>>>
    (functor, d_out, d_in, rows, cols);

  return Out;
}

template <class Functor>
__global__ void gBroadcastVec(Functor functor,
                              float* out, const float* in, size_t rows, size_t cols) {
  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    float vecValue = in[noColumn];
    int index = noColumn;
    for (int noRow = 0; noRow < rows; ++noRow) {
        out[index] = functor(out[index], vecValue);
        index += cols;
    }
  }
}

template <class Functor>
Matrix& BroadcastVec(Functor functor, Matrix& Out, const Matrix& In, cudaStream_t stream = 0) {
  size_t rows  = Out.Rows();
  size_t cols = Out.Cols();

  float* d_out = Out.data();
  const float* d_in = In.data();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + (cols % threads != 0);
  stream = CudaStreamHandler::GetStream();

  gBroadcastVec<<<blocks, threads, 0, stream>>>
    (functor, d_out, d_in, rows, cols);

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor, float* out,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
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
                         float* out, const float* in,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn = in + j * cols;

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
                         float* out, const float* in1, const float* in2,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn1 = in1 + j * cols;
      const float* rowIn2 = in2 + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn1[i], rowIn2[i]);
      }
    }
  }
}

template <class Functor>
Matrix& Element(Functor functor, Matrix& Out) {
  float* d_out = Out.data();
  int blocks  = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  cudaStream_t& stream = CudaStreamHandler::GetStream();

  gElement<<<blocks, threads, 0, stream>>>
    (functor, d_out, Out.Rows(), Out.Cols());

  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In) {
  float* d_out = Out.data();
  const float* d_in = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  cudaStream_t& stream = CudaStreamHandler::GetStream();

  gElement<<<blocks, threads, 0, stream>>>
    (functor, d_out, d_in, Out.Rows(), Out.Cols());

  return Out;
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In1, const Matrix& In2) {

  float* d_out = Out.data();
  const float* d_in1 = In1.data();
  const float* d_in2 = In2.data();

  int blocks  = std::min(MAX_BLOCKS, (int)Out.Rows());
  int threads = std::min(MAX_THREADS, (int)Out.Cols());
  cudaStream_t& stream = CudaStreamHandler::GetStream();

  gElement<<<blocks, threads, 0, stream>>>
    (functor, d_out, d_in1, d_in2, Out.Rows(), Out.Cols());

  return Out;
}

void SetColumn(Matrix& In, int noColumn, float value);




} // namespace mblas
} // namespace GPU
