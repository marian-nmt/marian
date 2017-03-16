#pragma once

#define MAX_THREADS 512
#define MAX_BLOCKS 65535

#include <cmath>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <iostream>

#include "gpu/mblas/thrust_functions.h"
#include "gpu/mblas/matrix.h"
#include "gpu/mblas/handles.h"

namespace amunmt {
namespace GPU {
namespace mblas {

template <class M>
void Debug(const M& m, size_t pos = 0, size_t l = 8) {
  std::cerr << m.Rows() << " " << m.Cols() << std::endl;
  for(size_t i = 0; i < m.Rows(); ++i) {
    std::cerr << i << ": ";
    for(size_t j = pos; j < m.Cols() && j < pos + l; ++j) {
      std::cerr << m.GetVec()[i * m.Cols() + j] << " ";
    }
    std::cerr << " ... ";

    for(size_t j = m.Cols() - l; j < m.Cols();  ++j) {
      std::cerr << m.GetVec()[i * m.Cols() + j] << " ";
    }
    std::cerr << std::endl;
    // if(i == 4)
      // break;
  }
}

template<typename T>
void copy(const T *in, size_t count, T *out,  cudaMemcpyKind kind) {
  HANDLE_ERROR( cudaMemcpyAsync(out, in, count * sizeof(T), kind, CudaStreamHandler::GetStream()) );
}

void Fill(Matrix& In, float value=0.0f);

Matrix& Swap(Matrix& Out, Matrix& In);

void Mean(Matrix& Out, const Matrix& In, const DeviceVector<int>& mapping);

void WeightedMean(Matrix& Out,const Matrix& Weights, const Matrix& In, const DeviceVector<int>& mapping);

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

void MapMatrix(Matrix& state, const DeviceVector<int>& mapping, size_t i);

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

Matrix& Prod2(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& Softmax(Matrix& Out, const DeviceVector<int>& batchIds, const DeviceVector<int>& srcMapping, size_t srcSize);

Matrix& LogSoftmax(Matrix& Out);

template <class Functor>
__global__ void gBroadcast(Functor functor,
                            float* out, const float* in1, const float* in2,
                            size_t rows, size_t rows1, size_t cols) {
  for (int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if (j < rows) {
      float* rowOut = out + j * cols;

      const float* rowIn1 = in1 + (j % rows1) * cols;
      const float* rowIn2 = in2 + (j / rows1) * cols;

      for (int tid = 0; tid < cols; tid += blockDim.x) {
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
__global__ void gBroadcast(Functor functor,
                           float* out, const float* in1, const float* in2,
                           size_t srcSize, size_t sumBeams, size_t cols, const int* batchMapping,
                           size_t batchMappingSize, size_t outSize, size_t in1Size, size_t in2Size,
                           size_t inRows)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < srcSize * inRows * cols) {
    int row = id / cols;
    int stateIdx = id % cols;

    int beamIdx = row / srcSize;
    int srcId = row % srcSize;

    int batchIdx = batchMapping[beamIdx];

    assert(id < outSize);
    assert((batchIdx * srcSize + srcId) * cols + stateIdx < in1Size);
    assert(beamIdx * cols + stateIdx < in2Size);

    out[id] = functor(in1[(batchIdx * srcSize + srcId) * cols + stateIdx],
                      in2[beamIdx * cols + stateIdx]);
  }
}

template <class Functor>
Matrix& Broadcast(Functor functor, Matrix& Out, const Matrix& In, const DeviceVector<int>& batchMapping, size_t srcSize) {
  size_t sumOfBeamSizes = In.Rows();

  //size_t rows = srcSize * sumOfBeamSizes;
  size_t cols  = Out.Cols();

  thread_local static Matrix Temp;
  Temp.Resize(sumOfBeamSizes, cols, srcSize);

  float* d_out = Temp.data();
  const float* d_in1 = Out.data();
  const float* d_in2 = In.data();

  int threads = 512;
  int blocks  = (Temp.size() / threads) + 1;

  /*
  std::cerr << "\nTemp=" << Temp.Debug() << std::endl;
  std::cerr << "Out=" << Out.Debug() << std::endl;
  std::cerr << "In=" << In.Debug() << std::endl;
  std::cerr << "srcSize=" << srcSize << std::endl;

  std::cerr << "batchMapping=" << batchMapping.size() << ":";
  for (size_t i = 0; i < batchMapping.size(); ++i) {
    std::cerr << batchMapping[i] << " ";
  }
  std::cerr << std::endl;
  */
  gBroadcast<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (functor, d_out, d_in1, d_in2, srcSize, batchMapping.size(), cols, thrust::raw_pointer_cast(batchMapping.data()),
        batchMapping.size(), Temp.size(), Out.size(), In.size(), In.Rows()
    );

  Swap(Out, Temp);
  return Out;
}

template <class Functor>
Matrix& BroadcastColumn(Functor functor, Matrix& Out, const Matrix& In) {
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
  if (threadIdx.x == 0) {
    for (int i = 0; i < rows; ++i)
      sdata[i] = in[i];
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
  //std::cerr << "Out=" << Out.Debug() << std::endl;
  //std::cerr << "In=" << In.Debug() << std::endl;

  size_t rows  = Out.Rows() * Out.Beam() * Out.Batches();
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
}
