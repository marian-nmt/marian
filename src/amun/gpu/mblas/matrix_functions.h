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
#include "gpu/mblas/matrix_wrapper.h"
#include "gpu/mblas/handles.h"

namespace amunmt {
namespace GPU {
namespace mblas {

template <class M>
void Debug(const M& m, size_t pos = 0, size_t l = 8) {
  std::cerr << m.dim(0) << " " << m.dim(1) << std::endl;
  for(size_t i = 0; i < m.dim(0); ++i) {
    std::cerr << i << ": ";
    for(size_t j = pos; j < m.dim(1) && j < pos + l; ++j) {
      std::cerr << m.GetVec()[i * m.dim(1) + j] << " ";
    }
    std::cerr << " ... ";

    for(size_t j = m.dim(1) - l; j < m.dim(1);  ++j) {
      std::cerr << m.GetVec()[i * m.dim(1) + j] << " ";
    }
    std::cerr << std::endl;
    // if(i == 4)
      // break;
  }
}

template<typename T>
std::string Debug(const DeviceVector<T> &vec, size_t verbosity = 1)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    T sum = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
      sum += vec[i];
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (size_t i = 0; i < vec.size(); ++i) {
      strm << " " << vec[i];
    }
  }

  return strm.str();
}

template<typename T>
std::string Debug(const thrust::host_vector<T> &vec, size_t verbosity = 1)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    T sum = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
      sum += vec[i];
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (size_t i = 0; i < vec.size(); ++i) {
      strm << " " << vec[i];
    }
  }

  return strm.str();
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
void PasteRows(Matrix& Out, const Matrix& In, const size_t rowNo, size_t colNo=0);

Matrix& CopyRow(Matrix& Out,
                const Matrix& In,
                const size_t r = 0,
                const size_t c = 0);

Matrix& Concat(Matrix& Out, const Matrix& In);

void MapMatrix(Matrix& state, const DeviceVector<int>& mapping, size_t i);

Matrix& CopyRows(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<size_t>& indices);

Matrix& Assemble(Matrix& Out,
                 const Matrix& In,
                 const DeviceVector<size_t>& indices);

Matrix& Slice(Matrix& Out,
              const Matrix& In,
              size_t n, size_t dim);

Matrix& Prod(Matrix& C, const Matrix& A, const Matrix& B,
             bool transA = false, bool transB = false);

Matrix& SoftmaxOld(Matrix& Out, const DeviceVector<int>& batchIds, const DeviceVector<int>& srcMapping, size_t batchSize);
Matrix& Softmax(Matrix& Out, const DeviceVector<int>& batchIds, const DeviceVector<int>& srcMapping, size_t batchSize);

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
  size_t rows1 = Out.dim(0);
  size_t rows2 = In.dim(0);

  size_t rows = rows1 * rows2;
  size_t cols  = Out.dim(1);

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
                           MatrixWrapper<float> outWrap,
                           const MatrixWrapper<float> in1Wrap,
                           const MatrixWrapper<float> in2Wrap,
                           const MatrixWrapper<int> batchMappingWrap,
                           float* out, const float* in1, const float* in2,
                           size_t srcSize, size_t sumBeams, size_t cols)
{
  size_t inRows = in2Wrap.dim(0);
  size_t outSize = outWrap.size();
  size_t in1Size = in1Wrap.size();
  size_t in2Size = in2Wrap.size();

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < srcSize * inRows * cols) {
    int row = id / cols;
    int stateIdx = id % cols;

    int beamIdx = row / srcSize;
    int srcId = row % srcSize;

    int batchIdx = batchMappingWrap[beamIdx];

    assert(id < outSize);
    assert((batchIdx * srcSize + srcId) * cols + stateIdx < in1Size);
    assert(beamIdx * cols + stateIdx < in2Size);

    out[id] = functor(in1[(batchIdx * srcSize + srcId) * cols + stateIdx],
                      in2[beamIdx * cols + stateIdx]);
  }
}

template <class Functor>
Matrix& Broadcast(Functor functor, Matrix& Out, const Matrix& In, const DeviceVector<int>& batchMapping, size_t srcSize) {
  size_t sumOfBeamSizes = In.dim(0);

  //size_t rows = srcSize * sumOfBeamSizes;
  size_t cols  = Out.dim(1);

  thread_local static Matrix Temp;
  Temp.Resize(sumOfBeamSizes, cols, srcSize);

  float* d_out = Temp.data();
  const float* d_in1 = Out.data();
  const float* d_in2 = In.data();

  MatrixWrapper<float> outWrap(Temp);
  const MatrixWrapper<float> in1Wrap(Out);
  const MatrixWrapper<float> in2Wrap(In);
  const MatrixWrapper<int> batchMappingWrap(batchMapping);

  int threads = MAX_THREADS;
  int blocks  = (Temp.size() / threads) + 1;

  gBroadcast<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (functor,
        outWrap, in1Wrap, in2Wrap, batchMappingWrap,
        d_out, d_in1, d_in2,
        srcSize, batchMapping.size(), cols);

  std::cerr << "nBlocks=" << blocks << std::endl;
  std::cerr << "nThreads=" << threads << std::endl;
  std::cerr << "outWrap=" << outWrap.Debug() << std::endl;
  std::cerr << "in1Wrap=" << in1Wrap.Debug() << std::endl;
  std::cerr << "in2Wrap=" << in2Wrap.Debug() << std::endl;
  std::cerr << "batchMappingWrap=" << batchMappingWrap.Debug() << std::endl;
  std::cerr << std::endl;

  HANDLE_ERROR(cudaDeviceSynchronize());


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
                                    MatrixWrapper<float> outWrap,
                                    const MatrixWrapper<float> inWrap) {
  extern __shared__ float sdata[];

  size_t rows  = outWrap.dim(0);
  size_t cols = outWrap.dim(1);

  if (threadIdx.x == 0) {
    for (int i = 0; i < rows; ++i)
      sdata[i] = inWrap[i];
  }
  __syncthreads();

  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    for (int noRow = 0; noRow < rows; ++noRow) {
      float &val = outWrap(noRow, noColumn, 0, 0);
      val = functor(val, sdata[noRow]);
    }
  }
}

template <class Functor>
Matrix& BroadcastVecColumn(Functor functor, Matrix& Out, const DeviceVector<float>& In) {
  size_t rows  = Out.dim(0);
  size_t cols = Out.dim(1);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + (cols % threads != 0);

  gBroadcastVecColumn<<<blocks, threads, rows * sizeof(float), CudaStreamHandler::GetStream()>>>
    (functor, outWrap, inWrap);

  return Out;
}

template <class Functor>
__global__ void gBroadcastVec(Functor functor,
                              MatrixWrapper<float> outWrap,
                              const MatrixWrapper<float> inWrap) {
  size_t rows  = outWrap.dim(0);
  size_t cols = outWrap.dim(1);

  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    float vecValue = inWrap(0, noColumn, 0, 0);
    for (int noRow = 0; noRow < rows; ++noRow) {
      float &val = outWrap(noRow, noColumn, 0, 0);
      val = functor(val, vecValue);
    }
  }
}

template <class Functor>
Matrix& BroadcastVec(Functor functor, Matrix& Out, const Matrix& In, cudaStream_t stream = 0) {
  //std::cerr << "Out=" << Out.Debug() << std::endl;
  //std::cerr << "In=" << In.Debug() << std::endl;

  size_t cols = Out.dim(1);

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + (cols % threads != 0);
  stream = CudaStreamHandler::GetStream();

  gBroadcastVec<<<blocks, threads, 0, stream>>>
    (functor, outWrap, inWrap);

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor,
                         MatrixWrapper<float> outWrap,
                         const MatrixWrapper<float> inWrap)
{
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < outWrap.size()) {
    outWrap[ind] = functor(outWrap[ind], inWrap[ind]);
  }
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In)
{
  assert(Out.size() == In.size());

  float* d_out = Out.data();
  const float* d_in = In.data();

  int threads = MAX_THREADS;
  int blocks  = Out.size() / threads + 1;
  cudaStream_t& stream = CudaStreamHandler::GetStream();

  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> inWrap(In);

  gElement<<<blocks, threads, 0, stream>>>
    (functor, outWrap, inWrap);

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor,
                         MatrixWrapper<float> outWrap,
                         const MatrixWrapper<float> in1Wrap,
                         const MatrixWrapper<float> in2Wrap)
{
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < outWrap.size()) {
    outWrap[ind] = functor(outWrap[ind], in1Wrap[ind], in2Wrap[ind]);
  }
}

template <class Functor>
Matrix& Element(Functor functor,
                Matrix& Out, const Matrix& In1, const Matrix& In2)
{
  assert(Out.size() == In1.size());
  assert(Out.size() == In2.size());

  float* d_out = Out.data();
  const float* d_in1 = In1.data();
  const float* d_in2 = In2.data();

  int threads = MAX_THREADS;
  int blocks  = Out.size() / threads + 1;
  cudaStream_t& stream = CudaStreamHandler::GetStream();

  //std::cerr << "Element3=" << Out.Debug(0) << std::endl;
  //std::cerr << "Element3=" << In1.Debug(0) << std::endl;
  //std::cerr << "Element3=" << In2.Debug(0) << std::endl;
  //std::cerr << std::endl;
  MatrixWrapper<float> outWrap(Out);
  const MatrixWrapper<float> in1Wrap(In1);
  const MatrixWrapper<float> in2Wrap(In2);
  //std::cerr << "outWrap=" << outWrap.Debug() << std::endl;

  gElement<<<blocks, threads, 0, stream>>>
    (functor, outWrap, in1Wrap, in2Wrap);

  //HANDLE_ERROR( cudaPeekAtLastError() );
  //HANDLE_ERROR( cudaDeviceSynchronize() );
  //HANDLE_ERROR( cudaPeekAtLastError() );

  return Out;
}

void SetColumn(Matrix& In, int noColumn, float value);

void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, const Matrix& beta,
                   float eps);

void Normalization(Matrix& out, const Matrix& in, const Matrix& alpha, float eps);

} // namespace mblas
} // namespace GPU
}
