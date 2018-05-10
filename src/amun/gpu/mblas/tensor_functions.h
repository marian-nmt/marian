#pragma once

#define MAX_THREADS 512
#define MAX_BLOCKS 65535

#include <cmath>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <iostream>

#include "gpu/mblas/thrust_functions.h"
#include "gpu/mblas/tensor.h"
#include "gpu/mblas/tensor_wrapper.h"
#include "gpu/mblas/handles.h"
#include "gpu/mblas/nth_element_kernels.h"
#include "gpu/mblas/vector_wrapper.h"

namespace amunmt {
namespace GPU {
namespace mblas {

template<typename T>
std::string Debug(const mblas::Vector<T> &vec, unsigned verbosity = 1)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    T sum(0);
    for (unsigned i = 0; i < vec.size(); ++i) {
      sum += vec[i];
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (unsigned i = 0; i < vec.size(); ++i) {
      strm << " " << vec[i];
    }
  }

  return strm.str();
}

template<typename T>
std::string Debug(const std::vector<T> &vec, unsigned verbosity = 1)
{
  std::stringstream strm;

  strm << "size=" << vec.size();

  if (verbosity) {
    T sum(0);
    for (unsigned i = 0; i < vec.size(); ++i) {
      sum += vec[i];
    }
    strm << " sum=" << sum;
  }

  if (verbosity == 2) {
    for (unsigned i = 0; i < vec.size(); ++i) {
      strm << " " << vec[i];
    }
  }

  return strm.str();
}

/*
template<typename T>
__global__ void gSum(const T *data, unsigned count, T &ret)
{
  ret = 0;
  for (unsigned i = 0; i < count; ++i) {
    ret += data[i];
  }
}

template<typename T>
T Sum(const T *data, unsigned count)
{
  T ret;
  T *d_ret;
  HANDLE_ERROR( cudaMalloc(&d_ret, sizeof(T)) );

  const cudaStream_t stream = CudaStreamHandler::GetStream();

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  gSum<<<1, 1, 0, stream>>>(data, count, *d_ret);
  HANDLE_ERROR(cudaGetLastError());

  HANDLE_ERROR( cudaMemcpyAsync(&ret, d_ret, sizeof(T), cudaMemcpyDeviceToHost, stream) );

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  HANDLE_ERROR(cudaFree(d_ret));

  return ret;
}
*/

template<typename T>
void copy(const T *in, unsigned count, T *out,  cudaMemcpyKind kind) {
  HANDLE_ERROR( cudaMemcpyAsync(out, in, count * sizeof(T), kind, CudaStreamHandler::GetStream()) );
}

void Fill(Tensor& In, float value=0.0f);

Tensor& Swap(Tensor& Out, Tensor& In);

void Mean(Tensor& Out,
          const Tensor& In,
          const mblas::Vector<unsigned> &sentenceLengths);

void WeightedMean(Tensor& Out,const Tensor& Weights, const Tensor& In, const mblas::Vector<unsigned>& mapping);

Tensor& Transpose(Tensor& Out, const Tensor& In);

Tensor& Transpose(Tensor& Out);

Tensor& Copy(Tensor& Out, const Tensor& In);

Tensor& PasteRow(Tensor& Out,
                 const Tensor& In,
                 const unsigned r = 0,
                 const unsigned c = 0);
void PasteRows(Tensor& Out, const Tensor& In, const unsigned rowNo, unsigned colNo=0);

Tensor& CopyRow(Tensor& Out,
                const Tensor& In,
                const unsigned r = 0,
                const unsigned c = 0);

Tensor& Concat(Tensor& Out, const Tensor& In);

void MapMatrix(Tensor& state,
              const mblas::Vector<unsigned> &sentenceLengths,
              unsigned i);

Tensor& CopyRows(Tensor& Out,
                 const Tensor& In,
                 const mblas::Vector<unsigned>& indices);

Tensor& Assemble(Tensor& Out,
                 const Tensor& In,
                 const mblas::Vector<unsigned>& indices);

Tensor& Slice(Tensor& Out,
              const Tensor& In,
              unsigned n, unsigned dim);

Tensor& Prod(Tensor& C, const Tensor& A, const Tensor& B,
             bool transB = false);

Tensor& Softmax(Tensor& Out,
                const mblas::Vector<unsigned>& batchIds,
                const mblas::Vector<unsigned> &sentenceLengths);

Tensor& LogSoftmax(Tensor& Out);

template <class Functor>
__global__ void gBroadcast(Functor functor,
                           TensorWrapper<float> outWrap,
                           const TensorWrapper<float> in1Wrap,
                           const TensorWrapper<float> in2Wrap,
                           const VectorWrapper<unsigned> batchMappingWrap)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < outWrap.size()) {
    /*
    unsigned indices[SHAPE_SIZE];
    outWrap.id2Indices(id, indices);

    unsigned srcId = indices[0];
    unsigned stateIdx = indices[1];
    unsigned beamIdx = indices[2];
    //assert(0 == indices[3]);
    */

    unsigned cols  = in1Wrap.dim(1);
    unsigned srcSize = outWrap.dim(0);

    unsigned row = id / cols;
    unsigned stateIdx = id % cols;
    unsigned beamIdx = row / srcSize;
    unsigned srcId = row % srcSize;

    unsigned batchIdx = batchMappingWrap[ beamIdx ];


    outWrap[id] = functor(in1Wrap[(batchIdx * srcSize + srcId) * cols + stateIdx],
                          in2Wrap[beamIdx * cols + stateIdx]);
    //outWrap[id] = functor(in1Wrap(indices[0], indices[1], 0, batchIdx),
    //                      in2Wrap(indices[2], indices[1], 0, 0));
    //outWrap(srcId, stateIdx, beamIdx, 0) = functor(in1Wrap(srcId, stateIdx, 0, batchIdx),
    //                                              in2Wrap(beamIdx, stateIdx, 0, 0));
  }
}

template <class Functor>
Tensor& Broadcast(Functor functor,
                  Tensor& out,
                  const Tensor& in1,
                  const Tensor& in2,
                  const mblas::Vector<unsigned>& batchMapping,
                  unsigned srcSize)
{
  BEGIN_TIMER("Broadcast");
  unsigned sumOfBeamSizes = in2.dim(0);

  //unsigned rows = srcSize * sumOfBeamSizes;
  unsigned cols  = in1.dim(1);

  out.NewSize(srcSize, cols, sumOfBeamSizes);

  TensorWrapper<float> outWrap(out);
  const TensorWrapper<float> in1Wrap(in1);
  const TensorWrapper<float> in2Wrap(in2);
  const VectorWrapper<unsigned> batchMappingWrap(batchMapping);

  unsigned size = out.size();
  unsigned threads = std::min((unsigned) MAX_THREADS, (unsigned)size);
  unsigned blocks  = (size / threads) + ((size % threads == 0) ?  0 : 1);

  gBroadcast<<<blocks, threads, 0, CudaStreamHandler::GetStream()>>>
    (functor, outWrap, in1Wrap, in2Wrap, batchMappingWrap);
  HANDLE_ERROR(cudaGetLastError());
  /*
  std::cerr << "size=" << size << std::endl;
  std::cerr << "nBlocks=" << blocks << std::endl;
  std::cerr << "nThreads=" << threads << std::endl;
  std::cerr << "outWrap=" << outWrap.Debug() << std::endl;
  std::cerr << "in1Wrap=" << in1Wrap.Debug() << std::endl;
  std::cerr << "in2Wrap=" << in2Wrap.Debug() << std::endl;
  std::cerr << "batchMapping=" << batchMapping.Debug() << std::endl;
  std::cerr << "srcSize=" << srcSize << std::endl;
  std::cerr << "sumOfBeamSizes=" << sumOfBeamSizes << std::endl;
  std::cerr << std::endl;

  HANDLE_ERROR(cudaDeviceSynchronize());
  */
  PAUSE_TIMER("Broadcast");
  return out;
}

template <class Functor>
__global__ void gBroadcastVecColumn(Functor functor,
                                    TensorWrapper<float> outWrap,
                                    const VectorWrapper<float> inWrap) {
  extern __shared__ float sdataOrig[];

  unsigned rows  = outWrap.dim(0);
  unsigned cols = outWrap.dim(1);

  VectorWrapper<float> sdata(sdataOrig, rows);

  if (threadIdx.x == 0) {
    for (int i = 0; i < rows; ++i)
      sdata[i] = inWrap[i];
  }
  __syncthreads();

  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    for (int noRow = 0; noRow < rows; ++noRow) {
      float &val = outWrap(noRow, noColumn);
      val = functor(val, sdata[noRow]);
    }
  }
}

template <class Functor>
Tensor& BroadcastVecColumn(Functor functor, Tensor& Out, const mblas::Vector<float>& In)
{
  unsigned rows  = Out.dim(0);
  unsigned cols = Out.dim(1);

  TensorWrapper<float> outWrap(Out);
  const VectorWrapper<float> inWrap(In);

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + ((cols % threads == 0) ?  0 : 1);

  gBroadcastVecColumn<<<blocks, threads, rows * sizeof(float), CudaStreamHandler::GetStream()>>>
    (functor, outWrap, inWrap);
  HANDLE_ERROR(cudaGetLastError());

  return Out;
}

template <class Functor>
__global__ void gBroadcastVec(Functor functor,
                              TensorWrapper<float> outWrap,
                              const TensorWrapper<float> inWrap)
{
  unsigned cols = outWrap.dim(1);

  int noColumn = threadIdx.x + blockDim.x * blockIdx.x;
  if (noColumn < cols) {
    float vecValue = inWrap(0, noColumn);

    for (int dim0 = 0; dim0 < outWrap.dim(0); ++dim0) {
      for (int dim2 = 0; dim2 < outWrap.dim(2); ++dim2) {
        for (int dim3 = 0; dim3 < outWrap.dim(3); ++dim3) {
          float &val = outWrap(dim0, noColumn, dim2, dim3);
          val = functor(val, vecValue);
        }
      }
    }

  }
}

template <class Functor>
Tensor& BroadcastVec(Functor functor, Tensor& Out, const Tensor& In)
{
  //std::cerr << "Out=" << Out.Debug() << std::endl;
  //std::cerr << "In=" << In.Debug() << std::endl;

  unsigned cols = Out.dim(1);

  TensorWrapper<float> outWrap(Out);
  const TensorWrapper<float> inWrap(In);

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks  = cols / threads  + ((cols % threads == 0) ?  0 : 1);
  const cudaStream_t& stream = CudaStreamHandler::GetStream();

  gBroadcastVec<<<blocks, threads, 0, stream>>>
    (functor, outWrap, inWrap);
  HANDLE_ERROR(cudaGetLastError());

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor,
                         TensorWrapper<float> outWrap)
{
  unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < outWrap.size()) {
    outWrap[ind] = functor(outWrap[ind]);
  }
}

template <class Functor>
Tensor& Element(Functor functor,
                Tensor& Out)
{
  unsigned size = Out.size();
  unsigned threads = std::min((unsigned) MAX_THREADS, (unsigned)size);
  unsigned blocks  = size / threads + ((size % threads == 0) ?  0 : 1);
  const cudaStream_t& stream = CudaStreamHandler::GetStream();

  TensorWrapper<float> outWrap(Out);

  gElement<<<blocks, threads, 0, stream>>>
    (functor, outWrap);
  HANDLE_ERROR(cudaGetLastError());

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor,
                         TensorWrapper<float> outWrap,
                         const TensorWrapper<float> inWrap)
{
  unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < outWrap.size()) {
    outWrap[ind] = functor(outWrap[ind], inWrap[ind]);
  }
}

template <class Functor>
Tensor& Element(Functor functor,
                Tensor& Out, const Tensor& In)
{
  assert(Out.size() == In.size());

  unsigned size = Out.size();
  unsigned threads = std::min((unsigned) MAX_THREADS, (unsigned)size);
  unsigned blocks  = size / threads + ((size % threads == 0) ?  0 : 1);
  const cudaStream_t& stream = CudaStreamHandler::GetStream();

  TensorWrapper<float> outWrap(Out);
  const TensorWrapper<float> inWrap(In);

  gElement<<<blocks, threads, 0, stream>>>
    (functor, outWrap, inWrap);
  HANDLE_ERROR(cudaGetLastError());

  return Out;
}

template <class Functor>
__global__ void gElement(Functor functor,
                         TensorWrapper<float> outWrap,
                         const TensorWrapper<float> in1Wrap,
                         const TensorWrapper<float> in2Wrap)
{
  unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < outWrap.size()) {
    outWrap[ind] = functor(outWrap[ind], in1Wrap[ind], in2Wrap[ind]);
  }
}

template <class Functor>
Tensor& Element(Functor functor,
                Tensor& Out, const Tensor& In1, const Tensor& In2)
{
  //std::cerr << "Out=" << Out.Debug() << std::endl;
  //std::cerr << "In1=" << In1.Debug() << std::endl;
  //std::cerr << "In2=" << In2.Debug() << std::endl;

  assert(Out.size() == In1.size());
  assert(Out.size() == In2.size());

  unsigned size = Out.size();
  unsigned threads = std::min((unsigned) MAX_THREADS, (unsigned)size);
  unsigned blocks  = size / threads + ((size % threads == 0) ?  0 : 1);
  const cudaStream_t& stream = CudaStreamHandler::GetStream();

  //std::cerr << "Element3=" << Out.Debug(0) << std::endl;
  //std::cerr << "Element3=" << In1.Debug(0) << std::endl;
  //std::cerr << "Element3=" << In2.Debug(0) << std::endl;
  //std::cerr << std::endl;
  TensorWrapper<float> outWrap(Out);
  const TensorWrapper<float> in1Wrap(In1);
  const TensorWrapper<float> in2Wrap(In2);
  //std::cerr << "outWrap=" << outWrap.Debug() << std::endl;

  gElement<<<blocks, threads, 0, stream>>>
    (functor, outWrap, in1Wrap, in2Wrap);
  HANDLE_ERROR(cudaGetLastError());

  //HANDLE_ERROR( cudaPeekAtLastError() );
  //HANDLE_ERROR( cudaDeviceSynchronize() );
  //HANDLE_ERROR( cudaPeekAtLastError() );

  return Out;
}

void SetColumn(Tensor& In, int noColumn, float value);

void Normalization(Tensor& out, const Tensor& in, const Tensor& alpha, const Tensor& beta,
                   float eps);

void Normalization(Tensor& out, const Tensor& in, const Tensor& alpha, float eps);

void LogSoftmaxAndNBest(mblas::Vector<NthOutBatch> &nBest,
                const Tensor& in,
                const Tensor& b4,
                const mblas::Vector<float> &costs,
                bool forbidUNK,
                unsigned maxBeamSize,
                const std::vector<unsigned>& beamSizes,
                unsigned beamSizeSum,
                bool isFirst,
                bool requireProb);

template<typename T>
void TestMemCpy(unsigned size, const T *data1)
{
  using namespace std;

  vector<T> h_vec2(size);

  T *d_vec;
  cudaMalloc(&d_vec, size * sizeof(T));

  // copy
  //cudaMemcpy(d_vec, h_vec1.data(), NUM * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(h_vec2.data(), d_vec, NUM * sizeof(float), cudaMemcpyDeviceToHost);

  cudaStream_t stream = mblas::CudaStreamHandler::GetStream();

  //cudaMemcpyAsync(d_vec, data1, size * sizeof(T), cudaMemcpyHostToDevice, stream);
  //cudaMemcpyAsync(h_vec2.data(), d_vec, size * sizeof(T), cudaMemcpyDeviceToHost, stream);

  mblas::copy(data1, size, d_vec, cudaMemcpyHostToDevice);
  mblas::copy(d_vec, size, h_vec2.data(), cudaMemcpyDeviceToHost);

  cerr << "h_vec2=";
  T sum = 0;
  for (unsigned i = 0; i < size; ++i) {
    //cerr << h_vec2[i] << " ";
    sum += h_vec2[i];
  }
  cerr << sum;
  cerr << endl;
  //cudaStreamDestroy(stream);
  cudaFree(d_vec);

}

void TestMemCpy();


} // namespace mblas
} // namespace GPU
}
