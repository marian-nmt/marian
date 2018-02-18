#pragma once

#include <cublas_v2.h>

#include <thrust/pair.h>

#include "tensors/tensor.h"

#include "tensors/allocator.h"

#include "gpu/shape.h"
#include "gpu/tmp.h"
#include "gpu/tensor.h"
#include "functional/functional.h"

namespace marian {

bool IsNan(Tensor in);

const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

cublasHandle_t create_handle(size_t);

template <size_t K, bool broadcast, class Functor>
__global__ void gElement(Functor functor,
                         gpu::Array<gpu::Tensor<float>, K> tensors) {

  int length = tensors[0].shape().elements();
  gpu::Array<int, gpu::Shape::size()> dims;
  gpu::Array<int, K> indices;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {

      indices.fill(index);

      if(broadcast) {
        tensors[0].shape().dims(index, dims);
        for(int i = 1; i < K; ++i)
          indices[i] = tensors[i].shape().bindex(dims);
      }

      tensors[0][index] = gpu::apply(functor, tensors, indices);
    }
  }
}

template <class Functor, class ...Tensors>
void Element(Functor functor, Tensor out, Tensors ...tensors) {
  cudaSetDevice(out->getDevice().no);

  constexpr size_t K = sizeof...(tensors) + 1;
  gpu::Array<gpu::Tensor<float>, K> gTensors = {out, tensors...};

  int length = gTensors[0].shape().elements();
  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  bool broadcast = false;
  for(int i = 1; i < K; ++i)
    broadcast = broadcast || gTensors[0].shape() != gTensors[i].shape();

  if(broadcast)
    gElement<K, true><<<blocks, threads>>>(functor, gTensors);
  else
    gElement<K, false><<<blocks, threads>>>(functor, gTensors);
}

void TransposeND(Tensor out, Tensor in, const std::vector<int>& vAxis);

void Select(Ptr<Allocator> allocator,
            Tensor out,
            Tensor in,
            int axis,
            const std::vector<size_t>&);

void Insert(Ptr<Allocator> allocator,
            Tensor out,
            Tensor in,
            int axis,
            const std::vector<size_t>&);

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax);

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax);

template <size_t K, class Functor>
__global__ void gAddGeneric(Functor functor,
                            const gpu::Shape full,
                            gpu::Tensor<float> out,
                            gpu::Array<gpu::Tensor<float>, K> ins,
                            float scale = 1.0) {

  int outLength = out.shape().elements();
  bool same = outLength == full.elements();
  for(int i = 0; i < K; ++i)
    same = same && outLength == ins[i].shape().elements();

  constexpr size_t N = gpu::Shape::size();
  gpu::Array<int, N> len;
  for(int i = 0; i < N; ++i)
    len[i] = full[i] / out.shape()[i];

  gpu::Array<int, N> dims;
  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {

      if(same) {
        out[index] += gpu::apply(functor, ins, index) * scale;
      } else {
        out.shape().dims(index, dims);
        out[index] += gpu::loops(functor, ins, len, dims) * scale;
      }

    }
  }
}

template <size_t K, class Functor>
__global__ void gAddEqual(Functor functor,
                          gpu::Tensor<float> out,
                          gpu::Array<gpu::Tensor<float>, K> ins,
                          float scale,
                          bool broadcast) {
  int length = out.shape().elements();
  gpu::Array<int, gpu::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      gpu::Array<int, K> indices;
      indices.fill(index);

      if(broadcast) {
        out.shape().dims(index, dims);
        for(size_t i = 0; i < K; ++i)
          indices[i] = ins[i].shape().bindex(dims);
      }

      out[index] += gpu::apply(functor, ins, indices) * scale;
    }
  }
}

template <size_t K, class Functor>
__global__ void gAddReduce(Functor functor,
                           const gpu::Shape full,
                           gpu::Tensor<float> out,
                           gpu::Array<gpu::Tensor<float>, K> ins,
                           float scale = 1.0) {

  int rows = full.elements() / full.back();
  int cols = full.back();

  bool same = true;
  for(int i = 0; i < K; ++i)
    same = same && ins[i].shape().elements() == full.elements();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      if(same) {
        _sum[threadIdx.x] = 0;
        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols)
            _sum[threadIdx.x] += gpu::apply(functor, ins, j * cols + id);
        }
      } else {
        gpu::Array<int, gpu::Shape::size()> dims;
        _sum[threadIdx.x] = 0;

        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            full.dims(j * cols + id, dims);
            gpu::Array<int, K> indices;
            for(int i = 0; i < K; ++i)
              indices[i] = ins[i].shape().bindex(dims);
            _sum[threadIdx.x] += gpu::apply(functor, ins, indices);
          }
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
      out[j] += _sum[0] * scale;
    }
  }
}

template <class Functor, class ...Tensors>
void Add(Functor functor,
         float scale,
         Tensor out,
         Tensors... tensors) {
  cudaSetDevice(out->getDevice().no);

  Shape full = Shape::broadcast({out, tensors...});

  int length = out->shape().elements();

  constexpr size_t K = sizeof...(Tensors);

  gpu::Tensor<float> gOut = out;
  gpu::Array<gpu::Tensor<float>, K> gIns = {tensors ...};

  if(full.back() != 1 && out->shape().back() == 1) {
    size_t m = full.elements() / length;
    size_t k = full.back();

    int blocks = std::min(MAX_BLOCKS, (int)m);
    int threads = std::min(MAX_THREADS, (int)k);
    int shared = sizeof(float) * threads * 2;

    gAddReduce<<<blocks, threads, shared>>>(functor, full, gOut, gIns, scale);

  } else if(out->shape() == full) {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    bool broadcast = false;
    for(int i = 0; i < K; ++i)
      broadcast = broadcast || gOut.shape() != gIns[i].shape();

    gAddEqual<<<blocks, threads>>>(functor, gOut, gIns, scale, broadcast);
  } else {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gAddGeneric<<<blocks, threads>>>(functor, full, gOut, gIns, scale);
  }
}

template <class Functor, class ...Tensors>
void Add(Functor functor,
         Tensor out,
         Tensors... tensors) {
  Add(functor, 1, out, tensors...);
}

template <class Functor, class ...Tensors>
void Reduce(Functor functor,
            float scale,
            Tensor out,
            Tensors... tensors) {
  out->set(0);
  Add(functor, scale, out, tensors...);
}

template <class Functor, class ...Tensors>
void Reduce(Functor functor,
            Tensor out,
            Tensors... tensors) {
  out->set(0);
  Add(functor, out, tensors...);
}

float L2Norm(Tensor in);

void Softmax(Tensor out, Tensor in, Tensor mask = nullptr);
void LogSoftmax(Tensor out, Tensor in);

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val);
void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnSoftmax(Tensor out, Tensor in);
void CudnnSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnLogSoftmax(Tensor out, Tensor in);
void CudnnLogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CrossEntropyPick(Tensor out, Tensor in, Tensor pick);
void CrossEntropyPickBackward(Tensor out, Tensor adj, Tensor a, Tensor pick);

void Argmax(Tensor Out, const Tensor In);

void Prod(cublasHandle_t handle,
          Tensor C,
          const Tensor A,
          const Tensor B,
          bool transA,
          bool transB,
          float beta = 0,
          float scalar = 1);

void ProdBatched(cublasHandle_t handle,
                 Tensor C,
                 const Tensor A,
                 const Tensor B,
                 bool transA,
                 bool transB,
                 float beta = 0,
                 float scalar = 1);

void CopyRowsByIndex(Tensor out,
                     const Tensor in,
                     thrust::pair<size_t, size_t>* ipair,
                     size_t length);

void CopyRows(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void PasteRows(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void CopyCols(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void PasteCols(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs);
void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs);
void LSTMCellBackward(std::vector<Tensor> outputs,
                      std::vector<Tensor> inputs,
                      Tensor adj);
void LSTMOutputBackward(std::vector<Tensor> outputs,
                        std::vector<Tensor> inputs,
                        Tensor adj);

void GRUFastForward(Tensor out, std::vector<Tensor> inputs, bool final = false);

void GRUFastBackward(std::vector<Tensor> outputs,
                     std::vector<Tensor> inputs,
                     Tensor adj,
                     bool final = false);

void Att(Tensor out, Tensor va, Tensor context, Tensor state);
void AttBack(Tensor gva,
             Tensor gContext,
             Tensor gState,
             Tensor va,
             Tensor context,
             Tensor state,
             Tensor adj);

void LayerNormalization(Tensor out,
                        Tensor in,
                        Tensor gamma,
                        Tensor beta,
                        float eps = 1e-9);
void LayerNormalizationGrad(Tensor gradX,
                            Tensor gradGamma,
                            Tensor gradBeta,
                            Tensor adj,
                            Tensor y,
                            Tensor x,
                            Tensor gamma,
                            Tensor beta,
                            float eps = 1e-9);

void Shift(Tensor out, Tensor in, Shape shift, bool invert = false);

void SetSparse(float*,
               const std::vector<size_t>& indeces,
               const std::vector<float>& values);

void HighwayForward(Tensor out,
                    const Tensor in1,
                    const Tensor in2,
                    const Tensor t);

void HighwayBackward(Tensor out1,
                     Tensor out2,
                     Tensor outt,
                     const Tensor in1,
                     const Tensor in2,
                     const Tensor t,
                     const Tensor adj);

void PoolingWithMaskingForward(Tensor out,
                               Tensor in,
                               Tensor mask,
                               int width,
                               bool isEven=false);

void PoolingWithMaskingBackward(Tensor adj,
                                Tensor adjIn,
                                Tensor in,
                                Tensor mask,
                                int width,
                                bool isEven=false);
}
