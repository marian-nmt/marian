#pragma once

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>

#include "tensors/tensor.h"

#include "kernels/shape_gpu.h"
#include "tensors/allocator.h"
#include "tensors/device_gpu.h"

namespace marian {

bool IsNan(Tensor in);

using namespace thrust::placeholders;
const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

class TensorGPU;

cublasHandle_t create_handle(size_t);

void Transpose4D(Tensor out, Tensor in, Shape tranpose);

void Select(Ptr<Allocator<DeviceGPU>> allocator,
            Tensor out,
            Tensor in,
            int axis,
            const std::vector<size_t>&);

void Insert(Ptr<Allocator<DeviceGPU>> allocator,
            Tensor out,
            Tensor in,
            int axis,
            const std::vector<size_t>&);

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax);

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax);

template <int K, class Functor>
struct FApply {};

template <class Functor>
struct FApply<1, Functor> {
  __device__ static inline float apply(Functor functor, const float** in, int* indices) {
    return functor(in[0][indices[0]]);
  }

  __device__ static inline float apply(Functor functor, const float** in, int index) {
    return functor(in[0][index]);
  }
};

template <class Functor>
struct FApply<2, Functor> {
  __device__ static inline float apply(Functor functor, const float** in, int* indices) {
    return functor(in[0][indices[0]], in[1][indices[1]]);
  }

  __device__ static inline float apply(Functor functor, const float** in, int index) {
    return functor(in[0][index], in[1][index]);
  }
};

template <class Functor>
struct FApply<3, Functor> {
  __device__ static inline float apply(Functor functor, const float** in, int* indices) {
    return functor(in[0][indices[0]], in[1][indices[1]], in[2][indices[2]]);
  }

  __device__ static inline float apply(Functor functor, const float** in, int index) {
    return functor(in[0][index], in[1][index], in[2][index]);
  }
};

template <int K, class Functor>
__device__ inline float apply(Functor functor, const float** in, int* indices) {
  return FApply<K, Functor>::apply(functor, in, indices);
}

template <int K, class Functor>
__device__ inline float apply(Functor functor, const float** in, int index) {
  return FApply<K, Functor>::apply(functor, in, index);
}


template <int N, int K>
struct Loop {
  template <class Functor>
  __device__ static float result(Functor functor,
                                 const float** in,
                                 const int** strides,
                                 int* pAcc,
                                 const int* length,
                                 const int* dim) {
    float sum = 0;
    int acc[K];
    const int* nStrides[K];
    for(int i = 0; i < *length; ++i) {
      for(int j = 0; j < K; ++j) {
        acc[j] = pAcc[j] + (*dim + i) * *strides[j];
        nStrides[j] = strides[j] + 1;
      }

      sum += Loop<N - 1, K>::result(functor,
                                    in,
                                    nStrides,
                                    acc,
                                    length + 1,
                                    dim + 1);
    }
    return sum;
  }
};

template <int K>
struct Loop<1, K> {
  template <class Functor>
  __device__ static float result(Functor functor,
                                 const float** in,
                                 const int** strides,
                                 int* pAcc,
                                 const int* length,
                                 const int* dim) {
    float sum = 0;
    int acc[K];
    for(int i = 0; i < *length; ++i) {
      for(int j = 0; j < K; ++j)
        acc[j] = pAcc[j] + (*dim + i) * *strides[j];

      sum += apply<K>(functor, in, acc);
    }
    return sum;
  }
};

template <int N, int K, class Functor>
__device__ inline float loops(Functor functor,
                              const float** in,
                              const int** strides,
                              const int* length,
                              const int* dim) {
  int pAcc[K] = {0};
  return Loop<N, K>::result(functor,
                            in, strides, pAcc,
                            length, dim);
}

template <class Functor>
__global__ void gAddR2(Functor functor,
                       float* out,
                       gpu::Shape outShape,

                       const float* in1,
                       const gpu::Shape in1Shape,

                       const gpu::Shape full,
                       float scale = 1.0) {
  int outLength = outShape.elements();

  bool same = outLength == full.elements();

  constexpr int N = gpu::Shape::size();
  int len[N];
  for(int i = 0; i < N; ++i)
    len[i] = full[i] / outShape[i];

  constexpr int K = 1;
  same = same && outLength == in1Shape.elements();
  const float* ins[K] = { in1 };
  const int* bstrides[K] = { in1Shape.bstride_ };

  int dims[N];
  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {
      if(same) {
        out[index] += apply<K>(functor, ins, index) * scale;
      } else {
        outShape.dims(index, dims);
        float sum = loops<N, K>(functor,
                                ins, bstrides,
                                len, dims);
        if(sum)
          out[index] += sum * scale;
      }
    }
  }
}


template <class Functor>
__global__ void gAddR2Eq(Functor functor,
                         float* out,
                         const gpu::Shape outShape,
                         const float* in1,
                         const gpu::Shape inShape1,
                         float scale,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[gpu::Shape::size()];

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      int inIndex1 = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex1 = inShape1.bindex(dims);
      }
      out[index] += functor(in1[inIndex1]) * scale;
    }
  }
}

template <class Functor>
__global__ void gAdd1R2(Functor functor,
                        float* out,
                        gpu::Shape outShape,
                        const float* in1,
                        const gpu::Shape in1Shape,
                        const gpu::Shape full,
                        float scale = 1.0) {
  int rows = full.elements() / full.back();
  int cols = full.back();
  bool same = in1Shape.elements() == full.elements();

  int dims[gpu::Shape::size()];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      if(same) {
        const float* sp1 = in1 + j * cols;
        _sum[threadIdx.x] = 0;
        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            _sum[threadIdx.x] += functor(sp1[id]);
          }
        }
      } else {
        _sum[threadIdx.x] = 0;

        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            full.dims(j * cols + id, dims);
            int in1Index = in1Shape.bindex(dims);
            _sum[threadIdx.x] += functor(in1[in1Index]);
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

template <class Functor>
void Add(Functor functor, Tensor out, Tensor in, float scale = 1.0) {
  cudaSetDevice(out->getDevice());

  auto full = Shape::broadcast({out, in});

  int length = out->shape().elements();

  if(full.back() != 1 && out->shape().back() == 1) {
    size_t m = full.elements() / length;
    size_t k = full.back();

    int blocks = std::min(MAX_BLOCKS, (int)m);
    int threads = std::min(MAX_THREADS, (int)k);
    int shared = sizeof(float) * threads * 2;

    gAdd1R2<<<blocks, threads, shared>>>(functor,
                                         out->data(),
                                         out->shape(),
                                         in->data(),
                                         in->shape(),
                                         full,
                                         scale);
  } else if(out->shape() == full) {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gAddR2Eq<<<blocks, threads>>>(functor,
                                  out->data(),
                                  out->shape(),
                                  in->data(),
                                  in->shape(),
                                  scale,
                                  out->shape() != in->shape());
  } else {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
    gAddR2<<<blocks, threads>>>(functor,
                                out->data(),
                                out->shape(),
                                in->data(),
                                in->shape(),
                                full,
                                scale);
  }
}

template <class Functor, class T1, class T2>
void Reduce(Functor functor, T1 out, T2 in, float scale = 1.0) {
  out->set(0);
  Add(functor, out, in, scale);
}


template <class Functor>
__global__ void gAddR3(Functor functor,
                       float* out,
                       gpu::Shape outShape,
                       const float* in1,
                       const gpu::Shape in1Shape,
                       const float* in2,
                       const gpu::Shape in2Shape,
                       const gpu::Shape full,
                       float scale = 1.0) {

  int outLength = outShape.elements();
  bool same = outLength == full.elements()
              && outLength == in1Shape.elements()
              && outLength == in2Shape.elements();

  constexpr size_t num = gpu::Shape::size();

  int len[num];
  for(int i = 0; i < num; ++i)
    len[i] = full[i] / outShape[i];

  const float* ins[2] = { in1, in2 };
  const int* strides[2] = { in1Shape.bstride_, in2Shape.bstride_ };

  int dims[num];

  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {
      if(same) {
        out[index] += apply<2>(functor, ins, index) * scale;
      } else {

        outShape.dims(index, dims);

        float sum = loops<num, 2>(functor,
                                  ins, strides,
                                  len, dims);

        if(sum)
          out[index] += sum * scale;
      }
    }
  }
}

template <class Functor>
__global__ void gAddR3Eq(Functor functor,
                         float* out,
                         gpu::Shape outShape,
                         const float* in1,
                         const gpu::Shape inShape1,
                         const float* in2,
                         const gpu::Shape inShape2,
                         float scale,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[gpu::Shape::size()];

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      int inIndex1 = index;
      int inIndex2 = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex1 = inShape1.bindex(dims);
        inIndex2 = inShape2.bindex(dims);
      }
      out[index] += functor(in1[inIndex1], in2[inIndex2]) * scale;
    }
  }
}

template <class Functor>
__global__ void gAdd1R3(Functor functor,
                        float* out,
                        gpu::Shape outShape,
                        const float* in1,
                        const gpu::Shape in1Shape,
                        const float* in2,
                        const gpu::Shape in2Shape,
                        const gpu::Shape full,
                        float scale = 1.0) {
  int rows = full.elements() / full.back();
  int cols = full.back();
  bool same = in1Shape.elements() == full.elements()
              && in2Shape.elements() == full.elements();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      if(same) {
        const float* sp1 = in1 + j * cols;
        const float* sp2 = in2 + j * cols;
        _sum[threadIdx.x] = 0;
        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            _sum[threadIdx.x] += functor(sp1[id], sp2[id]);
          }
        }
      } else {
        int dims[gpu::Shape::size()];
        _sum[threadIdx.x] = 0;

        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            full.dims(j * cols + id, dims);
            int in1Index = in1Shape.bindex(dims);
            int in2Index = in2Shape.bindex(dims);
            _sum[threadIdx.x] += functor(in1[in1Index], in2[in2Index]);
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

template <class Functor>
void Add(Functor functor,
         Tensor out,
         Tensor in1,
         Tensor in2,
         float scale = 1.0) {
  cudaSetDevice(out->getDevice());

  Shape full = Shape::broadcast({out, in1, in2});

  int length = out->shape().elements();

  if(full.back() != 1 && out->shape().back() == 1) {
    size_t m = full.elements() / length;
    size_t k = full.back();

    int blocks = std::min(MAX_BLOCKS, (int)m);
    int threads = std::min(MAX_THREADS, (int)k);
    int shared = sizeof(float) * threads * 2;

    gAdd1R3<<<blocks, threads, shared>>>(functor,
                                         out->data(),
                                         out->shape(),
                                         in1->data(),
                                         in1->shape(),
                                         in2->data(),
                                         in2->shape(),
                                         full,
                                         scale);
  } else if(out->shape() == full) {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
    gAddR3Eq<<<blocks, threads>>>(
        functor,
        out->data(),
        out->shape(),
        in1->data(),
        in1->shape(),
        in2->data(),
        in2->shape(),
        scale,
        out->shape() != in1->shape() || out->shape() != in2->shape());
  } else {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
    gAddR3<<<blocks, threads>>>(functor,
                                out->data(),
                                out->shape(),
                                in1->data(),
                                in1->shape(),
                                in2->data(),
                                in2->shape(),
                                full,
                                scale);
  }
}

template <class Functor>
void Reduce(Functor functor,
            Tensor out,
            Tensor in1,
            Tensor in2,
            float scale = 1.0) {
  out->set(0);
  Add(functor, out, in1, in2, scale);
}

template <class Functor>
__global__ void gAddR4(Functor functor,
                       float* out,
                       gpu::Shape outShape,
                       const float* in1,
                       const gpu::Shape in1Shape,
                       const float* in2,
                       const gpu::Shape in2Shape,
                       const float* in3,
                       const gpu::Shape in3Shape,
                       const gpu::Shape full) {

  int outLength = outShape.elements();

  bool same = outLength == full.elements() && outLength == in1Shape.elements()
              && outLength == in2Shape.elements()
              && outLength == in3Shape.elements();

  constexpr size_t num = gpu::Shape::size();

  int len[num];
  for(int i = 0; i < num; ++i)
    len[i] = full[i] / outShape[i];

  const float* ins[3] = { in1, in2, in3 };
  const int* strides[3] = { in1Shape.bstride_, in2Shape.bstride_, in3Shape.bstride_ };

  int dims[num];

  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {
      if(same) {
        out[index] += apply<3>(functor, ins, index);
      } else {

        outShape.dims(index, dims);

        float sum = loops<num, 3>(functor,
                                  ins, strides,
                                  len, dims);
        if(sum)
          out[index] += sum;
      }
    }
  }
}

template <class Functor>
__global__ void gAdd1R4(Functor functor,
                        float* out,
                        gpu::Shape outShape,
                        const float* in1,
                        const gpu::Shape in1Shape,
                        const float* in2,
                        const gpu::Shape in2Shape,
                        const float* in3,
                        const gpu::Shape in3Shape,
                        const gpu::Shape full) {
  int rows = full.elements() / full.back();
  int cols = full.back();

  bool same = in1Shape.elements() == full.elements()
              && in2Shape.elements() == full.elements()
              && in3Shape.elements() == full.elements();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ float _share[];
      float* _sum = _share + blockDim.x;

      if(same) {
        const float* sp1 = in1 + j * cols;
        const float* sp2 = in2 + j * cols;
        const float* sp3 = in3 + j * cols;
        _sum[threadIdx.x] = 0;
        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            _sum[threadIdx.x] += functor(sp1[id], sp2[id], sp3[id]);
          }
        }
      } else {
        int dims[gpu::Shape::size()];
        _sum[threadIdx.x] = 0;

        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            full.dims(j * cols + id, dims);
            int in1Index = in1Shape.bindex(dims);
            int in2Index = in2Shape.bindex(dims);
            int in3Index = in3Shape.bindex(dims);
            _sum[threadIdx.x]
                += functor(in1[in1Index], in2[in2Index], in3[in3Index]);
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
      out[j] += _sum[0];
    }
  }
}

template <class Functor>
__global__ void gAddR4Eq(Functor functor,
                         float* out,
                         gpu::Shape outShape,
                         const float* in1,
                         const gpu::Shape inShape1,
                         const float* in2,
                         const gpu::Shape inShape2,
                         const float* in3,
                         const gpu::Shape inShape3,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[gpu::Shape::size()];

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      int inIndex1 = index;
      int inIndex2 = index;
      int inIndex3 = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex1 = inShape1.bindex(dims);
        inIndex2 = inShape2.bindex(dims);
        inIndex3 = inShape3.bindex(dims);
      }
      out[index] += functor(in1[inIndex1], in2[inIndex2], in3[inIndex3]);
    }
  }
}

template <class Functor>
void Add(Functor functor, Tensor out, Tensor in1, Tensor in2, Tensor in3) {
  cudaSetDevice(out->getDevice());

  Shape full = Shape::broadcast({out, in1, in2, in3});

  int length = out->shape().elements();

  if(full.back() != 1 && out->shape().back() == 1) {
    size_t m = full.elements() / length;
    size_t k = full.back();

    int blocks = std::min(MAX_BLOCKS, (int)m);
    int threads = std::min(MAX_THREADS, (int)k);
    int shared = sizeof(float) * threads * 2;

    gAdd1R4<<<blocks, threads, shared>>>(functor,
                                         out->data(),
                                         out->shape(),
                                         in1->data(),
                                         in1->shape(),
                                         in2->data(),
                                         in2->shape(),
                                         in3->data(),
                                         in3->shape(),
                                         full);
  } else if(out->shape() == full) {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
    gAddR4Eq<<<blocks, threads>>>(functor,
                                  out->data(),
                                  out->shape(),
                                  in1->data(),
                                  in1->shape(),
                                  in2->data(),
                                  in2->shape(),
                                  in3->data(),
                                  in3->shape(),
                                  out->shape() != in1->shape()
                                      || out->shape() != in2->shape()
                                      || out->shape() != in3->shape());
  } else {
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
    gAddR4<<<blocks, threads>>>(functor,
                                out->data(),
                                out->shape(),
                                in1->data(),
                                in1->shape(),
                                in2->data(),
                                in2->shape(),
                                in3->data(),
                                in3->shape(),
                                full);
  }
}

template <class Functor>
void Reduce(Functor functor, Tensor out, Tensor in1, Tensor in2, Tensor in3) {
  out->set(0);
  Add(functor, out, in1, in2, in3);
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out,
                         gpu::Shape outShape,
                         const float* in,
                         const gpu::Shape inShape,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[gpu::Shape::size()];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      int inIndex = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex = inShape.bindex(dims);
      }
      out[index] = functor(out[index], in[inIndex]);
    }
  }
}

template <class Functor, class T1, class T2>
void Element(Functor functor, T1 out, T2 in) {
  cudaSetDevice(out->getDevice());

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gElement<<<blocks, threads>>>(functor,
                                out->data(),
                                out->shape(),
                                in->data(),
                                in->shape(),
                                out->shape() != in->shape());
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out,
                         gpu::Shape outShape,
                         const float* in1,
                         const gpu::Shape inShape1,
                         const float* in2,
                         const gpu::Shape inShape2,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[gpu::Shape::size()];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      int inIndex1 = index;
      int inIndex2 = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex1 = inShape1.bindex(dims);
        inIndex2 = inShape2.bindex(dims);
      }
      out[index] = functor(out[index], in1[inIndex1], in2[inIndex2]);
    }
  }
}

template <class Functor, class T1, class T2, class T3>
void Element(Functor functor, T1 out, T2 in1, T3 in2) {
  cudaSetDevice(out->getDevice());

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gElement<<<blocks, threads>>>(
      functor,
      out->data(),
      out->shape(),
      in1->data(),
      in1->shape(),
      in2->data(),
      in2->shape(),
      out->shape() != in1->shape() || out->shape() != in2->shape());
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out,
                         gpu::Shape outShape,
                         const float* in1,
                         const gpu::Shape inShape1,
                         const float* in2,
                         const gpu::Shape inShape2,
                         const float* in3,
                         const gpu::Shape inShape3,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[gpu::Shape::size()];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      int inIndex1 = index;
      int inIndex2 = index;
      int inIndex3 = index;
      if(broadcast) {
        outShape.dims(index, dims);
        inIndex1 = inShape1.bindex(dims);
        inIndex2 = inShape2.bindex(dims);
        inIndex3 = inShape3.bindex(dims);
      }
      out[index]
          = functor(out[index], in1[inIndex1], in2[inIndex2], in3[inIndex3]);
    }
  }
}

template <class Functor, class T1, class T2, class T3, class T4>
void Element(Functor functor, T1 out, T2 in1, T3 in2, T4 in3) {
  cudaSetDevice(out->getDevice());

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gElement<<<blocks, threads>>>(functor,
                                out->data(),
                                out->shape(),
                                in1->data(),
                                in1->shape(),
                                in2->data(),
                                in2->shape(),
                                in3->data(),
                                in3->shape(),
                                out->shape() != in1->shape()
                                    || out->shape() != in2->shape()
                                    || out->shape() != in3->shape());
}

template <class Functor>
__global__ void gElement(Functor functor, float* out, int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[index] = functor(out[index]);
    }
  }
}

template <class Functor, class T1>
void Element(Functor functor, T1 out) {
  cudaSetDevice(out->getDevice());

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gElement<<<blocks, threads>>>(functor, out->data(), length);
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

void Att(Tensor out, Tensor va, Tensor context, Tensor state, Tensor coverage);
void AttBack(Tensor gva,
             Tensor gContext,
             Tensor gState,
             Tensor gCoverage,
             Tensor va,
             Tensor context,
             Tensor state,
             Tensor coverage,
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
}
