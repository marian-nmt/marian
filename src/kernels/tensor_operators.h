#pragma once

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>

#include "kernels/shape_gpu.h"
#include "tensors/tensor.h"

namespace marian {

using namespace thrust::placeholders;
const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

class TensorGPU;

cublasHandle_t create_handle(size_t);

template <class Functor>
__global__ void gAddR2(Functor functor,
                       float* out,
                       ShapeGPU outShape,
                       const float* in1,
                       const ShapeGPU in1Shape,
                       const ShapeGPU full,
                       float scale = 1.0) {
  int outLength = outShape.elements();
  bool same = outLength == full.elements() && outLength == in1Shape.elements();

  int I = full[0] / outShape[0];
  int J = full[1] / outShape[1];
  int K = full[2] / outShape[2];
  int L = full[3] / outShape[3];

  int dims[4];
  int dimsFull[4];
  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {
      if(same) {
        out[index] += functor(in1[index]) * scale;
      } else {
        outShape.dims(index, dims);
        float sum = 0;
        for(int i = 0; i < I; ++i) {
          for(int j = 0; j < J; ++j) {
            for(int k = 0; k < K; ++k) {
              for(int l = 0; l < L; ++l) {
                dimsFull[0] = dims[0] + i;
                dimsFull[1] = dims[1] + j;
                dimsFull[2] = dims[2] + k;
                dimsFull[3] = dims[3] + l;

                int in1Index = in1Shape.bindex(dimsFull);
                sum += functor(in1[in1Index]);
              }
            }
          }
        }
        if(sum)
          out[index] += sum * scale;
      }
    }
  }
}

template <class Functor>
__global__ void gAddR2Eq(Functor functor,
                         float* out,
                         ShapeGPU outShape,
                         const float* in1,
                         const ShapeGPU inShape1,
                         float scale,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
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
                        ShapeGPU outShape,
                        const float* in1,
                        const ShapeGPU in1Shape,
                        const ShapeGPU full,
                        float scale = 1.0) {
  int rows = full[0] * full[2] * full[3];
  int cols = full[1];
  bool same = in1Shape.elements() == full.elements();

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
        int dims[4];
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

  auto full = out->shape();
  for(int i = 0; i < in->shape().size(); ++i)
    full.set(i, std::max(full[i], in->shape()[i]));

  int length = out->shape().elements();

  if(full[1] != 1 && out->shape()[1] == 1) {
    size_t m = full.elements() / length;
    size_t k = full[1];

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
                       ShapeGPU outShape,
                       const float* in1,
                       const ShapeGPU in1Shape,
                       const float* in2,
                       const ShapeGPU in2Shape,
                       const ShapeGPU full,
                       float scale = 1.0) {
  int outLength = outShape.elements();

  bool same = outLength == full.elements() && outLength == in1Shape.elements()
              && outLength == in2Shape.elements();

  int I = full[0] / outShape[0];
  int J = full[1] / outShape[1];
  int K = full[2] / outShape[2];
  int L = full[3] / outShape[3];

  int dims[4];
  int dimsFull[4];
  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {
      if(same) {
        out[index] += functor(in1[index], in2[index]) * scale;
      } else {
        outShape.dims(index, dims);
        float sum = 0;
        for(int i = 0; i < I; ++i) {
          for(int j = 0; j < J; ++j) {
            for(int k = 0; k < K; ++k) {
              for(int l = 0; l < L; ++l) {
                dimsFull[0] = dims[0] + i;
                dimsFull[1] = dims[1] + j;
                dimsFull[2] = dims[2] + k;
                dimsFull[3] = dims[3] + l;

                int in1Index = in1Shape.bindex(dimsFull);
                int in2Index = in2Shape.bindex(dimsFull);
                sum += functor(in1[in1Index], in2[in2Index]);
              }
            }
          }
        }
        if(sum)
          out[index] += sum * scale;
      }
    }
  }
}

template <class Functor>
__global__ void gAddR3Eq(Functor functor,
                         float* out,
                         ShapeGPU outShape,
                         const float* in1,
                         const ShapeGPU inShape1,
                         const float* in2,
                         const ShapeGPU inShape2,
                         float scale,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
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
                        ShapeGPU outShape,
                        const float* in1,
                        const ShapeGPU in1Shape,
                        const float* in2,
                        const ShapeGPU in2Shape,
                        const ShapeGPU full,
                        float scale = 1.0) {
  int rows = full[0] * full[2] * full[3];
  int cols = full[1];
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
        int dims[4];
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
void Add(
    Functor functor, Tensor out, Tensor in1, Tensor in2, float scale = 1.0) {
  cudaSetDevice(out->getDevice());

  auto full = out->shape();
  for(int i = 0; i < in1->shape().size(); ++i)
    full.set(i, std::max(full[i], in1->shape()[i]));
  for(int i = 0; i < in2->shape().size(); ++i)
    full.set(i, std::max(full[i], in2->shape()[i]));

  int length = out->shape().elements();

  if(full[1] != 1 && out->shape()[1] == 1) {
    size_t m = full.elements() / length;
    size_t k = full[1];

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
void Reduce(
    Functor functor, Tensor out, Tensor in1, Tensor in2, float scale = 1.0) {
  out->set(0);
  Add(functor, out, in1, in2, scale);
}

template <class Functor>
__global__ void gAddR4(Functor functor,
                       float* out,
                       ShapeGPU outShape,
                       const float* in1,
                       const ShapeGPU in1Shape,
                       const float* in2,
                       const ShapeGPU in2Shape,
                       const float* in3,
                       const ShapeGPU in3Shape,
                       const ShapeGPU full) {
  int outLength = outShape.elements();

  bool same = outLength == full.elements() && outLength == in1Shape.elements()
              && outLength == in2Shape.elements()
              && outLength == in3Shape.elements();

  int I = full[0] / outShape[0];
  int J = full[1] / outShape[1];
  int K = full[2] / outShape[2];
  int L = full[3] / outShape[3];

  int dims[4];
  int dimsFull[4];
  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(same) {
      out[index] += functor(in1[index], in2[index], in3[index]);
    } else {
      if(index < outLength) {
        outShape.dims(index, dims);
        float sum = 0;
        for(int i = 0; i < I; ++i) {
          for(int j = 0; j < J; ++j) {
            for(int k = 0; k < K; ++k) {
              for(int l = 0; l < L; ++l) {
                dimsFull[0] = dims[0] + i;
                dimsFull[1] = dims[1] + j;
                dimsFull[2] = dims[2] + k;
                dimsFull[3] = dims[3] + l;

                int in1Index = in1Shape.bindex(dimsFull);
                int in2Index = in2Shape.bindex(dimsFull);
                int in3Index = in3Shape.bindex(dimsFull);
                sum += functor(in1[in1Index], in2[in2Index], in3[in3Index]);
              }
            }
          }
        }
        if(sum)
          out[index] += sum;
      }
    }
  }
}

template <class Functor>
__global__ void gAdd1R4(Functor functor,
                        float* out,
                        ShapeGPU outShape,
                        const float* in1,
                        const ShapeGPU in1Shape,
                        const float* in2,
                        const ShapeGPU in2Shape,
                        const float* in3,
                        const ShapeGPU in3Shape,
                        const ShapeGPU full) {
  int rows = full[0] * full[2] * full[3];
  int cols = full[1];

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
        int dims[4];
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
                         ShapeGPU outShape,
                         const float* in1,
                         const ShapeGPU inShape1,
                         const float* in2,
                         const ShapeGPU inShape2,
                         const float* in3,
                         const ShapeGPU inShape3,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
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

  auto full = out->shape();
  for(int i = 0; i < in1->shape().size(); ++i)
    full.set(i, std::max(full[i], in1->shape()[i]));
  for(int i = 0; i < in2->shape().size(); ++i)
    full.set(i, std::max(full[i], in2->shape()[i]));
  for(int i = 0; i < in3->shape().size(); ++i)
    full.set(i, std::max(full[i], in3->shape()[i]));

  int length = out->shape().elements();

  if(full[1] != 1 && out->shape()[1] == 1) {
    size_t m = full.elements() / length;
    size_t k = full[1];

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

/*
template <int B, class T>
struct BroadcastView {
  T* ptr;
  ShapeGPU shape;

  template <>
  inline T& at<0b0000>(size_t i) {
    return ptr[i];
  }

  template <>
  inline T& at<0b0001>(size_t i) {
    return ptr[i % shape[3]];
  }

  template <>
  inline T& at<0b0010>(size_t i) {
    return ptr[i % shape[2]];
  }

  template <>
  inline T& at<0b0011>(size_t i) {
    return ptr[i % (shape[2] * shape[3])];
  }

  template <>
  inline T& at<0b0100>(size_t i) {
    return ptr[i % shape[1]];
  }

  template <>
  inline T& at<0b0101>(size_t i) {
    return ptr[i % (shape[1] * shape[3])];
  }

  template <>
  inline T& at<0b0110>(size_t i) {
    return ptr[i % (shape[1] * shape[2])];
  }

  template <>
  inline T& at<0b0111>(size_t i) {
    return ptr[i % (shape[1] * shape[2] * shape[3])];
  }

  template <>
  inline T& at<0b1111>(size_t i) {
    return ptr[0];
  }

  template <>
  inline T& at<0b1000>(size_t i) {
    return ptr[i % shape[0]];
  }

  T& operator[](size_t i) {
    return at<B>(i);
  }
};
*/

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out,
                         ShapeGPU outShape,
                         const float* in,
                         const ShapeGPU inShape,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
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
                         ShapeGPU outShape,
                         const float* in1,
                         const ShapeGPU inShape1,
                         const float* in2,
                         const ShapeGPU inShape2,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
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
                         ShapeGPU outShape,
                         const float* in1,
                         const ShapeGPU inShape1,
                         const float* in2,
                         const ShapeGPU inShape2,
                         const float* in3,
                         const ShapeGPU inShape3,
                         bool broadcast) {
  int length = outShape.elements();
  int dims[4];
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

/**************** Pick ************************/

template <class Functor>
__global__ void gPick(Functor functor,
                      float* out,
                      ShapeGPU outShape,
                      const float* pick) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      outShape.dims(index, dims);
      int row = dims[0];
      int col = dims[1];
      float picked = col == (int)pick[row];
      out[index] = functor(out[index], picked);
    }
  }
}

template <class Functor, class T1, class T2>
void Pick(Functor functor, T1 out, const T2 picks) {
  cudaSetDevice(out->getDevice());

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gPick<<<blocks, threads>>>(functor, out->data(), out->shape(), picks->data());
}

template <class Functor>
__global__ void gPick(Functor functor,
                      float* out,
                      ShapeGPU outShape,
                      const float* in,
                      const ShapeGPU inShape,
                      const float* pick) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      outShape.dims(index, dims);
      int inIndex = inShape.bindex(dims);
      int row = dims[0];
      int col = dims[1];
      float picked = col == (int)pick[row];
      out[index] = functor(out[index], in[inIndex], picked);
    }
  }
}

template <class Functor, class T1, class T2, class T3>
void Pick(Functor functor, T1 out, const T2 in, const T3 picks) {
  cudaSetDevice(out->getDevice());

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gPick<<<blocks, threads>>>(functor,
                             out->data(),
                             out->shape(),
                             in->data(),
                             in->shape(),
                             picks->data());
}

template <class Functor>
__global__ void gPickReduce(Functor functor,
                            float* out,
                            ShapeGPU outShape,
                            const float* in,
                            const ShapeGPU inShape,
                            const float* pick) {
  int length = inShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      inShape.dims(index, dims);
      int row = dims[0];
      int col = dims[1];
      int outIndex = outShape.bindex(dims);
      float picked = col == (int)pick[row];
      float result = functor(in[index], picked);

      if(result)
        atomicAdd(out + outIndex, result);
    }
  }
}

template <class Functor, class T1, class T2, class T3>
void PickReduce(Functor functor, T1 out, const T2 in, const T3 picks) {
  cudaSetDevice(out->getDevice());

  int length = in->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  out->set(0);
  gPickReduce<<<blocks, threads>>>(functor,
                                   out->data(),
                                   out->shape(),
                                   in->data(),
                                   in->shape(),
                                   picks->data());
}

template <class Functor>
__global__ void gPick(Functor functor,
                      float* out,
                      ShapeGPU outShape,
                      const float* in1,
                      const ShapeGPU inShape1,
                      const float* in2,
                      const ShapeGPU inShape2,
                      const float* pick) {
  int length = outShape.elements();
  int dims[4];
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      outShape.dims(index, dims);
      int inIndex1 = inShape1.bindex(dims);
      int inIndex2 = inShape2.bindex(dims);
      int row = dims[0];
      int col = dims[1];
      float picked = col == (int)pick[row];
      out[index] = functor(out[index], in1[inIndex1], in2[inIndex2], picked);
    }
  }
}

template <class Functor, class T1, class T2, class T3, class T4>
void Pick(Functor functor, T1 out, const T2 in1, const T3 in2, const T4 picks) {
  cudaSetDevice(out->getDevice());

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  gPick<<<blocks, threads>>>(functor,
                             out->data(),
                             out->shape(),
                             in1->data(),
                             in1->shape(),
                             in2->data(),
                             in2->shape(),
                             picks->data());
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
          float beta = 0);

void CopyRowsByIndex(Tensor out,
                     const Tensor in,
                     thrust::pair<size_t, size_t>* ipair,
                     size_t length);

void CopyRows(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void PasteRows(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void CopyCols(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void PasteCols(Tensor out, const Tensor in, const std::vector<size_t>& indeces);

void Transpose(cublasHandle_t cublasHandle, Tensor out, const Tensor in);

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax);

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax);

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs);
void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs);
void LSTMCellBackward(std::vector<Tensor> outputs, std::vector<Tensor> inputs, Tensor adj);
void LSTMOutputBackward(std::vector<Tensor> outputs, std::vector<Tensor> inputs, Tensor adj);

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

void LayerNormalization(
    Tensor out, Tensor in, Tensor gamma, Tensor beta, float eps = 1e-9);
void LayerNormalizationGrad(Tensor gradX,
                            Tensor gradGamma,
                            Tensor gradBeta,
                            Tensor adj,
                            Tensor y,
                            Tensor x,
                            Tensor gamma,
                            Tensor beta);

void Shift(Tensor out, Tensor in, ShapeGPU shift, bool invert = false);

void SetSparse(float*,
               const std::vector<size_t>& indeces,
               const std::vector<float>& values);
}
