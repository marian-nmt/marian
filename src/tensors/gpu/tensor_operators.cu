#include "common/types.h"
#include "tensors/tensor_operators.h"

#include "functional/functional.h"
#include "functional/tensor.h"
#include "tensors/allocator.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"

#include "tensors/gpu/add_all.h"

namespace marian {

namespace gpu {

namespace atomics {

static inline  __device__ void atomicAdd(float *address, float val) {
  ::atomicAdd(address, val);
}

#if COMPILE_FP16
// @TODO: copied from CuTorch, adapt this better, give credit.
static inline  __device__ void atomicAdd(half *address, half val) {
#if __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000 // compute capability 70 and higher with CUDA 10
  ::atomicAdd(address, val);
#else // __CUDA_ARCH__ < 700
  unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
  #if CUDA_VERSION < 9000
    half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
  #else
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = hsum + val;
    hsum = __half_raw(tmpres);
  #endif
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
#endif // __CUDA_ARCH__
}
#endif // COMPILE_FP16


}

template <typename T>
__global__ void gIsNaN(const T* in, int length, bool* isNaN, bool* isInf) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      if(isnan((float)in[index])) *isNaN = true;
      if(isinf((float)in[index])) *isInf = true;
    }
  }
}

void IsNaN(const Tensor in, Ptr<Allocator> allocator, bool& isNaN, bool& isInf) {
  cudaSetDevice(in->getDeviceId().no);

  int length = in->size();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  auto mem = allocator->alloc<bool>(2);
  bool* dIsNaN = &mem->data<bool>()[0];
  bool* dIsInf = &mem->data<bool>()[1];
  fill(in->getBackend(), dIsNaN, dIsNaN + 2, false);

  if(in->type() == Type::float32) {
    gIsNaN<<<blocks, threads>>>(in->data<float>(), length, dIsNaN, dIsInf);
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    gIsNaN<<<blocks, threads>>>(in->data<half>(), length, dIsNaN, dIsInf);
#endif
  } else {
    ABORT("IsNaN for type {} not implemented", in->type());
  }

  CudaCopy(dIsNaN, dIsNaN + 1, &isNaN);
  CudaCopy(dIsInf, dIsInf + 1, &isInf);

  allocator->free(mem);

  cudaStreamSynchronize(0);
}

template <typename T>
__global__ void gSanitizeGradient(T* in, int length,
                                  bool* isNaN, bool* isInf,
                                  bool pruneNaN, bool clipInf,
                                  float forNaN = 0.f, float forInf = 65504.f, float forInfNeg = -65504.f) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      float v = (float)in[index];
      // handle NaN
      if(isnan(v)) {
        if(pruneNaN) {
          in[index] = (T)forNaN;
        } else {
          *isNaN = true;
        }
      }
      // handle +/- Inf
      if(isinf(v)) {
        if(clipInf) {
          in[index] = v > 0 ? (T)forInf : (T)forInfNeg;
        } else {
         *isInf = true;
        }
      }
    }
  }
}

// This function is meant to clean gradients, i.e. clip infinities and prune NaNs if required. 
// If all NaNs and Infs have been removed we return `true` for indicating a sane gradient. 
// If `clipInf` is set, infinities are replaced with the maximum/minimum non-inf value for the tensor. 
// In that case infinities do not result in a bad gradient, since they get clipped.
// If `pruneNaN` is set, NaNs are replaced with 0. Since NaNs get removed now they do not result 
// in a bad gradient.
// If NaNs or infinities are detected but not removed (either because of `pruneNaN=false` or `clipInf=false`), 
// we return `false` indicating a bad gradient. 
bool SanitizeGradient(marian::Tensor in, Ptr<Allocator> allocator, bool pruneNaN, bool clipInf) {
  cudaSetDevice(in->getDeviceId().no);

  int length = in->size();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  auto mem = allocator->alloc<bool>(2);
  bool* dIsNaN = &mem->data<bool>()[0];
  bool* dIsInf = &mem->data<bool>()[1];
  fill(in->getBackend(), dIsNaN, dIsNaN + 2, false);

  float forNaN    = 0.f;
  float forInf    = NumericLimits<float>(in->type()).max;
  float forInfNeg = NumericLimits<float>(in->type()).lowest;

  if(in->type() == Type::float32) {
    gSanitizeGradient<<<blocks, threads>>>(in->data<float>(), length, dIsNaN, dIsInf, pruneNaN, clipInf, forNaN, forInf, forInfNeg);
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    gSanitizeGradient<<<blocks, threads>>>(in->data<half>(), length, dIsNaN, dIsInf, pruneNaN, clipInf, forNaN, forInf, forInfNeg);
#endif
  } else {
    ABORT("gSanitizeGradient for type {} not implemented", in->type());
  }

  bool isNaN, isInf;
  CudaCopy(dIsNaN, dIsNaN + 1, &isNaN);
  CudaCopy(dIsInf, dIsInf + 1, &isInf);

  allocator->free(mem);

  cudaStreamSynchronize(0);

  return !isNaN && !isInf;
}

template <bool add, typename To, typename From>
__global__ void gCopyCastTo(To* out, const From* in, int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      if(add)
        out[index] += (To)in[index];
      else 
        out[index]  = (To)in[index];
    }
  }
}

template <bool add, typename To, typename From>
void CopyCastTo(To* out, const From* in, int length) {
  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));
  gCopyCastTo<add><<<blocks, threads>>>(out, in, length);
}

template <bool add, typename T>
void CopyCastFrom(Tensor out, const T* in, int length) {
  if(out->type() == Type::float32) {
    CopyCastTo<add>(out->data<float>(), in, length);
#if COMPILE_FP16
  } else if(out->type() == Type::float16) {
    CopyCastTo<add>(out->data<half>(), in, length);
#endif
  } else if(out->type() == Type::float64) {
    CopyCastTo<add>(out->data<double>(), in, length);
  } else {
    ABORT("CopyCastTo to type {} not implemented", out->type());
  }
}

void CopyCast(Tensor out, const Tensor in) {
  cudaSetDevice(out->getDeviceId().no);

  if(in->type() == Type::float32) {
    CopyCastFrom</*add=*/false>(out, in->data<float>(), (int)in->size());
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    CopyCastFrom</*add=*/false>(out, in->data<half>(), (int)in->size());
#endif
  } else if(in->type() == Type::float64) {
    CopyCastFrom</*add=*/false>(out, in->data<double>(), (int)in->size());
  } else if(in->type() == Type::uint32) {
    CopyCastFrom</*add=*/false>(out, in->data<uint32_t>(), (int)in->size());
  } else {
    ABORT("CopyCastFrom from type {} not implemented", in->type());
  }
}

void AddCast(Tensor out, const Tensor in) {
  cudaSetDevice(out->getDeviceId().no);

  if(in->type() == Type::float32) {
    CopyCastFrom</*add=*/true>(out, in->data<float>(), (int)in->size());
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    CopyCastFrom</*add=*/true>(out, in->data<half>(), (int)in->size());
#endif
  } else if(in->type() == Type::float64) {
    CopyCastFrom</*add=*/true>(out, in->data<double>(), (int)in->size());
  } else if(in->type() == Type::uint32) {
    CopyCastFrom</*add=*/true>(out, in->data<uint32_t>(), (int)in->size());
  } else {
    ABORT("CopyCastFrom from type {} not implemented", in->type());
  }
}

void ConcatCont(Tensor out, const std::vector<Tensor>& inputs, int axis) {
  cudaSetDevice(out->getDeviceId().no);
  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= out->shape()[i];

  size_t offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto in : inputs) {
      size_t size = (in->shape().elements() / step) * sizeOf(out->type());
      size_t offset2 = i * size;

      cudaMemcpy(out->data<uint8_t>() + offset1,
                 in->data<uint8_t>() + offset2,
                 size,
                 cudaMemcpyDeviceToDevice);

      offset1 += size;
    }
  }
  cudaStreamSynchronize(0);
}

template <bool add, typename T>
__global__ void gInsertCols(T* out,
                            const T* in,
                            size_t rows,
                            size_t cols,
                            size_t cols_out,
                            size_t cols_in,
                            size_t offset_out,
                            size_t offset_in) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) { // @TODO: change to if j == rows then break, as that's what it means. In 4 functions in here.
      T* rowOut = out + j * cols_out + offset_out;
      const T* rowIn = in + j * cols_in + offset_in;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          if(add)
            rowOut[i] += rowIn[i];
          else
            rowOut[i] = rowIn[i];
      }
    }
  }
}

// Special version for axis = -1 @TODO: write common better version
void Concatenate1(Tensor out, const std::vector<Tensor>& inputs) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();

  size_t offset = 0;
  int cols_out = out->shape().back();

  for(auto in : inputs) {
    ABORT_IF(rows != in->shape().elements() / in->shape().back(),
             "First dimension must be equal");
    int cols_in = in->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_in);

    if(out->type() == Type::float32) {
      gInsertCols<false><<<blocks, threads>>>(out->data<float>(), in->data<float>(), rows, cols_in, cols_out, cols_in, offset, 0);
#if COMPILE_FP16
    } else if(out->type() == Type::float16) {
      gInsertCols<false><<<blocks, threads>>>(out->data<half>(), in->data<half>(), rows, cols_in, cols_out, cols_in, offset, 0);
#endif
    } else {
      ABORT("Concatenate1 not implemented for type {}", out->type());
    }
    offset += cols_in;
  }
  cudaStreamSynchronize(0);
}

template <typename T>
__global__ void gJoin2(T* out,
                       size_t rowBatch,
                       size_t cols,
                       const T* in1,
                       size_t inStride1,
                       const T* in2,
                       size_t inStride2) {
  int outStride = inStride1 + inStride2;
  int rows = rowBatch * outStride;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols;

      int curBatch = j / outStride;
      int curPos = j % outStride;

      int jIn1 = (curBatch * inStride1) + curPos;
      int jIn2 = (curBatch * inStride2) + curPos - inStride1;

      const T* rowIn1 = in1 + jIn1 * cols;
      const T* rowIn2 = in2 + jIn2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(curPos < inStride1)
            rowOut[i] = rowIn1[i];
          else
            rowOut[i] = rowIn2[i];
        }
      }
    }
  }
}

// Special version for axis = -2 @TODO: write common better version
void Concatenate2(Tensor out, Tensor in1, Tensor in2) {
  cudaSetDevice(out->getDeviceId().no);

  size_t rows = out->shape().elements() / out->shape().back();
  size_t cols = out->shape().back();

  size_t rowStride1 = in1->shape()[-2];
  size_t rowStride2 = in2->shape()[-2];

  size_t rowBatch = rows / out->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);

  if(out->type() == Type::float32) {
     gJoin2<<<blocks, threads>>>(out->data<float>(),
                                 rowBatch,
                                 cols,
                                 in1->data<float>(),
                                 rowStride1,
                                 in2->data<float>(),
                                 rowStride2);
#if COMPILE_FP16
  } else if(out->type() == Type::float16) {
     gJoin2<<<blocks, threads>>>(out->data<half>(),
                                 rowBatch,
                                 cols,
                                 in1->data<half>(),
                                 rowStride1,
                                 in2->data<half>(),
                                 rowStride2);
#endif
  } else {
    ABORT("Concatenate2 not implemented for type {}", out->type());
  }

  cudaStreamSynchronize(0);
}

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax) {
  if(ax == out->shape().size() - 1)
    Concatenate1(out, inputs);
  else if(ax == out->shape().size() - 2 && inputs.size() == 2)
    Concatenate2(out, inputs[0], inputs[1]);
  else
    ConcatCont(out, inputs, ax);
}

void Split1(std::vector<Tensor>& outputs, const Tensor in) {
  cudaSetDevice(in->getDeviceId().no);

  size_t offset = 0;
  int rows = in->shape().elements() / in->shape().back();
  int cols_in = in->shape().back();
  for(auto out : outputs) {
    ABORT_IF(rows != out->shape().elements() / out->shape().back(),
             "First dimension must be equal");
    int cols_out = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols_out);

    if(out->type() == Type::float32) {
      gInsertCols<true><<<blocks, threads>>>(
          out->data<float>(), in->data<float>(), rows, cols_out, cols_out, cols_in, 0, offset);
#if COMPILE_FP16
    } else if(out->type() == Type::float16) {
      gInsertCols<true><<<blocks, threads>>>(
          out->data<half>(), in->data<half>(), rows, cols_out, cols_out, cols_in, 0, offset);
#endif
    } else {
      ABORT("Split1 not implemented for type {}", out->type());
    }

    offset += cols_out;
  }
  cudaStreamSynchronize(0);
}

// @TODO: this function is just a temporary fix until I come up with
// something better for the situation below.
template <typename T>
__global__ void gAddRow(T* out, const T* in, int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[index] = in[index] + out[index];
    }
  }
}

void SplitCont(std::vector<Tensor>& outputs, const Tensor in, int axis) {
  cudaSetDevice(in->getDeviceId().no);

  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= in->shape()[i];

  int offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto out : outputs) {
      int size = out->shape().elements() / step;
      int offset2 = i * size;

      // BUG: this is does not add gradients
      // cudaMemcpyAsync(out->data() + offset2,
      //                in->data() + offset1,
      //                size * sizeof(float),
      //                cudaMemcpyDeviceToDevice);

      // @TODO: this is a quick but bad fix for the above bug
      int threads = std::min(MAX_THREADS, size);
      int blocks = std::min(MAX_BLOCKS, size / threads + (size % threads != 0));

      if(out->type() == Type::float32) {
        gAddRow<<<blocks, threads>>>(out->data<float>() + offset2, in->data<float>() + offset1, size);
#if COMPILE_FP16
      } else if(out->type() == Type::float16) {
        gAddRow<<<blocks, threads>>>(out->data<half>() + offset2, in->data<half>() + offset1, size);
#endif
      } else {
        ABORT("SplitCont not implemented for type {}", out->type());
      }
      offset1 += size;
    }
  }
  cudaStreamSynchronize(0);
}

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax) {
  if(ax == in->shape().size() - 1)
    Split1(outputs, in);
  else
    SplitCont(outputs, in, ax);
}

template <bool add, typename T>
__global__ void gTransposeND(
    functional::Tensor<T> out,
    const functional::Tensor<T> in,
    const functional::Array<int, functional::Shape::size()> permute) {
  constexpr size_t N = functional::Shape::size();
  functional::Array<int, N> oDims;
  functional::Array<int, N> pDims;

  int length = out.shape().elements();
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out.shape().dims(index, oDims);
      for(int i = 0; i < N; ++i)
        pDims[permute[i]] = oDims[i];

      int inIndex = in.shape().index(pDims);

      // TODO: operates on raw indices, change to
      // converting Tensor::operator[]
      if(add)
        out.data()[index] += in.data()[inIndex];
      else
        out.data()[index] = in.data()[inIndex];
    }
  }
}

template <bool add, typename T>
__global__ void gTranspose0213(T* out,
                               const T* in,
                               int rows,
                               int cols,
                               int stride1,
                               int stride2) {
  int stride = stride1 * stride2;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols;

      int z = j / stride;
      int y = (j % stride) / stride1;
      int x = (j % stride) % stride1;
      int j2 = z * stride + x * stride2 + y;

      const T* rowIn = in + j2 * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(add)
            rowOut[i] += rowIn[i];
          else
            rowOut[i] = rowIn[i];
        }
      }
    }
  }
}

void TransposeND(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  cudaSetDevice(out->getDeviceId().no);
  if(vAxis == std::vector<int>({0, 2, 1, 3})) {
    int rows = out->shape().elements() / out->shape().back();
    int cols = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols);

    int stride1 = out->shape()[-2];
    int stride2 = out->shape()[-3];

    if(in->type() == Type::float32) {
      gTranspose0213<false><<<blocks, threads>>>(out->data<float>(), in->data<float>(), rows, cols, stride1, stride2);
    } else if(in->type() == Type::uint32) {
      gTranspose0213<false><<<blocks, threads>>>(out->data<uint32_t>(), in->data<uint32_t>(), rows, cols, stride1, stride2);
#if COMPILE_FP16
    } else if(in->type() == Type::float16) {
      gTranspose0213<false><<<blocks, threads>>>(out->data<half>(), in->data<half>(), rows, cols, stride1, stride2);
#endif
    } else {
      ABORT("Transpose for type {} not implemented", in->type());
    }
  } else {
    functional::Array<int, functional::Shape::size()> axes;
    int diff = functional::Shape::size() - vAxis.size();
    for(int i = 0; i < axes.size(); ++i)
      if(i < diff)
        axes[i] = i;
      else
        axes[i] = vAxis[i - diff] + diff;

    int length = out->shape().elements();
    int threads = std::min(MAX_THREADS, length);
    int blocks
        = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    if(in->type() == Type::float32) {
      gTransposeND<false, float><<<blocks, threads>>>(out, in, axes);
    } else if(in->type() == Type::uint32) {
      gTransposeND<false, uint32_t><<<blocks, threads>>>(out, in, axes);
#if COMPILE_FP16
    } else if(in->type() == Type::float16) {
      gTransposeND<false, half><<<blocks, threads>>>(out, in, axes);
#endif
    } else {
      ABORT("Transpose for type {} not implemented", in->type());
    }
  }
}

//@TODO: code duplication?
void TransposeNDGrad(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  cudaSetDevice(out->getDeviceId().no);
  if(vAxis == std::vector<int>({0, 2, 1, 3})) {
    int rows = out->shape().elements() / out->shape().back();
    int cols = out->shape().back();

    int blocks = std::min(MAX_BLOCKS, rows);
    int threads = std::min(MAX_THREADS, cols);

    int stride1 = out->shape()[-2];
    int stride2 = out->shape()[-3];

    if(in->type() == Type::float32) {
      gTranspose0213<true><<<blocks, threads>>>(out->data<float>(), in->data<float>(), rows, cols, stride1, stride2);
#if COMPILE_FP16
    } else if(in->type() == Type::float16) {
      gTranspose0213<true><<<blocks, threads>>>(out->data<half>(), in->data<half>(), rows, cols, stride1, stride2);
#endif
    } else {
      ABORT("Transpose for type {} not implemented", in->type());
    }
  } else {
    functional::Array<int, functional::Shape::size()> axes;
    int diff = functional::Shape::size() - vAxis.size();
    for(int i = 0; i < axes.size(); ++i)
      if(i < diff)
        axes[i] = i;
      else
        axes[i] = vAxis[i - diff] + diff;

    int length = out->shape().elements();
    int threads = std::min(MAX_THREADS, length);
    int blocks  = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    if(in->type() == Type::float32) {
      gTransposeND<true, float><<<blocks, threads>>>(out, in, axes);
#if COMPILE_FP16
    } else if(in->type() == Type::float16) {
      gTransposeND<true, half><<<blocks, threads>>>(out, in, axes);
#endif
    } else {
      ABORT("Transpose for type {} not implemented", in->type());
    }
  }
}

// Computes the softmax
// in - input tensor
// out - output tensor
// we compute the softmax over the the cols (last dimension)
// rows are time, batch or beam dimensions
// number of threads is number of cols or MAX_THREADS
// number of blocks is number of rows or MAX_BLOCKS
// @TODO: handle half2
template <typename T, typename AccType = float>
__global__ void gSoftmax(T* out,
                         functional::Shape outShape,
                         const T* in) {
  using namespace functional;

  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) { // loop over blocks of rows
    int j = bid + blockIdx.x; // blockIdx.x - row index (within block of rows)
    if(j < rows) { // compute softmax over one row, row elements distributed over threads
      T* so = out + j * cols; // pointer to row input data
      const T* sp = in + j * cols;

      // CUDA complains if type or size of shared memory changes, keep size constant.
      extern __shared__ uint8_t _sharedBytes[];
      T* _share = (T*)_sharedBytes;
      AccType* _shareAccType = (AccType*)_sharedBytes;

      // determine max (used below to improve numeric stability)
      T* _max = _share;
      
      // @TODO: what's going on here with fp16?
      _max[threadIdx.x] = -CUDA_FLT_MAX;  // mask
      // find max over column indices that have the same relative column index (=threadIdx.x) across all blocks of columns
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        // threadIdx.x = column index within block of columns; we reduce over columns within a block, then over blocks
        int i = tid + threadIdx.x;
        if(i < cols) {
          if(sp[i] > _max[threadIdx.x])
            _max[threadIdx.x] = sp[i];
        }
      }
      __syncthreads();
      // max over columns within a column block via tree reduction
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
      T max = _max[0];
      __syncthreads();

      // compute denominator
      AccType* _sum = _shareAccType; // accumulate into AccType
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          // @TODO: is it faster to cache the result of expf() in GPU RAM, or would it be faster to recompute it below?
          T ex = Ops<T>::exp(sp[i] - max);
          so[i] = (T)ex;
          _sum[threadIdx.x] += (AccType)ex; // accumulate into AccType
        }
      }
      __syncthreads();
      // now reduce over all columns within the block
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();

      // produce final output data
      AccType sum = _sum[0];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          so[i] = (T)((AccType)so[i] / sum); // divide as AccType then convert
        }
      }
    }
    __syncthreads();
  }
}

void Softmax(Tensor out, Tensor in) {
  cudaSetDevice(out->getDeviceId().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads;  // accumulate into float

  if(in->type() == Type::float32) {
    gSoftmax<float, float><<<blocks, threads, shared>>>(out->data<float>(), out->shape(), in->data<float>());
#if COMPILE_FP16
  } else if (in->type() == Type::float16) {
    gSoftmax<half, float><<<blocks, threads, shared>>>(out->data<half>(), out->shape(), in->data<half>());
#endif
  } else {
    ABORT("Softmax not implemented for type {}", in->type());
  }
}

template <typename T>
__global__ void gSinusoidalPositionEmbeddings(T* out,
                                              functional::Shape outShape,
                                              int start) {
  using namespace functional;

  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  float numTimescales = (float)cols / 2.f;
  float logTimescaleIncrement = Ops<float>::log(10000.f) / (numTimescales - 1.f);

  for(int bid = 0; bid < rows; bid += gridDim.x) { // loop over blocks of rows
    int j = bid + blockIdx.x; // blockIdx.x - row index (within block of rows)
    if(j < rows) { // compute softmax over one row, row elements distributed over threads
      T* outRow = out + j * cols; // pointer to row data
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float v = (float)(j + start) * Ops<float>::exp((float)(i % (int)numTimescales) * -logTimescaleIncrement);
          outRow[i] = (T)(i < (int)numTimescales ? Ops<float>::sin(v) : Ops<float>::cos(v));
        }
      }
    }
  }
}

void SinusoidalPositionEmbeddings(Tensor out, int start) {
  cudaSetDevice(out->getDeviceId().no);

  size_t rows = out->shape().elements() / out->shape().back();
  size_t cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);

  if(out->type() == Type::float32) {
    gSinusoidalPositionEmbeddings<float><<<blocks, threads>>>(out->data<float>(), out->shape(), start);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gSinusoidalPositionEmbeddings<half><<<blocks, threads>>>(out->data<half>(), out->shape(), start);
#endif
  } else {
    ABORT("SinusoidalPositionEmbeddings not implemented for type {}", out->type());
  }
}


// @TODO: refactor to reuse code from softmax, add comments
template <typename T, typename AccType = float>
__global__ void gLogSoftmax(T* out,
                            const functional::Shape outShape,
                            const T* in) {

  using namespace functional;

  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* so = out + j * cols;
      const T* sp = in + j * cols;

      // CUDA complains if type or size of shared memory changes, keep size constant.
      extern __shared__ uint8_t _sharedBytes[];
      T* _share = (T*)_sharedBytes;
      AccType* _shareAccType = (AccType*)_sharedBytes;

      T* _max = _share; // 16-bit is ok for max if applicable
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
      T max = _max[0];
      __syncthreads();

      AccType* _sum = _shareAccType; // keep AccType for accumulation

      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          T sm = sp[id] - max;
          AccType ex = Ops<AccType>::exp(sm); // sum with AccType
          so[id] = sm;
          _sum[threadIdx.x] += ex; // sum with AccType
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
      AccType sum = _sum[0];
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          so[id] -= (T)Ops<AccType>::log(sum); // take log at the end and convert
      }
    }
    __syncthreads();
  }
}

void LogSoftmax(Tensor out, Tensor in) {
  cudaSetDevice(out->getDeviceId().no);

  size_t m = out->shape().elements() / out->shape().back();
  size_t k = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads; // use float32 as accumulation type

  if(in->type() == Type::float32) {
    gLogSoftmax<float, float><<<blocks, threads, shared>>>(out->data<float>(), out->shape(), in->data<float>());
#if COMPILE_FP16
  } else if (in->type() == Type::float16) {
    gLogSoftmax<half, float><<<blocks, threads, shared>>>(out->data<half>(), out->shape(), in->data<half>());
#endif
  } else {
    ABORT("LogSoftmax not implemented for type {}", in->type());
  }
}

///////////////////////////////////////////////////////

template <typename T, typename AccType = float>
__global__ void gSoftmaxGrad(T* grad,
                             const T* adj,
                             const T* val,
                             const int rows,
                             const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {

      extern __shared__ uint8_t _sharedBytes[];
      AccType* _sum = (AccType*)_sharedBytes;

      T* gradRow = grad + j * cols;
      const T* adjRow = adj + j * cols;
      const T* valRow = val + j * cols;
      _sum[threadIdx.x] = (AccType)0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (AccType)valRow[id] * (AccType)adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip]; // accumulates in AccType
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType val = (AccType)valRow[id] * ((AccType)adjRow[id] - _sum[0]);
          if(val)
            gradRow[id] += (T)val;
        }
      }
    }
    __syncthreads();
  }
}

// @TODO: refactor with logsoftmax, add math
void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  cudaSetDevice(adj->getDeviceId().no);
  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad->shape().elements() / grad->shape().back();
  int k = grad->shape().back();

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads;

  if(grad->type() == Type::float32) {
    gSoftmaxGrad<float, float><<<blocks, threads, shared>>>(
      grad->data<float>(), adj->data<float>(), val->data<float>(), m, k);
#if COMPILE_FP16
  } else if (grad->type() == Type::float16) {
    // Accumulate into float
    gSoftmaxGrad<half, float><<<blocks, threads, shared>>>(
      grad->data<half>(), adj->data<half>(), val->data<half>(), m, k);
#endif
  } else {
    ABORT("SoftmaxGrad not implemented for type {}", grad->type());
  }
}

template <typename T, typename AccType = float>
__global__ void gLogSoftmaxGrad(T* grad,
                                const T* adj,
                                const T* val,
                                const int rows,
                                const int cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      extern __shared__ uint8_t _sharedBytes[];
      AccType* _sum = (AccType*)_sharedBytes;

      T* gradRow = grad + j * cols;
      const T* adjRow = adj + j * cols;
      const T* valRow = val + j * cols;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (AccType)adjRow[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sum[threadIdx.x] += _sum[threadIdx.x + skip]; // AccType
        len = (len + 1) >> 1;
      }
      __syncthreads();
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols)
          gradRow[id] += (T)((AccType)adjRow[id] - (functional::Ops<AccType>::exp((AccType)valRow[id]) * _sum[0]));
      }
    }
    __syncthreads();
  }
}

void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  cudaSetDevice(adj->getDeviceId().no);

  // grad and val are both m-by-k matrices, passed as input.
  // A weighted average of each row of grad (according to the weights
  // specified in val) is computed and subtracted from Out.
  // adj is multiplied for each element to get backward step in autodiff
  int m = grad->shape().elements() / grad->shape().back();
  int k = grad->shape().back();

  int blocks = std::min(MAX_BLOCKS, m);
  int threads = std::min(MAX_THREADS, k);
  int shared = sizeof(float) * threads; // Use float32 as accumulation type

  if(grad->type() == Type::float32) {
    gLogSoftmaxGrad<float, float><<<blocks, threads, shared>>>(
      grad->data<float>(), adj->data<float>(), val->data<float>(), m, k);
#if COMPILE_FP16
  } else if (grad->type() == Type::float16) {
    // accumulate into float
    gLogSoftmaxGrad<half, float><<<blocks, threads, shared>>>(
      grad->data<half>(), adj->data<half>(), val->data<half>(), m, k);
#endif
  } else {
    ABORT("LogSoftmaxGrad not implemented for type {}", grad->type());
  }
}

///////////////////////////////////////////////////////

template <typename T>
__global__ void gCopyRows(T* out,
                          const T* in,
                          size_t cols,
                          const IndexType* sourceRowIdx,
                          size_t rows) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      size_t dstId = j;
      size_t srcId = sourceRowIdx[j];

      T* rowOut = out + dstId * cols;
      const T* rowIn = in + srcId * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = rowIn[i];
      }
    }
  }
}

void CopyRows(Tensor out,
              const Tensor in,
              const Tensor indices) {

  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  size_t cols = in->shape().back();
  size_t rowsToCopy = indices->size();

  int threads = std::min(MAX_THREADS, (int)cols);
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);

  if(out->type() == Type::float32) {
    gCopyRows<<<blocks, threads>>>(
      out->data<float>(), in->data<float>(), cols, indices->data<IndexType>(), rowsToCopy);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gCopyRows<<<blocks, threads>>>(
      out->data<half>(), in->data<half>(), cols, indices->data<IndexType>(), rowsToCopy);
#endif
  } else {
    ABORT("CopyRows not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gPasteRows(T* out,
                           const T* in,
                           size_t cols,
                           const IndexType* targetRowIdx,
                           size_t rows) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x; // index into 'indices' vector
    if(j < rows) {
      size_t dstId = targetRowIdx[j];
      size_t srcId = j;

      T* rowOut = out + dstId * cols;
      const T* rowIn = in + srcId * cols;

      // aggregate the entire row
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x; // column index   --@TODO: column index should be called 'j'
        if(i < cols) {
          // Note: atomicAdd() not needed if number of blocks is 1. Avoid it because it is slow for fp16.
          if (gridDim.x == 1)
            rowOut[i] += rowIn[i];
          else
            atomics::atomicAdd(rowOut + i, rowIn[i]);
        }
      }
    }
  }
}

void PasteRows(Tensor out,
               const Tensor in,
               const Tensor indices) {

  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  size_t cols = in->shape().back();
  size_t rowsToCopy = indices->size();

  int threads = std::min(MAX_THREADS, (int)cols);
#if DETERMINISTIC
  // If we only use one block, then each core operates on a different column,
  // hence the summation becomes deterministic.
  // However, we only use e.g. 512 cores out of possibly 3000+, so this will be
  // 6 x slower in this example.
  int blocks = 1;
#else
  int blocks = std::min(MAX_BLOCKS, (int)rowsToCopy);
#endif

  if(out->type() == Type::float32) {
    gPasteRows<<<blocks, threads>>>(
      out->data<float>(), in->data<float>(), cols, indices->data<IndexType>(), rowsToCopy);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gPasteRows<<<blocks, threads>>>(
      out->data<half>(), in->data<half>(), cols, indices->data<IndexType>(), rowsToCopy);
#endif
  } else {
    ABORT("CopyRows not implemented for type {}", out->type());
  }
}

/////////////

template <typename T>
__global__ void gCopyCols(T* out,
                          const T* in,
                          size_t rows,
                          size_t colsIn,
                          const IndexType* sourceColIdx,
                          size_t colsOut) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const T* rowIn = in + j * colsIn;
      T* rowOut = out + j * colsOut;

      for(int tid = 0; tid < colsOut; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsOut)
          rowOut[i] = rowIn[sourceColIdx[i]];
      }
    }
  }
}

void CopyCols(Tensor out, const Tensor in, const Tensor indices) {
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  size_t rows = in->shape().elements() / in->shape().back();
  size_t cols = in->shape().back();

  size_t colsToCopy = indices->size();

  int threads = std::min(MAX_THREADS, (int)colsToCopy);
  int blocks = std::min(MAX_BLOCKS, (int)rows);

  if(out->type() == Type::float32) {
    gCopyCols<<<blocks, threads>>>(
      out->data<float>(), in->data<float>(), rows, cols, indices->data<IndexType>(), colsToCopy);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gCopyCols<<<blocks, threads>>>(
      out->data<half>(), in->data<half>(), rows, cols, indices->data<IndexType>(), colsToCopy);
#endif
  } else {
    ABORT("CopyCols not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gPasteCols(T* out,
                           const T* in,
                           size_t rows,
                           size_t colsOut,
                           const IndexType* targetColIdx,
                           size_t colsIn) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const T* rowIn = in + j * colsIn;
      T* rowOut = out + j * colsOut;

      for(int tid = 0; tid < colsIn; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < colsIn)
          rowOut[targetColIdx[i]] += rowIn[i]; // @TODO: atomicAdd?
      }
    }
  }
}

void PasteCols(Tensor out,
               const Tensor in,
               const Tensor indices) {
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  size_t rows = in->shape().elements() / in->shape().back();
  size_t cols = in->shape().back();

  size_t colsToCopy = indices->size();

  int threads = std::min(MAX_THREADS, (int)colsToCopy);
  int blocks = std::min(MAX_BLOCKS, (int)rows);

  if(out->type() == Type::float32) {
    gPasteCols<<<blocks, threads>>>(
      out->data<float>(), in->data<float>(), rows, cols, indices->data<IndexType>(), colsToCopy);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gPasteCols<<<blocks, threads>>>(
      out->data<half>(), in->data<half>(), rows, cols, indices->data<IndexType>(), colsToCopy);
#endif
  } else {
    ABORT("PasteCols not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gSelect(T* out,
                        functional::Shape outShape,
                        const T* in,
                        const functional::Shape inShape,
                        int axis,
                        const IndexType* d_indices,
                        const functional::Shape idxShape) {
  int length = outShape.elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      outShape.dims(index, dims);
      int idxIndex = idxShape.bindex(dims); // broadcast index into indices tensor
      dims[axis] = (int)d_indices[idxIndex];    
      int inIndex = inShape.index(dims);
      out[index] = in[inIndex];
    }
  }
}

template <bool add, typename T>
__global__ void gInsert(T* out,
                        functional::Shape outShape,
                        const T* in,
                        const functional::Shape inShape,
                        int axis,
                        const IndexType* d_indices,
                        const functional::Shape idxShape) {
  int length = inShape.elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      inShape.dims(index, dims);
      int idxIndex = idxShape.bindex(dims); // broadcast index into indices tensor
      dims[axis] = (int)d_indices[idxIndex];    
      int outIndex = outShape.index(dims);
      if(add)
        out[outIndex] += in[index]; // this is probably wrong, atomicAdd?
      else
        out[outIndex] = in[index];     
    }
  }
}

void Select(Tensor out,
            const Tensor in,
            const Tensor indices,
            int axis) {
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  int axisGPU = axis + functional::Shape::size() - out->shape().size();

  if(out->type() == Type::float32) {
    gSelect<<<blocks, threads>>>(out->data<float>(),
                                 out->shape(),
                                 in->data<float>(),
                                 in->shape(),
                                 axisGPU,
                                 indices->data<IndexType>(), 
                                 indices->shape());
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gSelect<<<blocks, threads>>>(out->data<half>(),
                                 out->shape(),
                                 in->data<half>(),
                                 in->shape(),
                                 axisGPU,
                                 indices->data<IndexType>(),
                                 indices->shape());
#endif
  } else if(out->type() == Type::uint32) {
    gSelect<<<blocks, threads>>>(out->data<IndexType>(),
                                 out->shape(),
                                 in->data<IndexType>(),
                                 in->shape(),
                                 axisGPU,
                                 indices->data<IndexType>(), 
                                 indices->shape());
  } else {
    ABORT("Select not implemented for type {}", out->type());
  }
}

template <bool add>
void Insert(Tensor out,
            const Tensor in,
            const Tensor indices,
            int axis) {
  matchOrAbort<IndexType>(indices->type());
  cudaSetDevice(in->getDeviceId().no);

  int length = in->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  int axisGPU = axis + functional::Shape::size() - out->shape().size();

  if(out->type() == Type::float32) {
    gInsert<add><<<blocks, threads>>>(out->data<float>(),
                                               out->shape(),
                                               in->data<float>(),
                                               in->shape(),
                                               axisGPU,
                                               indices->data<IndexType>(),
                                               indices->shape());
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gInsert<add><<<blocks, threads>>>(out->data<half>(),
                                               out->shape(),
                                               in->data<half>(),
                                               in->shape(),
                                               axisGPU,
                                               indices->data<IndexType>(),
                                               indices->shape());
#endif
  } else {
    ABORT("Insert not implemented for type {}", out->type());
  }
}

template void Insert<true>(Tensor out, const Tensor in, const Tensor indices, int axis);
template void Insert<false>(Tensor out, const Tensor in, const Tensor indices, int axis);

template <typename T>
__global__ void gGRUFastForward(T* out,
                                const T* state,
                                const T* xW,
                                const T* sU,
                                const T* b,
                                const T* mask,
                                size_t rows,
                                size_t cols,
                                bool final) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];
      T* rowOut = out + j * cols;
      const T* rowState = state + j * cols;

      const T* xWrow = xW + j * cols * 3;
      const T* sUrow = sU + j * cols * 3;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float r = functional::Ops<float>::sigmoid((float)xWrow[i] + (float)sUrow[i] + (float)b[i]);

          int k = i + cols;

          float z = functional::Ops<float>::sigmoid((float)xWrow[k] + (float)sUrow[k] + (float)b[k]);

          int l = i + 2 * cols;
          float h;
          if(final)
            h = functional::Ops<float>::tanh((float)xWrow[l] + ((float)sUrow[l] + (float)b[l]) * r);
          else
            h = functional::Ops<float>::tanh((float)xWrow[l] + (float)sUrow[l] * r + (float)b[l]);

          float out = (1.f - z) * h + z * (float)rowState[i];
          rowOut[i] = (T)(m * out + (1.f - m) * (float)rowState[i]);
        }
      }
    }
  }
}

void GRUFastForward(Tensor out, std::vector<Tensor> inputs, bool final) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  if(out->type() == Type::float32) {
    gGRUFastForward<<<blocks, threads>>>(
        out->data<float>(),                                // output
        inputs[0]->data<float>(),                          // state
        inputs[1]->data<float>(),                          // xW
        inputs[2]->data<float>(),                          // sU
        inputs[3]->data<float>(),                          // b
        inputs.size() > 4 ? inputs[4]->data<float>() : 0,  // mask
        rows,
        cols,
        final);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gGRUFastForward<<<blocks, threads>>>(
        out->data<half>(),                                // output
        inputs[0]->data<half>(),                          // state
        inputs[1]->data<half>(),                          // xW
        inputs[2]->data<half>(),                          // sU
        inputs[3]->data<half>(),                          // b
        inputs.size() > 4 ? inputs[4]->data<half>() : 0,  // mask
        rows,
        cols,
        final);
#endif
  } else {
    ABORT("GRUFastForward not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gGRUFastBackward(T* outState,
                                 T* outXW,
                                 T* outSU,
                                 T* outB,
                                 const T* state,
                                 const T* xW,
                                 const T* sU,
                                 const T* b,
                                 const T* mask,
                                 const T* adj,
                                 size_t rows,
                                 size_t cols,
                                 bool final) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float m = !mask || mask[j];

      T* rowOutState = outState + j * cols;
      T* rowOutXW = outXW + j * cols * 3;
      T* rowOutSU = outSU + j * cols * 3;
      T* rowOutB  = outB ? outB + j * cols * 3 : nullptr;

      const T* rowState = state + j * cols;
      const T* rowXW    = xW + j * cols * 3;
      const T* rowSU    = sU + j * cols * 3;
      const T* rowAdj   = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + cols;
          int l = i + 2 * cols;

          float r = functional::Ops<float>::sigmoid((float)rowXW[i] + (float)rowSU[i] + (float)b[i]);
          float z = functional::Ops<float>::sigmoid((float)rowXW[k] + (float)rowSU[k] + (float)b[k]);

          float h;
          if(final)
            h = functional::Ops<float>::tanh((float)rowXW[l] + ((float)rowSU[l] + (float)b[l]) * r);
          else
            h = functional::Ops<float>::tanh((float)rowXW[l] + (float)rowSU[l] * r + (float)b[l]);

          float adj = rowAdj[i];

          float t = (1.f - z) * (1.f - h * h);

          // df/ds
          if(outState)
            rowOutState[i] += (T)((m * z - m + 1.f) * adj);

          // df/d(xW_r) ...
          float dfdxW_r = m * r * (1.f - r) * t * adj;
          if(final)
            dfdxW_r *= (float)rowSU[l] + (float)b[l];
          else
            dfdxW_r *= (float)rowSU[l];
          if(outXW)
            rowOutXW[i] += (T)dfdxW_r;
          if(outSU)
            rowOutSU[i] += (T)dfdxW_r;
          if(outB)
            rowOutB[i] += (T)dfdxW_r;

          // df/d(xW_z) ...
          float dfdxW_z = m * (1.f - z) * z * ((float)rowState[i] - h) * adj;
          if(outXW)
            rowOutXW[k] += (T)dfdxW_z;
          if(outSU)
            rowOutSU[k] += (T)dfdxW_z;
          if(outB)
            rowOutB[k] += (T)dfdxW_z;

          // df/d(xW_x) ...
          float dfdxW_x = m * t * adj;
          if(outXW)
            rowOutXW[l] += (T)dfdxW_x;
          if(outSU)
            rowOutSU[l] += (T)(dfdxW_x * r);
          if(outB)
            if(final)
              rowOutB[l] += (T)(dfdxW_x * r);
            else
              rowOutB[l] += (T)dfdxW_x;
        }
      }
    }
  }
}

void GRUFastBackward(Ptr<Allocator> allocator,
                     std::vector<Tensor> outputs,
                     std::vector<Tensor> inputs,
                     Tensor adj,
                     bool final) {
  cudaSetDevice(adj->getDeviceId().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  Tensor tempGradBias, tempOnes; 
  MemoryPiece::PtrType tempGradBiasMemory, tempOnesMemory;
  if(outputs[3]) {
    Shape memShape = {rows, outputs[3]->shape()[-1]};

    tempGradBiasMemory = allocator->alloc(memShape.elements() * sizeOf(outputs[3]->type()));
    tempGradBias = TensorBase::New(tempGradBiasMemory, memShape, outputs[3]->type(), outputs[3]->getBackend());
    tempGradBias->set(0.f);

    tempOnesMemory = allocator->alloc(rows * sizeOf(outputs[3]->type()));
    tempOnes = TensorBase::New(tempOnesMemory, Shape({1, rows}), outputs[3]->type(), outputs[3]->getBackend());
    tempOnes->set(1.f);
  }

  if(adj->type() == Type::float32) {
    gGRUFastBackward<<<blocks, threads>>>(
        outputs[0] ? outputs[0]->data<float>() : 0,        // state - adj
        outputs[1] ? outputs[1]->data<float>() : 0,        // xW - adj
        outputs[2] ? outputs[2]->data<float>() : 0,        // sU - adj
        outputs[3] ? tempGradBias->data<float>() : 0,      // b - adj
        inputs[0]->data<float>(),                          // state
        inputs[1]->data<float>(),                          // xW
        inputs[2]->data<float>(),                          // sU
        inputs[3]->data<float>(),                          // b
        inputs.size() > 4 ? inputs[4]->data<float>() : 0,  // mask
        adj->data<float>(),
        rows,
        cols,
        final);
#if COMPILE_FP16
  } else if (adj->type() == Type::float16) {
    gGRUFastBackward<<<blocks, threads>>>(
        outputs[0] ? outputs[0]->data<half>() : 0,        // state - adj
        outputs[1] ? outputs[1]->data<half>() : 0,        // xW - adj
        outputs[2] ? outputs[2]->data<half>() : 0,        // sU - adj
        outputs[3] ? tempGradBias->data<half>() : 0,        // b - adj
        inputs[0]->data<half>(),                          // state
        inputs[1]->data<half>(),                          // xW
        inputs[2]->data<half>(),                          // sU
        inputs[3]->data<half>(),                          // b
        inputs.size() > 4 ? inputs[4]->data<half>() : 0,  // mask
        adj->data<half>(),
        rows,
        cols,
        final);
#endif
  } else {
    ABORT("gGRUFastBackward not implemented for type {}", adj->type());
  }

  // We use this go get rid of the atomicAdd and perform a reduce of the gradients afterwards.
  // This is much faster for fp16 which seems to have a broken atomicAdd implementation.
  // We reduce bias gradients with a matrix multiply, but use a 32-bit compute type. 
  // This preserves precision with larger batches where all batch entries reduce into a single vector.
  // See also AffineNodeOp where we do the same for biases
  if(outputs[3]) {
    gpu::Prod(outputs[3], tempOnes, tempGradBias, false, false, 1, 1, Type::float32); // beta set to one to add
    allocator->free(tempGradBiasMemory);
    allocator->free(tempOnesMemory);
  }
}

template <typename T, typename AccType = float>
__global__ void gCrossEntropyPick(AccType* out,
                                  const functional::Shape outShape,
                                  const T* in,
                                  const functional::Shape inShape,
                                  const IndexType* pick,
                                  AccType labelSmoothingAlpha = AccType(0.f)) {
  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

  extern __shared__ uint8_t _sharedBytes[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const T* sp = in + j * cols;

      T* _max = (T*)_sharedBytes;
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
      T max = _max[0];
      __syncthreads();

      AccType* _acc = (AccType*)_sharedBytes;
      _acc[2 * threadIdx.x    ] = (AccType)0.0f;
      _acc[2 * threadIdx.x + 1] = (AccType)0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _acc[2 * threadIdx.x    ] += functional::Ops<AccType>::exp(sp[id] - max);
          _acc[2 * threadIdx.x + 1] += (AccType)(sp[id] - max);
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _acc[2 * threadIdx.x    ] += _acc[2 * (threadIdx.x + skip)    ];
          _acc[2 * threadIdx.x + 1] += _acc[2 * (threadIdx.x + skip) + 1];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      AccType sumexp = _acc[0];

      // H(u, p) = 1/N * logsoftmax(h) = mean(h - max) - log(sum(exp(h - max)))
      AccType mean = _acc[1] / (AccType)cols; // mean(h - max)

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id == (int)pick[j]) {
          AccType logsumexp = functional::Ops<AccType>::log(sumexp);
          AccType ce = logsumexp - (AccType)sp[id] + (AccType)max; // cross-entropy    H(y^, p)
          AccType ls = logsumexp - mean;                           // label smoothing  H(u, p)
          out[j] = (1.f - labelSmoothingAlpha) * ce + labelSmoothingAlpha * ls;  // (1 - alpha) * H(y^, p) + alpha * H(u, p)
        }
      }
    }
    __syncthreads();
  }
}

// In each j-th row, take the corresponding j-th label index i from indices and compute:
// For each vocabulary item v, the only non-zero element in a row in the sum is the item
// that matches the label indexed by i (the picked element).
// C = sum_{v in V}(-logsoftmax(A) * delta(v, i) = -logsoftmax(A)[i]
void CrossEntropyPick(Tensor out, Tensor in, Tensor indices, float labelSmoothingAlpha) {
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = sizeof(float) * threads * 2; // Use float32 as accumulation type

  if(out->type() == Type::float32 && in->type() == Type::float32) {
    gCrossEntropyPick<float, float><<<blocks, threads, shared>>>(
      out->data<float>(), out->shape(), in->data<float>(), in->shape(), indices->data<IndexType>(), labelSmoothingAlpha);
#if COMPILE_FP16
  } else if(out->type() == Type::float32 && in->type() == Type::float16) {
    gCrossEntropyPick<half, float><<<blocks, threads, shared>>>(
      out->data<float>(), out->shape(), in->data<half>(), in->shape(), indices->data<IndexType>(), labelSmoothingAlpha);
#endif
  } else {
    ABORT("CrossEntropyPick not implemented for input type {} and output type{}", in->type(), out->type());
  }
}

template <typename T, typename AccType = float>
__global__ void gCrossEntropyPickBackward(T* out,
                                          const functional::Shape outShape,
                                          const AccType* adj,
                                          const T* in,
                                          const IndexType* pick,
                                          AccType labelSmoothingAlpha = AccType(0.f)) {
  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

  extern __shared__ uint8_t _sharedBytes[];

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const T* sp = in + j * cols;
      T* so = out + j * cols;
      T* _max = (T*)_sharedBytes;
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
      T max = _max[0];
      __syncthreads();

      AccType* _sum = (AccType*)_sharedBytes;
      _sum[threadIdx.x] = 0.0;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType ex = functional::Ops<AccType>::exp(sp[id] - max);
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
          AccType sub = (AccType)(id == (int)pick[j]);
          AccType dce = functional::Ops<AccType>::exp(sp[id] - max) / _sum[0] - sub;
          AccType dls = labelSmoothingAlpha * (sub - 1.f / (AccType)cols);
          so[id] += (T)(adj[j] * (dce + dls));
        }
      }
    }
    __syncthreads();
  }
}

void CrossEntropyPickBackward(Tensor out, Tensor adj, Tensor a, Tensor indices, float labelSmoothingAlpha) {
  matchOrAbort<IndexType>(indices->type());

  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = sizeof(float) * threads; // use float as accumulation type

  if(out->type() == Type::float32 && adj->type() == Type::float32) {
    gCrossEntropyPickBackward<float, float><<<blocks, threads, shared>>>(
      out->data<float>(), out->shape(), adj->data<float>(), a->data<float>(), indices->data<IndexType>(), labelSmoothingAlpha);
#if COMPILE_FP16
  } else if(out->type() == Type::float16 && adj->type() == Type::float32) {
    gCrossEntropyPickBackward<half, float><<<blocks, threads, shared>>>(
      out->data<half>(), out->shape(), adj->data<float>(), a->data<half>(), indices->data<IndexType>(), labelSmoothingAlpha);
#endif
  } else {
    ABORT("CrossEntropyPickBackward not implemented for type {} and adjoint type {}", out->type(), adj->type());
  }
}

// computes the L2Norm of tensor and returns value as flaot on the CPU, 
// this is mostly used for diagnostic purposes and gradient clipping
float L2Norm(Tensor in, Ptr<Allocator> allocator) { // @TODO: reverse order of arguments
  cudaSetDevice(in->getDeviceId().no);

  int size = in->shape().elements();
  int threads = std::min(MAX_THREADS, size);
  int blocks  = std::min(MAX_BLOCKS, size / threads + (size % threads != 0));

  using namespace functional;
  float l2Norm;
  if(in->type() == Type::float32) {
    l2Norm = std::sqrt(AggregateAllAndReturn</*ElementType=*/float, /*AccType=*/float>(allocator, /*functor=*/_1 * _1, /*aggInit=*/0.f, /*aggFunctor=*/_1 + _2, /*scale=*/1.f, in));
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    l2Norm = std::sqrt(AggregateAllAndReturn</*ElementType=*/half, /*AccType=*/float>(allocator, /*functor=*/_1 * _1, /*aggInit=*/0.f, /*aggFunctor=*/_1 + _2, /*scale=*/1.f, in));
#endif
  } else {
    ABORT("L2Norm not implemented for type {}", in->type());
  }
  return l2Norm;
}

template <typename T, typename AccType = float>
__global__ void gAtt(T* out,
                     const T* va,
                     const T* ctx,
                     const T* state,
                     int m,  // total rows (batch x time x beam)
                     int k,  // depth
                     int b,  // batch size
                     int t   // time of ctx
) {
  int rows = m;
  int cols = k;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      const T* vaRow = va;
      const T* ctxRow = ctx + (j % (b * t)) * cols;
      const T* stateRow = state + ((j / (b * t)) * b + j % b) * cols;

      extern __shared__ AccType _share[];
      AccType* _sum = _share;

      _sum[threadIdx.x] = 0.f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType z = (AccType)ctxRow[id] + (AccType)stateRow[id];
          AccType ex = functional::Ops<AccType>::tanh(z) * (AccType)vaRow[id];
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
      out[j] = (T)_sum[0];
    }
    __syncthreads();
  }
}

void Att(Tensor out, Tensor va, Tensor context, Tensor state) {
  cudaSetDevice(out->getDeviceId().no);

  size_t totalRows       = out->shape().elements() / out->shape().back(); // number of rows
  size_t modelDim        = context->shape()[-1];                          // number of cols
  size_t batchDim        = context->shape()[-2];
  size_t contextWordsDim = context->shape()[-3];

  int blocks = std::min(MAX_BLOCKS, (int)totalRows);   
  int threads = std::min(MAX_THREADS, (int)modelDim);
  int shared = sizeof(float) * threads;

  if(out->type() == Type::float32) {
    gAtt<float, float><<<blocks, threads, shared>>>(
      out->data<float>(), va->data<float>(), context->data<float>(), state->data<float>(), totalRows, modelDim, batchDim, contextWordsDim);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gAtt<half, float><<<blocks, threads, shared>>>(
      out->data<half>(), va->data<half>(), context->data<half>(), state->data<half>(), totalRows, modelDim, batchDim, contextWordsDim);
#endif
  } else {
    ABORT("gAtt not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gAttBack(T* gVa,
                         T* gContext,
                         T* gState,
                         const T* va,
                         const T* context,
                         const T* state,
                         const T* adj,
                         int m,  // rows
                         int k,  // cols
                         int n   // batch size
) {
  int rows = m;
  int cols = k;
  for(int bid = 0; bid < m; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* gcRow = gContext + j * cols;
      T* gsRow = gState + (j % n) * cols;

      const T* cRow = context + j * cols;
      const T* sRow = state + (j % n) * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          T z = cRow[id] + sRow[id];

          T t = functional::Ops<T>::tanh(z);
          T r = va[id] * ((T)1.f - t * t);

          gcRow[id] += r * adj[j]; // atomicAdd? reasons for instabilities?
          gsRow[id] += r * adj[j];
          atomics::atomicAdd(gVa + id, t * adj[j]); // @TODO: get rid of atomicAdd via Matmul
        }
      }
    }
  }
}

void AttBack(Tensor gVa,
             Tensor gContext,
             Tensor gState,
             Tensor va,
             Tensor context,
             Tensor state,
             Tensor adj) {
  cudaSetDevice(adj->getDeviceId().no);

  size_t m = adj->shape().elements() / adj->shape()[-1];
  size_t k = context->shape()[-1];
  size_t n = context->shape()[-2];

  int blocks = std::min(MAX_BLOCKS, (int)n);
  int threads = std::min(MAX_THREADS, (int)k);

  if(gVa->type() == Type::float32) {
    gAttBack<<<blocks, threads>>>(gVa->data<float>(),
                                  gContext->data<float>(),
                                  gState->data<float>(),
                                  va->data<float>(),
                                  context->data<float>(),
                                  state->data<float>(),
                                  adj->data<float>(),
                                  m,
                                  k,
                                  n);
#if COMPILE_FP16
  } else if (gVa->type() == Type::float16) {
    gAttBack<<<blocks, threads>>>(gVa->data<half>(),
                                  gContext->data<half>(),
                                  gState->data<half>(),
                                  va->data<half>(),
                                  context->data<half>(),
                                  state->data<half>(),
                                  adj->data<half>(),
                                  m,
                                  k,
                                  n);
#endif
  } else {
    ABORT("gAttBack not implemented for type {}", gVa->type());
  }
}

template <typename T, typename AccType = float>
__global__ void gLNormalization(T* out,
                                const T* in,
                                const T* gamma,
                                const T* beta,
                                int rows,
                                int cols,
                                AccType eps = 1e-9) {
  extern __shared__ uint8_t _sharedBytes[];
  AccType* _shareAccType = (AccType*)_sharedBytes;

  AccType N = cols;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* yRow       = out + j * cols;
      const T* xRow =  in + j * cols;

      AccType* _sum = _shareAccType; // accumulate into floats
      _sum[threadIdx.x] = (AccType)0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          _sum[threadIdx.x] += (AccType)xRow[id];
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
      AccType mean = _sum[0] / N;
      __syncthreads();

      AccType* _sqSum = _shareAccType;

      _sqSum[threadIdx.x] = (AccType)0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType xv = (AccType)xRow[id];
          AccType ex = xv - mean;
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
      AccType sigma = functional::Ops<AccType>::sqrt(_sqSum[0] / N + eps); // all AccType
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType gammav = (AccType)gamma[id];
          AccType xv     = (AccType)xRow[id];
          AccType betav  = beta ? (AccType)beta[id] : (AccType)0.f;
          AccType lv     = (xv - mean) / sigma;
          AccType y      = gammav * lv + betav;
          yRow[id]       = (T)y;
        }
      }
    }
    __syncthreads();
  }
}

void LayerNormalization(Tensor out,
                        Tensor in,
                        Tensor gamma,
                        Tensor beta,
                        float eps) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = threads * sizeof(float);

  if(out->type() == Type::float32) {
    gLNormalization<float, float><<<blocks, threads, shared>>>(out->data<float>(),
                                                 in->data<float>(),
                                                 gamma->data<float>(),
                                                 beta ? beta->data<float>() : nullptr,
                                                 rows,
                                                 cols,
                                                 eps);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gLNormalization<half, float><<<blocks, threads, shared>>>(out->data<half>(),
                                                 in->data<half>(),
                                                 gamma->data<half>(),
                                                 beta ? beta->data<half>() : nullptr,
                                                 rows,
                                                 cols,
                                                 eps);
#endif
  } else {
    ABORT("LayerNormalization not implemented for type {}", out->type());
  }
}

template <typename T, typename AccType = float>
__global__ void gLayerNormalizationGrad(T* gradX,
                                        T* gradGamma,
                                        T* adj,
                                        T* y,
                                        T* x,
                                        T* gamma,
                                        T* beta,
                                        int rows,
                                        int cols,
                                        AccType eps = 1e-9) {
  extern __shared__ uint8_t sharedBytes[];
  AccType* shared = (AccType*)sharedBytes;

  AccType N = cols;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      AccType* sum_adj   = shared;                   // sum of gradient coming in
      AccType* sum_adj_l = shared +     blockDim.x;  // sum of gradient coming in times layerNorm from value
      AccType* sum_x     = shared + 2 * blockDim.x;  // sum of input value x
      AccType* sum_sqr   = shared + 3 * blockDim.x;  // sum of (x - mean)^2

      const T* xRow   =   x + j * cols;
      const T* yRow   =   y + j * cols;
      const T* adjRow = adj + j * cols;

      sum_x[threadIdx.x]     = (AccType)0.0f;
      sum_adj[threadIdx.x]   = (AccType)0.0f;
      sum_adj_l[threadIdx.x] = (AccType)0.0f;
      sum_sqr[threadIdx.x]   = (AccType)0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType xv     = xRow[id];
          AccType yv     = yRow[id];
          AccType betav  = beta ? (AccType)beta[id] : (AccType)0.f;
          AccType gammav = (AccType)gamma[id];
          AccType adjv   = adjRow[id];
          AccType lv     = (yv - betav) / gammav; // go back to LN(x) from scaled and shifted version for accumulation

          sum_x[threadIdx.x]     += xv;
          sum_adj_l[threadIdx.x] += adjv * lv;
          sum_adj[threadIdx.x]   += adjv;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_x[threadIdx.x]     += sum_x[threadIdx.x     + skip]; // Accumulates in AccType
          sum_adj[threadIdx.x]   += sum_adj[threadIdx.x   + skip]; // Accumulates in AccType
          sum_adj_l[threadIdx.x] += sum_adj_l[threadIdx.x + skip]; // Accumulates in AccType
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      AccType mean = sum_x[0] / N;
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType xv = xRow[id];
          AccType ex = xv - mean;
          sum_sqr[threadIdx.x] += ex * ex;
        }
      }

      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          sum_sqr[threadIdx.x] += sum_sqr[threadIdx.x + skip]; // Accumulates in AccType
        len = (len + 1) >> 1;
      }
      __syncthreads();
      AccType sigma = functional::Ops<AccType>::sqrt(sum_sqr[0] / N + eps);
      __syncthreads();

      // Jacobian of layer norm
      // J = [ \frac{1}{N\sigma} (N\delta_{ij} - l_i l_j - 1) ]_{ij}
      // J * a = dC/dx_i = ( N a_i - l_i \sum_j l_j a_j - \sum_j a_j ) / (N \sigma)

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {

          AccType xv     = xRow[id];
          AccType gammav = (AccType)gamma[id];
          AccType adjv   = adjRow[id];
          AccType lv     = (xv - mean) / sigma;

          AccType gradLv = N * adjv - lv * sum_adj_l[0] - sum_adj[0];
          gradLv        /= N * sigma; 

          AccType gradXv = gammav * gradLv;

          // Keep LN gradient between [-1000, 1000] for TensorOps, this currently used for making values fit into fp16. This wil also clip inf. 
          // @TODO: to be fixed and removed.
          AccType sign = functional::Ops<AccType>::sgn(gradXv);
          AccType cutoff = (AccType)1000.f; // @TODO: expose this somehow as an option? or better: make obsolete.
          gradXv = functional::Ops<AccType>::abs(gradXv) > cutoff ? sign * cutoff : gradXv; // if gradXv is NaN the value return is NaN too because NaN > value is false.

          // @TODO: frankly, this is embarrasing and should rather be removed or optional? It does help for low precision computation though. Maybe turn into option?
          gradXv = isnan(gradXv) ? 0.f : gradXv; // turn NaN into 0.

          T* gradXRow      = gradX     + j * cols;
          gradXRow[id]    += (T)(gradXv);

          T* gradGammaRow  = gradGamma + j * cols;
          // assignment is correct here as this gets summed up
          // in the next kernel via matrix product
          gradGammaRow[id] = (T)(adjv * lv);
        }
      }
    }
    __syncthreads();
  }
}

void LayerNormalizationGrad(Ptr<Allocator> allocator,
                            Tensor gradX,
                            Tensor gradGamma,
                            Tensor gradBeta,
                            Tensor adj,
                            Tensor y,
                            Tensor x,
                            Tensor gamma,
                            Tensor beta,
                            float eps) {
  cudaSetDevice(adj->getDeviceId().no);
  int rows = y->shape().elements() / y->shape()[-1];
  int cols = y->shape()[-1];

  int threads = std::min(MAX_THREADS, cols);
  int blocks = std::min(MAX_BLOCKS, rows);

  auto tempGradGammaMemory = allocator->alloc(adj->memory()->size());
  Tensor tempGradGamma = TensorBase::New(tempGradGammaMemory, adj->shape(), adj->type(), adj->getBackend());
  tempGradGamma->set(0.f);

  auto tempOnesMemory = allocator->alloc(rows * sizeOf(adj->type()));
  Tensor tempOnes = TensorBase::New(tempOnesMemory, Shape({1, rows}), adj->type(), adj->getBackend());
  tempOnes->set(1.f);

  if(gradX->type() == Type::float32) {
    int shared = sizeof(float) * threads * 4;
    gLayerNormalizationGrad<float, float><<<blocks, threads, shared>>>(
      gradX->data<float>(),
      tempGradGamma->data<float>(),
      adj->data<float>(),
      y->data<float>(),
      x->data<float>(),
      gamma->data<float>(),
      (beta) ? beta->data<float>() : nullptr,
      rows,
      cols,
      eps);
#if COMPILE_FP16
  } else if (gradX->type() == Type::float16) {
    // accumulate in float
    int shared = sizeof(float) * threads * 4;
    gLayerNormalizationGrad<half, float><<<blocks, threads, shared>>>(
      gradX->data<half>(),
      tempGradGamma->data<half>(),
      adj->data<half>(),
      y->data<half>(),
      x->data<half>(),
      gamma->data<half>(),
      (beta) ? beta->data<half>() : nullptr,
      rows,
      cols,
      eps);
#endif
  } else {
    ABORT("LayerNormalizationGrad not implemented for type {}", gradX->type());
  }

  // We use this go get rid of the atomicAdd and perform a reduce of the gradients afterwards.
  // This is much faster for fp16 which seems to have a broken atomicAdd implementation.
  // We reduce bias gradients with a matrix multiply, but use a 32-bit compute type. 
  // This preserves precision with larger batches where all batch entries reduce into a single vector.
  // See also AffineNodeOp where we do the same for biases
  gpu::Prod(gradGamma, tempOnes, tempGradGamma, false, false, 1, 1, Type::float32); // beta set to one to add

  if(gradBeta) // dC/dbeta = adj - inverse broadcasting (reduction)
    gpu::Prod(gradBeta, tempOnes, adj, false, false, 1, 1, Type::float32); // beta set to one to add

  allocator->free(tempGradGammaMemory);
  allocator->free(tempOnesMemory);
}

template <typename T, typename AccType = float>
__global__ void gRMSNormalization(T* out,
                                  const T* in,
                                  const T* gamma,
                                  const T* beta,
                                  int rows,
                                  int cols,
                                  AccType eps = 1e-9) {
  extern __shared__ uint8_t _sharedBytes[];
  AccType* _shareAccType = (AccType*)_sharedBytes;

  AccType N = cols;
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* yRow       = out + j * cols;
      const T* xRow =  in + j * cols;

      AccType* _sqSum = _shareAccType;

      _sqSum[threadIdx.x] = (AccType)0.0f;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType xv = (AccType)xRow[id];
          _sqSum[threadIdx.x] += xv * xv;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      AccType rms = functional::Ops<AccType>::sqrt(_sqSum[0] / N + eps); // all AccType
      __syncthreads();

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType gammav  = (AccType)gamma[id];
          AccType xv      = (AccType)xRow[id];
          AccType betav   = beta ? (AccType)beta[id] : (AccType)0.f;
          AccType rmsNorm = xv / rms;
          AccType y       = gammav * rmsNorm + betav;
          yRow[id]        = (T)y;
        }
      }
    }
    __syncthreads();
  }
}

void RMSNormalization(Tensor out,
                      Tensor in,
                      Tensor gamma,
                      Tensor beta,
                      float eps) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();

  int blocks = std::min(MAX_BLOCKS, (int)rows);
  int threads = std::min(MAX_THREADS, (int)cols);
  int shared = threads * sizeof(float);

  if(out->type() == Type::float32) {
    gRMSNormalization<float, float><<<blocks, threads, shared>>>(out->data<float>(),
                                                                 in->data<float>(),
                                                                 gamma->data<float>(),
                                                                 beta ? beta->data<float>() : nullptr,
                                                                 rows,
                                                                 cols,
                                                                 eps);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gRMSNormalization<half, float><<<blocks, threads, shared>>>(out->data<half>(),
                                                                in->data<half>(),
                                                                gamma->data<half>(),
                                                                beta ? beta->data<half>() : nullptr,
                                                                rows,
                                                                cols,
                                                                eps);
#endif
  } else {
    ABORT("RMSNormalization not implemented for type {}", out->type());
  }
}

template <typename T, typename AccType = float>
__global__ void gRMSNormalizationGrad(T* gradX,
                                      T* gradGamma,
                                      T* adj,
                                      T* y,
                                      T* x,
                                      T* gamma,
                                      T* beta,
                                      int rows,
                                      int cols,
                                      AccType eps = 1e-9) {
  extern __shared__ uint8_t sharedBytes[];
  AccType* shared = (AccType*)sharedBytes;

  AccType N = cols;

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      AccType* sum_adj_r = shared;  // sum of gradient coming in times layerNorm from value
      AccType* sum_sqr   = shared + blockDim.x;  // sum of x^2

      const T* xRow   =   x + j * cols;
      const T* yRow   =   y + j * cols;
      const T* adjRow = adj + j * cols;

      sum_adj_r[threadIdx.x] = (AccType)0.0f;
      sum_sqr[threadIdx.x]   = (AccType)0.0f;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          AccType xv     = xRow[id];
          AccType yv     = yRow[id];
          AccType betav  = beta ? (AccType)beta[id] : (AccType)0.f;
          AccType gammav = (AccType)gamma[id];
          AccType adjv   = adjRow[id];
          AccType rv     = (yv - betav) / gammav; // go back to RMSNorm(x) from scaled and shifted version for accumulation

          sum_adj_r[threadIdx.x] += adjv * rv;
          sum_sqr[threadIdx.x]   += xv * xv;
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          sum_adj_r[threadIdx.x] += sum_adj_r[threadIdx.x + skip]; // Accumulates in AccType
          sum_sqr[threadIdx.x]   += sum_sqr[threadIdx.x   + skip]; // Accumulates in AccType
        }
        len = (len + 1) >> 1;
      }

      __syncthreads();
      AccType rms = functional::Ops<AccType>::sqrt(sum_sqr[0] / N + eps);
      __syncthreads();

      // Jacobian of RMS norm
      // J = [ \frac{1}{N * rms} (N\delta_{ij} - RN_i RN_j) ]_{ij}
      // J * a = dC/dx_i = ( N a_i - RN_i \sum_j RN_j a_j ) / (N * rms)

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {

          AccType xv      = xRow[id];
          AccType gammav  = (AccType)gamma[id];
          AccType adjv    = adjRow[id];
          AccType rmsNorm = xv / rms;

          AccType gradNorm = N * adjv - rmsNorm * sum_adj_r[0];
          gradNorm        /= N * rms; 

          AccType gradXv = gammav * gradNorm;

          // Keep RMSN gradient between [-1000, 1000] for TensorOps, this currently used for making values fit into fp16. This wil also clip inf. 
          // @TODO: to be fixed and removed.
          AccType sign = functional::Ops<AccType>::sgn(gradXv);
          AccType cutoff = (AccType)1000.f; // @TODO: expose this somehow as an option? or better: make obsolete.
          gradXv = functional::Ops<AccType>::abs(gradXv) > cutoff ? sign * cutoff : gradXv; // if gradXv is NaN the value return is NaN too because NaN > value is false.

          // @TODO: frankly, this is embarrasing and should rather be removed or optional? It does help for low precision computation though. Maybe turn into option?
          gradXv = isnan(gradXv) ? 0.f : gradXv; // turn NaN into 0.

          T* gradXRow      = gradX     + j * cols;
          gradXRow[id]    += (T)(gradXv);

          T* gradGammaRow  = gradGamma + j * cols;
          // assignment is correct here as this gets summed up
          // in the next kernel via matrix product
          gradGammaRow[id] = (T)(adjv * rmsNorm);
        }
      }
    }
    __syncthreads();
  }
}

void RMSNormalizationGrad(Ptr<Allocator> allocator,
                          Tensor gradX,
                          Tensor gradGamma,
                          Tensor gradBeta,
                          Tensor adj,
                          Tensor y,
                          Tensor x,
                          Tensor gamma,
                          Tensor beta,
                          float eps) {
  cudaSetDevice(adj->getDeviceId().no);
  int rows = y->shape().elements() / y->shape()[-1];
  int cols = y->shape()[-1];

  int threads = std::min(MAX_THREADS, cols);
  int blocks = std::min(MAX_BLOCKS, rows);

  auto tempGradGammaMemory = allocator->alloc(adj->memory()->size());
  Tensor tempGradGamma = TensorBase::New(tempGradGammaMemory, adj->shape(), adj->type(), adj->getBackend());
  tempGradGamma->set(0.f);

  auto tempOnesMemory = allocator->alloc(rows * sizeOf(adj->type()));
  Tensor tempOnes = TensorBase::New(tempOnesMemory, Shape({1, rows}), adj->type(), adj->getBackend());
  tempOnes->set(1.f);

  if(gradX->type() == Type::float32) {
    int shared = sizeof(float) * threads * 2;
    gRMSNormalizationGrad<float, float><<<blocks, threads, shared>>>(
      gradX->data<float>(),
      tempGradGamma->data<float>(),
      adj->data<float>(),
      y->data<float>(),
      x->data<float>(),
      gamma->data<float>(),
      (beta) ? beta->data<float>() : nullptr,
      rows,
      cols,
      eps);
#if COMPILE_FP16
  } else if (gradX->type() == Type::float16) {
    // accumulate in float
    int shared = sizeof(float) * threads * 2;
    gRMSNormalizationGrad<half, float><<<blocks, threads, shared>>>(
      gradX->data<half>(),
      tempGradGamma->data<half>(),
      adj->data<half>(),
      y->data<half>(),
      x->data<half>(),
      gamma->data<half>(),
      (beta) ? beta->data<half>() : nullptr,
      rows,
      cols,
      eps);
#endif
  } else {
    ABORT("RMSNormalizationGrad not implemented for type {}", gradX->type());
  }

  // We use this go get rid of the atomicAdd and perform a reduce of the gradients afterwards.
  // This is much faster for fp16 which seems to have a broken atomicAdd implementation.
  // We reduce bias gradients with a matrix multiply, but use a 32-bit compute type. 
  // This preserves precision with larger batches where all batch entries reduce into a single vector.
  // See also AffineNodeOp where we do the same for biases
  gpu::Prod(gradGamma, tempOnes, tempGradGamma, false, false, 1, 1, Type::float32); // beta set to one to add

  if(gradBeta) // dC/dbeta = adj - inverse broadcasting (reduction)
    gpu::Prod(gradBeta, tempOnes, adj, false, false, 1, 1, Type::float32); // beta set to one to add

  allocator->free(tempGradGammaMemory);
  allocator->free(tempOnesMemory);
}


template <bool add, typename T>
__global__ void gShift(T* out,
                       const T* in,
                       int length,
                       int offset,
                       float padValue) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      if(add) {
        if(index - offset >= 0 && index - offset < length)
          out[index] += in[index - offset];
      } else {
        if(index - offset < 0 || index - offset >= length)
          out[index] = (T)padValue;
        else
          out[index] = in[index - offset];
      }
    }
  }
}

void Shift(Tensor out,
           Tensor in,
           marian::Shape shift,
           float padValue,
           bool invert) {
  ABORT_IF(in->shape().size() != shift.size(), "bad dimensions");

  // BUGBUG: This can only shift along the first axis. Shifting, e.g., along the
  // last axis cannot be implemented this way.
  int offset = 0;
  for(int i = 0; i < shift.size(); ++i)
    offset += in->shape().stride(i) * shift[i];

  if(invert)
    offset = -offset;

  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  if(out->type() == Type::float32) {
    gShift<false>
        <<<blocks, threads>>>(out->data<float>(), in->data<float>(), length, offset, padValue);
#if COMPILE_FP16
  } else if(out->type() == Type::float16) {
    gShift<false>
        <<<blocks, threads>>>(out->data<half>(), in->data<half>(), length, offset, padValue);
#endif
  } else {
    ABORT("Shift not implemented for type {}", out->type());
  }
}

void ShiftGrad(Tensor out, Tensor in, marian::Shape shift, bool invert) {
  ABORT_IF(in->shape().size() != shift.size(), "bad dimensions");

  // BUGBUG: This can only shift along the first axis. Shifting, e.g., along the
  // last axis cannot be implemented this way.
  int offset = 0;
  for(int i = 0; i < shift.size(); ++i)
    offset += in->shape().stride(i) * shift[i];

  if(invert)
    offset = -offset;

  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  if(out->type() == Type::float32) {
    gShift<true>
        <<<blocks, threads>>>(out->data<float>(), in->data<float>(), length, offset, 0.f); // @TODO: What about padValue?
#if COMPILE_FP16
  } else if(out->type() == Type::float16) {
    gShift<true>
        <<<blocks, threads>>>(out->data<half>(), in->data<half>(), length, offset, 0.f);
#endif
  } else {
    ABORT("Shift not implemented for type {}", out->type());
  }
}

__global__ void gSetSparse(float* out,
                           const size_t* indices,
                           const float* values,
                           int length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      out[indices[index]] = values[index];
    }
  }
}

void SetSparse(float* out,
               const std::vector<size_t>& indices,
               const std::vector<float>& values) {
  int length = indices.size();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  size_t* d_indices;
  CUDA_CHECK(cudaMalloc(&d_indices, length * sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(d_indices,
                        indices.data(),
                        length * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  float* d_values;
  CUDA_CHECK(cudaMalloc(&d_values, length * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(
      d_values, values.data(), length * sizeof(float), cudaMemcpyHostToDevice));

  gSetSparse<<<blocks, threads>>>(out, d_indices, d_values, length);

  cudaFree(d_indices);
  cudaFree(d_values);
}

/******************************************************************************/

template <typename T>
__global__ void gLSTMCellForward(T* out,
                                 const T* cell,
                                 const T* xW,
                                 const T* sU,
                                 const T* b,
                                 const T* mask,
                                 size_t rows,
                                 size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T m = !mask || mask[j];

      T* rowOut = out + j * cols;
      const T* rowCell = cell + j * cols;

      const T* xWrow = xW + j * cols * 4;
      const T* sUrow = sU + j * cols * 4;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          T gf = functional::Ops<T>::sigmoid(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;
          T gi = functional::Ops<T>::sigmoid(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          T gc = functional::Ops<T>::tanh(xWrow[l] + sUrow[l] + b[l]);

          T cout = gf * rowCell[i] + gi * gc;
          rowOut[i] = m * cout + ((T)1.f - m) * rowCell[i];
        }
      }
    }
  }
}

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  if(out->type() == Type::float32) {
   gLSTMCellForward<<<blocks, threads>>>(
      out->data<float>(),                                // output
      inputs[0]->data<float>(),                          // cell state
      inputs[1]->data<float>(),                          // xW
      inputs[2]->data<float>(),                          // sU
      inputs[3]->data<float>(),                          // b
      inputs.size() > 4 ? inputs[4]->data<float>() : 0,  // mask
      rows,
      cols);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gLSTMCellForward<<<blocks, threads>>>(
      out->data<half>(),                                // output
      inputs[0]->data<half>(),                          // cell state
      inputs[1]->data<half>(),                          // xW
      inputs[2]->data<half>(),                          // sU
      inputs[3]->data<half>(),                          // b
      inputs.size() > 4 ? inputs[4]->data<half>() : 0,  // mask
      rows,
      cols);
#endif
  } else {
    ABORT("LSTMCellForward not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gLSTMOutputForward(T* out,
                                   const T* cell,
                                   const T* xW,
                                   const T* sU,
                                   const T* b,
                                   size_t rows,
                                   size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols;
      const T* rowCell = cell + j * cols;

      const T* xWrow = xW + j * cols * 4;
      const T* sUrow = sU + j * cols * 4;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + 3 * cols;
          T go = functional::Ops<T>::sigmoid(xWrow[k] + sUrow[k] + b[k]);
          rowOut[i] = go * functional::Ops<T>::tanh(rowCell[i]);
        }
      }
    }
  }
}

void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs) {
  cudaSetDevice(out->getDeviceId().no);

  int rows = out->shape().elements() / out->shape().back();
  int cols = out->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  if(out->type() == Type::float32) {
    gLSTMOutputForward<<<blocks, threads>>>(out->data<float>(),        // output
                                            inputs[0]->data<float>(),  // cell state
                                            inputs[1]->data<float>(),  // xW
                                            inputs[2]->data<float>(),  // sU
                                            inputs[3]->data<float>(),  // b
                                            rows,
                                            cols);
#if COMPILE_FP16
  } else if (out->type() == Type::float16) {
    gLSTMOutputForward<<<blocks, threads>>>(out->data<half>(),        // output
                                            inputs[0]->data<half>(),  // cell state
                                            inputs[1]->data<half>(),  // xW
                                            inputs[2]->data<half>(),  // sU
                                            inputs[3]->data<half>(),  // b
                                            rows,
                                            cols);
#endif
  } else {
    ABORT("gLSTMOutputForward not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gLSTMCellBackward(T* outCell,
                                  T* outXW,
                                  T* outSU,
                                  T* outB,
                                  const T* cell,
                                  const T* xW,
                                  const T* sU,
                                  const T* b,
                                  const T* mask,
                                  const T* adj,
                                  size_t rows,
                                  size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T m = !mask || mask[j];

      T* rowOutCell = outCell + j * cols;
      T* rowOutXW = outXW + j * cols * 4;
      T* rowOutSU = outSU + j * cols * 4;

      const T* rowCell = cell + j * cols;
      const T* xWrow = xW + j * cols * 4;
      const T* sUrow = sU + j * cols * 4;

      const T* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          T gf = functional::Ops<T>::sigmoid(xWrow[i] + sUrow[i] + b[i]);

          int k = i + cols;
          T gi = functional::Ops<T>::sigmoid(xWrow[k] + sUrow[k] + b[k]);

          int l = i + 2 * cols;
          T gc = functional::Ops<T>::tanh(xWrow[l] + sUrow[l] + b[l]);

          T adj = rowAdj[i];

          // dc/dc_{t-1}
          if(outCell)
            rowOutCell[i] += (m * gf - m + (T)1.f) * adj;

          // dc/d(b_f) = dc/d(xW_f) ...
          T dcdxf = m * rowCell[i] * gf * ((T)1.f - gf) * adj;
          if(outXW)
            rowOutXW[i] += dcdxf;
          if(outSU)
            rowOutSU[i] += dcdxf;
          if(outB)
            atomics::atomicAdd(outB + i, dcdxf); // @TODO: get rid of atomicAdd everywhere!

          // dc/d(b_i) ...
          T dcdb_i = m * gc * gi * ((T)1.f - gi) * adj;
          if(outXW)
            rowOutXW[k] += dcdb_i;
          if(outSU)
            rowOutSU[k] += dcdb_i;
          if(outB)
            atomics::atomicAdd(outB + k, dcdb_i);

          // dc/d(b_c) ...
          T dcdxc = m * gi * ((T)1.f - gc * gc) * adj;
          if(outXW)
            rowOutXW[l] += dcdxc;
          if(outSU)
            rowOutSU[l] += dcdxc;
          if(outB)
            atomics::atomicAdd(outB + l, dcdxc);
        }
      }
    }
  }
}

void LSTMCellBackward(std::vector<Tensor> outputs,
                      std::vector<Tensor> inputs,
                      Tensor adj) {
  cudaSetDevice(adj->getDeviceId().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  if(adj->type() == Type::float32) {
    gLSTMCellBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data<float>() : 0,        // state - adj
      outputs[1] ? outputs[1]->data<float>() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data<float>() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data<float>() : 0,        // b - adj
      inputs[0]->data<float>(),                          // state
      inputs[1]->data<float>(),                          // xW
      inputs[2]->data<float>(),                          // sU
      inputs[3]->data<float>(),                          // b
      inputs.size() > 4 ? inputs[4]->data<float>() : 0,  // mask
      adj->data<float>(),
      rows,
      cols);
#if COMPILE_FP16
  } else if (adj->type() == Type::float16) {
    gLSTMCellBackward<<<blocks, threads>>>(
      outputs[0] ? outputs[0]->data<half>() : 0,        // state - adj
      outputs[1] ? outputs[1]->data<half>() : 0,        // xW - adj
      outputs[2] ? outputs[2]->data<half>() : 0,        // sU - adj
      outputs[3] ? outputs[3]->data<half>() : 0,        // b - adj
      inputs[0]->data<half>(),                          // state
      inputs[1]->data<half>(),                          // xW
      inputs[2]->data<half>(),                          // sU
      inputs[3]->data<half>(),                          // b
      inputs.size() > 4 ? inputs[4]->data<half>() : 0,  // mask
      adj->data<half>(),
      rows,
      cols);
#endif
  } else {
    ABORT("gLSTMCellBackward not implemented for type {}", adj->type());
  }

}

template <typename T>
__global__ void gLSTMOutputBackward(T* outCell,
                                    T* outXW,
                                    T* outSU,
                                    T* outB,
                                    const T* cell,
                                    const T* xW,
                                    const T* sU,
                                    const T* b,
                                    const T* adj,
                                    size_t rows,
                                    size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOutCell = outCell + j * cols;
      T* rowOutXW = outXW + j * cols * 4;
      T* rowOutSU = outSU + j * cols * 4;

      const T* rowCell = cell + j * cols;
      const T* xWrow = xW + j * cols * 4;
      const T* sUrow = sU + j * cols * 4;

      const T* rowAdj = adj + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          int k = i + 3 * cols;
          T go = functional::Ops<T>::sigmoid(xWrow[k] + sUrow[k] + b[k]);

          T t = functional::Ops<T>::tanh(rowCell[i]);

          T adj = rowAdj[i];

          // dc/dc_{t-1}
          if(outCell)
            rowOutCell[i] += go * ((T)1.f - t * t) * adj;

          // dc/d(b_o) = dc/d(xW_f) ...
          float dcdxo = t * go * ((T)1.f - go) * adj;
          if(outXW)
            rowOutXW[k] += dcdxo;
          if(outSU)
            rowOutSU[k] += dcdxo;
          if(outB)
            atomics::atomicAdd(outB + k, dcdxo); // @TODO: get rid of atomicAdd
        }
      }
    }
  }
}

void LSTMOutputBackward(std::vector<Tensor> outputs,
                        std::vector<Tensor> inputs,
                        Tensor adj) {
  cudaSetDevice(adj->getDeviceId().no);

  int rows = adj->shape().elements() / adj->shape().back();
  int cols = adj->shape().back();

  int blocks = std::min(MAX_BLOCKS, rows);
  int threads = std::min(MAX_THREADS, cols);

  if(adj->type() == Type::float32) {
    gLSTMOutputBackward<<<blocks, threads>>>(
        outputs[0] ? outputs[0]->data<float>() : 0,  // state - adj
        outputs[1] ? outputs[1]->data<float>() : 0,  // xW - adj
        outputs[2] ? outputs[2]->data<float>() : 0,  // sU - adj
        outputs[3] ? outputs[3]->data<float>() : 0,  // b - adj
        inputs[0]->data<float>(),                    // state
        inputs[1]->data<float>(),                    // xW
        inputs[2]->data<float>(),                    // sU
        inputs[3]->data<float>(),                    // b
        adj->data<float>(),
        rows,
        cols);
#if COMPILE_FP16
  } else if (adj->type() == Type::float16) {
    gLSTMOutputBackward<<<blocks, threads>>>(
        outputs[0] ? outputs[0]->data<half>() : 0,  // state - adj
        outputs[1] ? outputs[1]->data<half>() : 0,  // xW - adj
        outputs[2] ? outputs[2]->data<half>() : 0,  // sU - adj
        outputs[3] ? outputs[3]->data<half>() : 0,  // b - adj
        inputs[0]->data<half>(),                    // state
        inputs[1]->data<half>(),                    // xW
        inputs[2]->data<half>(),                    // sU
        inputs[3]->data<half>(),                    // b
        adj->data<half>(),
        rows,
        cols);
#endif
  } else {
    ABORT("gLSTMOutputBackward not implemented for type {}", adj->type());
  }
}

template <typename T>
__global__ void gHighwayForward(T* out,
                                const T* in1,
                                const T* in2,
                                const T* t,
                                size_t length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      T sigma = functional::Ops<T>::sigmoid(t[index]);
      out[index] = in1[index] * sigma + in2[index] * ((T)1.f - sigma);
    }
  }
}

void HighwayForward(Tensor out,
                    const Tensor in1,
                    const Tensor in2,
                    const Tensor t) {
  cudaSetDevice(out->getDeviceId().no);

  int length = out->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  if(out->type() == Type::float32) {
    gHighwayForward<<<blocks, threads>>>(
        out->data<float>(), in1->data<float>(), in2->data<float>(), t->data<float>(), length);
#if COMPILE_FP16
  } else if(out->type() == Type::float16) {
    gHighwayForward<<<blocks, threads>>>(
        out->data<half>(), in1->data<half>(), in2->data<half>(), t->data<half>(), length);
#endif
  } else {
    ABORT("HighwayForward not implemented for type {}", out->type());
  }
}

template <typename T>
__global__ void gHighwayBackward(T* out1,
                                 T* out2,
                                 T* outt,
                                 const T* in1,
                                 const T* in2,
                                 const T* t,
                                 const T* adj,
                                 size_t length) {
  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      T sigma = functional::Ops<T>::sigmoid(t[index]);
      out1[index] = sigma * adj[index];
      out2[index] = ((T)1.f - sigma) * adj[index];
      outt[index]
          = sigma * ((T)1.f - sigma) * (in1[index] - in2[index]) * adj[index];
    }
  }
}

void HighwayBackward(Tensor out1,
                     Tensor out2,
                     Tensor outt,
                     const Tensor in1,
                     const Tensor in2,
                     const Tensor t,
                     const Tensor adj) {
  cudaSetDevice(out1->getDeviceId().no);

  int length = out1->shape().elements();

  int threads = std::min(MAX_THREADS, length);
  int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

  if(out1->type() == Type::float32) {
    gHighwayBackward<<<blocks, threads>>>(out1->data<float>(),
                                          out2->data<float>(),
                                          outt->data<float>(),
                                          in1->data<float>(),
                                          in2->data<float>(),
                                          t->data<float>(),
                                          adj->data<float>(),
                                          length);
#if COMPILE_FP16
  } else if(out1->type() == Type::float16) {
    gHighwayBackward<<<blocks, threads>>>(out1->data<half>(),
                                          out2->data<half>(),
                                          outt->data<half>(),
                                          in1->data<half>(),
                                          in2->data<half>(),
                                          t->data<half>(),
                                          adj->data<half>(),
                                          length);
#endif
  } else {
    ABORT("HighwayForward not implemented for type {}", out1->type());
  }
}

__global__ void gMaxPoolingForward(float* out,
                                   int outRows,
                                   int outCols,
                                   float* in,
                                   int inRows,
                                   int inCols,
                                   float* mask,
                                   int numKernels,
                                   int maskCols,
                                   int width,
                                   int lastWidth) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid >= outRows * outCols)
    return;

  int rowId = tid / outRows;
  int colId = tid % outRows;

  float* b = in + (rowId * inCols) + (colId * width);
  float* localMask = mask + (rowId / numKernels) * maskCols + colId * width;

  if(colId == outRows - 1) {
    width = lastWidth;
  }

  float currentMax = b[0] * localMask[0];
  for(int i = 1; i < width; ++i) {
    if(b[i] * localMask[i] > currentMax) {
      currentMax = b[i] * localMask[i];
    }
  }

  out[rowId + (colId * outCols)] = currentMax;
}

void PoolingWithMaskingForward(Tensor out,
                               Tensor in,
                               Tensor mask,
                               int width,
                               bool isEven) {
  matchOrAbort<float>(out->type());
  int n = out->shape().elements();
  int threads = std::min(n, MAX_THREADS);
  int blocks = n / threads + (n % threads != 0);

  auto& inShape = in->shape();
  int inRows = inShape[0] * inShape[1];
  int inCols = inShape[2];

  auto& outShape = out->shape();
  int outRows = outShape[2];
  int outCols = outShape[0] * outShape[1];

  int lastWidth
      = ((inCols - isEven) % width == 0) ? width : (inCols - isEven) % width;

  gMaxPoolingForward<<<blocks, threads>>>(out->data(),
                                          outRows,
                                          outCols,
                                          in->data(),
                                          inRows,
                                          inCols,
                                          mask->data(),
                                          outShape[1],
                                          mask->shape()[2],
                                          width,
                                          lastWidth);
}

__global__ void gMaxPoolingBackward(float* adj,
                                    int adjRows,
                                    int adjCols,
                                    float* in,
                                    float* adjIn,
                                    int inRows,
                                    int inCols,
                                    float* mask,
                                    int numKernels,
                                    int maskCols,
                                    int width,
                                    int lastWidth) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid >= adjRows * adjCols)
    return;

  int rowId = tid / adjRows;
  int colId = tid % adjRows;

  float* b = in + (rowId * inCols) + (colId * width);

  if(colId == adjRows - 1) {
    width = lastWidth;
  }

  float* localMask = mask + (rowId / numKernels) * maskCols + colId * width;
  size_t currentMaxIdx = 0;
  for(int i = 1; i < width; ++i) {
    if(b[i] * localMask[i] > b[currentMaxIdx] * localMask[currentMaxIdx]) {
      currentMaxIdx = i;
    }
  }

  adjIn[(rowId * inCols) + (colId * width) + currentMaxIdx]
      += adj[rowId + (colId * adjCols)];
}

void PoolingWithMaskingBackward(Tensor adj,
                                Tensor adjIn,
                                Tensor in,
                                Tensor mask,
                                int width,
                                bool isEven) {
  matchOrAbort<float>(adj->type());
  int n = adj->shape().elements();
  int threads = std::min(n, 512);
  int blocks = n / threads + (n % threads != 0);

  auto& inShape = in->shape();
  int inRows = inShape[0] * inShape[1];
  int inCols = inShape[2];

  auto& adjShape = adj->shape();
  int adjRows = adjShape[2];
  int adjCols = adjShape[0] * adjShape[1];

  int lastWidth
      = ((inCols - isEven) % width == 0) ? width : (inCols - isEven) % width;

  gMaxPoolingBackward<<<blocks, threads>>>(adj->data(),
                                           adjRows,
                                           adjCols,
                                           in->data(),
                                           adjIn->data(),
                                           inRows,
                                           inCols,
                                           mask->data(),
                                           adjShape[1],
                                           mask->shape()[2],
                                           width,
                                           lastWidth);
}
}  // namespace gpu
}  // namespace marian
