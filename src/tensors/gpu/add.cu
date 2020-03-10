#include "tensors/gpu/add.h"
#include "tensors/gpu/add_all.h"

#include "tensors/gpu/cuda_helpers.h"

#include "functional/functional.h"
#include "functional/shape.h"
#include "functional/tensor.h"
#include "functional/tmp.h"

namespace marian {

namespace gpu {

template <size_t K, class Functor, class AggFunctor, typename T, typename AccType>
__global__ void gAggregateGeneric(Functor functor,                                 // functor applied to single corresponding elements in tensors (via broadcasting),
                                  AccType aggInit,                                 // aggInit is starting value of accumulation (e.g. 0 for sum),
                                  AggFunctor aggFunctor,                           // aggFunctor is used to accumulate values (e.g. sum),
                                  const functional::Shape full,                    // maximal combined shape of all tensors via broadcasting
                                  functional::Tensor<T> out,                       // output tensor
                                  functional::Array<functional::Tensor<T>, K> ins, // input tensors
                                  AccType scale = 1.0) {                           // scale accumulation result by scale. e.g. used for computing mean from sum over N elements with scale 1/N
  int outLength = out.shape().elements();
  bool same = outLength == full.elements();
  for(int i = 0; i < K; ++i)
    same = same && outLength == ins[i].shape().elements();

  constexpr size_t N = functional::Shape::size();
  functional::Array<int, N> len;
  for(int i = 0; i < N; ++i)
    len[i] = full[i] / out.shape()[i];

  functional::Array<int, N> dims;
  for(int bid = 0; bid < outLength; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < outLength) {
      if(same) {
        out[index] = (T)aggFunctor((AccType)out[index], functional::applyWithCast<AccType>(functor, ins, index) * scale); // apply functors to with arguments cast to AccType
      } else {
        out.shape().dims(index, dims);
        out[index] = (T)aggFunctor((AccType)out[index], functional::loops(functor, aggInit, aggFunctor, ins, len, dims) * scale); // apply functors to with arguments cast to AccType
      }
    }
  }
}

template <size_t K, class Functor, class AggFunctor, typename T, typename AccType>
__global__ void gAggregateEqual(Functor functor, AggFunctor aggFunctor,
                                functional::Tensor<T> out,
                                functional::Array<functional::Tensor<T>, K> ins,
                                AccType scale,
                                bool broadcast) {
  int length = out.shape().elements();
  functional::Array<int, functional::Shape::size()> dims;

  for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
    int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
    if(index < length) {
      functional::Array<int, K> indices;
      indices.fill(index);

      if(broadcast) {
        out.shape().dims(index, dims);
        for(size_t i = 0; i < K; ++i)
          indices[i] = ins[i].shape().bindex(dims);
      }

      out[index] = (T)aggFunctor((AccType)out[index], functional::applyWithCast<AccType>(functor, ins, indices) * scale);
    }
  }
}

template <size_t K, class Functor, class AggFunctor, typename T, typename AccType = float>
__global__ void gAggregateReduce(Functor functor, AccType aggInit, AggFunctor aggFunctor,
                                 const functional::Shape full,
                                 functional::Tensor<T> out,
                                 functional::Array<functional::Tensor<T>, K> ins,
                                 AccType scale = 1.0) {
  int rows = full.elements() / full.back();
  int cols = full.back();

  bool same = true; // do all inputs have the same number of elements?
  for(int i = 0; i < K; ++i)
    same = same && ins[i].shape().elements() == full.elements();

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      // make sure shared memory is the same for different types
      // by using bytes instead of type T
      extern __shared__ uint8_t _sharedBytes[];
      AccType* _sum = (AccType*)_sharedBytes;

      if(same) {
        _sum[threadIdx.x] = aggInit;
        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols)
            _sum[threadIdx.x] = aggFunctor(_sum[threadIdx.x], functional::applyWithCast<AccType>(functor, ins, j * cols + id)); // casts to AccType before applying functor which then performs operation in AccType
        }
      } else {
        functional::Array<int, functional::Shape::size()> dims;
        _sum[threadIdx.x] = aggInit;

        for(int tid = 0; tid < cols; tid += blockDim.x) {
          int id = tid + threadIdx.x;
          if(id < cols) {
            full.dims(j * cols + id, dims);
            functional::Array<int, K> indices;
            for(int i = 0; i < K; ++i)
              indices[i] = ins[i].shape().bindex(dims);
            _sum[threadIdx.x] = aggFunctor(_sum[threadIdx.x], functional::applyWithCast<AccType>(functor, ins, indices));// casts to AccType before applying functor which then performs operation in AccType
          }
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1)) {
          _sum[threadIdx.x] = aggFunctor(_sum[threadIdx.x], _sum[threadIdx.x + skip]);
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      if(threadIdx.x == 0) // only set value when in thread 0 in block
        out[j] = aggFunctor(out[j], (T)(_sum[0] * scale));
    }
    __syncthreads();
  }
}

template <typename T, typename AccType, class Functor, class AggFunctor, class... Tensors>
void AggregateTyped(Functor functor, AccType aggInit, AggFunctor aggFunctor, AccType scale, marian::Tensor out, Tensors... tensors) {
  cudaSetDevice(out->getDeviceId().no);

  auto full = marian::Shape::broadcast({out, tensors...});

  int length = out->shape().elements();

  constexpr size_t K = sizeof...(Tensors);

  functional::Tensor<T> gOut = out;
  functional::Array<functional::Tensor<T>, K> gIns = {tensors...};

  if(out->shape().elements() == 1) { // reduce everything into a single element
    AggregateAll<T, AccType>(nullptr, functor, aggInit, aggFunctor, scale, out, tensors...); // @TODO: pass allocator in here, currently uses cudaMalloc
  } else if(full.back() != 1 && out->shape().back() == 1 && full.elements() / full.back() == length) { // element number of out and full shape on axis that are not reduced must match
    size_t m = full.elements() / full.back(); // how many rows are we iterating over?
    size_t k = full.back();                   // how many columns are being reduced to 1 in each row?

    int blocks  = std::min(MAX_BLOCKS,  (int)m);
    int threads = std::min(MAX_THREADS, (int)k);
    int shared  = sizeof(AccType) * threads;
    gAggregateReduce<K, Functor, AggFunctor, T, AccType><<<blocks, threads, shared>>>(functor, aggInit, aggFunctor, full, gOut, gIns, scale);
  } else if(out->shape() == full) {
    int threads = std::min(MAX_THREADS, length);
    int blocks  = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    bool broadcast = false;
    for(int i = 0; i < K; ++i)
      broadcast = broadcast || gOut.shape() != gIns[i].shape();
    gAggregateEqual<<<blocks, threads>>>(functor, aggFunctor, gOut, gIns, scale, broadcast);
  } else {
    int threads = std::min(MAX_THREADS, length);
    int blocks = std::min(MAX_BLOCKS, length / threads + (length % threads != 0));

    gAggregateGeneric<<<blocks, threads>>>(functor, aggInit, aggFunctor, full, gOut, gIns, scale);
  }
}

template <class Functor, class AggFunctor, class... Tensors>
void Aggregate(Functor functor, float aggInit, AggFunctor aggFunctor, float scale, marian::Tensor out, Tensors... tensors) {
  if(out->type() == Type::float32) {
    AggregateTyped<float, float>(functor, aggInit, aggFunctor, scale, out, tensors...);
  } else if(out->type() == Type::float16) {
#if COMPILE_FP16
    AggregateTyped<half,  float>(functor, aggInit, aggFunctor, scale, out, tensors...);
#else
     ABORT("FP16 not supported with current hardware or CUDA version");
#endif
  } else {
    ABORT("Type {} not yet supported", out->type());
  }
}

template <class Functor, class... Tensors>
void Add(Functor functor, float scale, marian::Tensor out, Tensors... tensors) {
  auto addFunctor = functional::_1 + functional::_2;
  Aggregate(functor, 0.f, addFunctor, scale, out, tensors...);
}

#include "tensors/gpu/add.inc"
}  // namespace gpu
}  // namespace marian
