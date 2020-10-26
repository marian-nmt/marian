/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECINDIRECFunctor, T, AccTyf, pe, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACSf, TRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "functional/tmp.h"
#include <cooperative_groups.h>

namespace marian {

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};


/*
    This version adds multiple elements per thread sequentially.  This reduces
   the overall cost of the algorithm while keeping the work complexity O(n) and
   the step complexity O(log n). (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <typename T, typename AccType, unsigned int blockSize, bool nIsPow2Greater1, size_t K, class Functor, class AggFunctor>
__global__ void reduceSinglePass(Functor functor, AccType aggInit, AggFunctor aggFunctor, AccType scale,
                                 const functional::Shape full,
                                 functional::Tensor<AccType> out,
                                 functional::Array<functional::Tensor<T>, K> ins) {
  int n = full.elements();

  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  AccType *sdata = SharedMemory<AccType>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  AccType mySum = aggInit;

  // we reduceSinglePass multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum = aggFunctor(mySum, functional::applyWithCast<AccType>(functor, ins, i));

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2Greater1 || i + blockSize < n) 
      mySum = aggFunctor(mySum, functional::applyWithCast<AccType>(functor, ins, i + blockSize));

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = aggFunctor(mySum, sdata[tid + 256]);
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = aggFunctor(mySum, sdata[tid + 128]);
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = aggFunctor(mySum, sdata[tid + 64]);
  }

  cg::sync(cta);

  // leverage that blockSize is always pow of 2 so no special logic needed in reduction loop.
  constexpr int partitionSize = blockSize > 32 ? 32 : blockSize;
  cg::thread_block_tile<partitionSize> tile = cg::tiled_partition<partitionSize>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) 
      mySum = aggFunctor(mySum, sdata[tid + 32]);
    // reduce final warp using shuffle
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
      mySum = aggFunctor(mySum, tile.shfl_down(mySum, offset));
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) 
    out[blockIdx.x] = aggFunctor(out[blockIdx.x], mySum * scale); // aggFunctor?
}

static inline bool isPow2Greater1(unsigned int x) { // is power of two but also larger than 1, otherwise an out-of-bounds read occurs
  return x > 1 && ((x & (x - 1)) == 0);
}

static inline unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <typename T, typename AccType, size_t K, class Functor, class AggFunctor>
void reduceSinglePass(Functor functor, AccType aggInit, AggFunctor aggFunctor, AccType scale,
                      const functional::Shape full,
                      functional::Tensor<AccType> out,
                      functional::Array<functional::Tensor<T>, K> ins,
                      int threads, int blocks) {
  int size = full.elements();
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(AccType) : threads * sizeof(AccType);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if (isPow2Greater1(size)) {
    switch (threads) {
      case 512:
        reduceSinglePass<T, AccType, 512, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 256:
        reduceSinglePass<T, AccType, 256, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 128:
        reduceSinglePass<T, AccType, 128, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 64:
        reduceSinglePass<T, AccType, 64, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 32:
        reduceSinglePass<T, AccType, 32, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 16:
        reduceSinglePass<T, AccType, 16, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 8:
        reduceSinglePass<T, AccType, 8, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 4:
        reduceSinglePass<T, AccType, 4, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 2:
        reduceSinglePass<T, AccType, 2, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 1:
        reduceSinglePass<T, AccType, 1, true><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
    }
  } else {
    switch (threads) {
      case 512:
        reduceSinglePass<T, AccType, 512, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 256:
        reduceSinglePass<T, AccType, 256, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 128:
        reduceSinglePass<T, AccType, 128, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 64:
        reduceSinglePass<T, AccType, 64, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 32:
        reduceSinglePass<T, AccType, 32, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 16:
        reduceSinglePass<T, AccType, 16, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 8:
        reduceSinglePass<T, AccType, 8, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 4:
        reduceSinglePass<T, AccType, 4, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 2:
        reduceSinglePass<T, AccType, 2, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
      case 1:
        reduceSinglePass<T, AccType, 1, false><<<dimGrid, dimBlock, smemSize>>>(functor, aggInit, aggFunctor, scale, full, out, ins);
        break;
    }
  }
}

}
