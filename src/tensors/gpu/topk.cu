#include "tensors/tensor_operators.h"
#include "tensors/gpu/cuda_helpers.h"
#include "tensors/allocator.h"

#include <cuda.h>

// GPU implementation of proper Marian top-k operator for TopkNodeOp
// This file contains a lot of code-duplicaton with src/translator/nth_element.cu
// the goal is to replace the beam-search specific topk search with this code. 
// Currently this is only used in the unit tests, but we will move forward and 
// make the beam-search more graph and operator-based.

namespace marian {
namespace gpu {  

const int MAX_BINS   = 500;
const int BLOCK_SIZE = 512;

#define UNROLL_MAXARG_LOOP(n, max)                    \
  if(tid < (n) && tid + (n) < (max)) {                \
    if(sharedValues[tid + (n)] > sharedValues[tid]) { \
      sharedIndices[tid] = sharedIndices[tid + (n)];  \
      sharedValues[tid]  = sharedValues[tid + (n)];   \
    }                                                 \
  }

// finds maximum element (first step) 
template <typename T>
__global__ void gMaxElement(IndexType* binIndices, // out: top-k positions
                            T* binValues,          // out: top-k scores
                            const T* inValues,     // this is the probs array, only one with type float or half
                            int rows,              // we iterate over this many rows, row-major layout
                            int cols,              // a row has that many columns, row-major layout
                            float minimal,         // minimal is the smallest possible value. For simplicity we assume we look for the maxmimum.
                            bool descending)       // This will be the largest possible value if the order is reversed (i.e. we look for the minimum).
{
  extern __shared__ float sharedValues[];
  __shared__ IndexType sharedIndices[BLOCK_SIZE];

  // id of current thread within block
  int tid = threadIdx.x;

  float flip = descending ? 1.f : -1.f;

  // Roll over every row in row-major 2D representation of the data
  for(int rowIdx = 0; rowIdx < rows; ++rowIdx) {
    int begin = rowIdx * cols;        // start index of a row
    int end   = rowIdx * cols + cols; // end index of a row

    // We look at at most blockDim.x * 2 = 1024 values within a block, i.e. each thread reduces two values.
    // Here we set the position to begin + blockId * 1024 + threadId. If a row has more values we 
    // partition the row according to blocks of 1024 values.
    int i = begin + blockIdx.x * (blockDim.x * 2) + tid;

    // Initialize shared values to minimal value.
    sharedValues[tid] = minimal;

    // Do first set of comparisons outside loop, saves one iteration.
    if(i + blockDim.x < end) { // Are we in a position for which we can access and compare two values in a row partition (shifted by block size)?
      // yes, hence compare:
      float a = flip * (float)inValues[i];              // value from first half of row parition for this block
      float b = flip * (float)inValues[i + blockDim.x]; // value from second half of row partition for this block
      if(a > b) { // just a max
        sharedIndices[tid] = i;
        sharedValues[tid]  = a;
      } else {
        sharedIndices[tid] = i + blockDim.x;
        sharedValues[tid]  = b;
      }
    } else if(i < end) { // Are we instead in a position that has access to one value in the row partition (shifting by block size would be out of bounds)?
      // Yes, hence save the current value and index as new max, no need to compare.
      sharedIndices[tid] = i;
      sharedValues[tid]  = flip * (float)inValues[i];
    } // nothing else to do here

    // We move to the next set of 1024 values shifted by block size times number of blocks
    // and look at two of them according to thread id.
    while(i + 2 * gridDim.x * blockDim.x < end) {
      i += 2 * gridDim.x * blockDim.x;

      // Check if first value is larger than what we have seen so far
      float a = flip * (float)inValues[i];
      if(a > sharedValues[tid]) {
        // Yes, hence save index and value
        sharedIndices[tid] = i;
        sharedValues[tid]  = a;
      }

      // Check if second value is larger than what we have seen so far
      if(i + blockDim.x < end) {
        float b = flip * (float)inValues[i + blockDim.x];
        if(b > sharedValues[tid]) {
          // Yes, hence save index and value
          sharedIndices[tid] = i + blockDim.x;
          sharedValues[tid]  = b;
        }
      }
    }

    // We are done with the first sweep and have populated shared memory, time to wait for the other threads and reduce it all
    __syncthreads();

    // Reduce over shared memory, here per loop until we hit the last 32 unreduced elements
    for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if(tid < s && tid + s < end) {
        if(sharedValues[tid + s] > sharedValues[tid]) {
          // keep the max
          sharedIndices[tid] = sharedIndices[tid + s];
          sharedValues[tid]  = sharedValues[tid + s];
        }
      }
      __syncthreads();
    }

    // Reduce over shared memory, here per unrolled code for powers of 2 lower equal 32.
    // Because we are at 32 (warp size) the threads run in lock-step and we can abandon syncing.
    UNROLL_MAXARG_LOOP(32, end);
    UNROLL_MAXARG_LOOP(16, end);
    UNROLL_MAXARG_LOOP(8, end);
    UNROLL_MAXARG_LOOP(4, end);
    UNROLL_MAXARG_LOOP(2, end);
    UNROLL_MAXARG_LOOP(1, end);

    // OK, we are done with the reduction and in the first thread
    if(tid == 0) {
      // assign the final maximal value to the bin, one bin per row and block
      binIndices[rowIdx * gridDim.x + blockIdx.x] = sharedIndices[0]; // [rows, num_blocks]
      binValues[rowIdx * gridDim.x + blockIdx.x]  = sharedValues[0];  // [rows, num_blocks]
    }
    __syncthreads();
  }
}

// This runs after the function above, we now have the maximum value per row and block and can look further
// for the top-k results. As above we pretend this does only maximum search.
// This runs restricted to one row (one row per block)
template <typename T>
__global__ void gMaxElementUpdate(IndexType* binIndices, // memory for bin indices
                                  T* binValues,          // memory for bin costs
                                  IndexType* outIndices, // result indices
                                  T* outValues,          // result costs
                                  T* inValues,           // should work well enough with half, uses float everywhere else
                                  const int cols,        // size of continous memory we search over
                                  const int K,           // how many top-K elements?
                                  int numBlocks,        // number of blocks/bins used in above function (per row)
                                  float minimal,         // value for minimal element
                                  bool descending)
{
  extern __shared__ float sharedValues[];
  __shared__ int    sharedIndices[BLOCK_SIZE];
  __shared__ float  bestBinCost;
  __shared__ int    bestBinCostIdx;

  const int tid    = threadIdx.x;

  float flip = descending ? 1.f : -1.f;

  // we only look at one row in this kernel
  const int rowIdx = blockIdx.x;           // index of the row corresponds to block index
  const int begin  = rowIdx * cols;        // start offset for this row relative to inValues tensor start
  const int end    = rowIdx * cols + cols; // end offset for this row relative to inValues tensor start

  int num_bins = numBlocks; // why not just use numBlocks? 

  // iterate over top-k results
  for(int k = 0; k < K; ++k) {

    int kthOutIdx = rowIdx * K + k;  // offset into output tensor relative to outIndices/outValues tensor start
    int i = tid;

    sharedValues[tid] = minimal; // initialize to smallest value, everything else will be larger

    // as in the function above, the code here does a tree reduction over shared memory to find the single maximum element
    if(i + blockDim.x < num_bins) {
      float a = binValues[rowIdx * numBlocks + i];
      float b = binValues[rowIdx * numBlocks + i + blockDim.x];
      if(a > b) {
        sharedValues[tid] = a;
        sharedIndices[tid] = i;
      } else {
        sharedValues[tid] = b;
        sharedIndices[tid] = i + blockDim.x;
      }
    } else if(i < num_bins) {
      sharedValues[tid] = binValues[rowIdx * numBlocks + i];
      sharedIndices[tid] = i;
    }

    while(i + 2 * blockDim.x < num_bins) {
      i += 2 * blockDim.x;

      float a = binValues[rowIdx * numBlocks + i];
      if(a > sharedValues[tid]) {
        sharedValues[tid] = a;
        sharedIndices[tid] = i;
      }

      if(i + blockDim.x < num_bins) {
        float b = binValues[rowIdx * numBlocks + i + blockDim.x];
        if(b > sharedValues[tid]) {
          sharedValues[tid] = b;
          sharedIndices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if(tid < s && tid + s < num_bins) {
        if(sharedValues[tid + s] > sharedValues[tid]) {
          sharedValues[tid] = sharedValues[tid + s];
          sharedIndices[tid] = sharedIndices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, num_bins);
    UNROLL_MAXARG_LOOP(16, num_bins);
    UNROLL_MAXARG_LOOP(8, num_bins);
    UNROLL_MAXARG_LOOP(4, num_bins);
    UNROLL_MAXARG_LOOP(2, num_bins);
    UNROLL_MAXARG_LOOP(1, num_bins);

    if(tid == 0) {
      bestBinCost = sharedValues[0];
      bestBinCostIdx = rowIdx * numBlocks + sharedIndices[0];

      inValues[binIndices[bestBinCostIdx]] = flip * minimal; // this is restored in the last lines of this function

      outIndices[kthOutIdx] = binIndices[bestBinCostIdx] - begin; // relative to beginning of row hence substract `begin`
      outValues[kthOutIdx]  = flip * bestBinCost; // undo flip by flipping again
    }

    __syncthreads();

    // Second part of the algorithm, why it that not replacing the first function call?? 
    // Also shouldn't we skip here if k == K - 1?

    // After marking the previously largest element with "flip * minimal" we populate again
    // shared memory with the largest element as in gMaxElement(...)

    if(k < K - 1) {
      i = begin + (bestBinCostIdx - rowIdx * numBlocks) * (blockDim.x * 2) + tid;
      const int dist = num_bins * 2 * blockDim.x;

      sharedValues[tid] = minimal;

      if(i + blockDim.x < end) {
        float a = flip * (float)inValues[i];
        float b = flip * (float)inValues[i + blockDim.x];
        if(a > b) {
          sharedIndices[tid] = i;
          sharedValues[tid]  = a;
        } else {
          sharedIndices[tid] = i + blockDim.x;
          sharedValues[tid]  = b;
        }
      } else if(i < end) {
        sharedIndices[tid] = i;
        sharedValues[tid] = flip * (float)inValues[i];
      }

      while(i + dist < end) {
        i += dist;

        float a = flip * (float)inValues[i];
        if(a > sharedValues[tid]) {
          sharedIndices[tid] = i;
          sharedValues[tid]  = a;
        }

        if(i + blockDim.x < end) {
          float b = flip * (float)inValues[i + blockDim.x];
          if(b > sharedValues[tid]) {
            sharedIndices[tid] = i + blockDim.x;
            sharedValues[tid] =  b;
          }
        }
      }

      __syncthreads();

      for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
        if(tid < s && tid + s < end) {
          if(sharedValues[tid + s] > sharedValues[tid]) {
            sharedIndices[tid] = sharedIndices[tid + s];
            sharedValues[tid] = sharedValues[tid + s];
          }
        }
        __syncthreads();
      }

      UNROLL_MAXARG_LOOP(32, end);
      UNROLL_MAXARG_LOOP(16, end);
      UNROLL_MAXARG_LOOP(8, end);
      UNROLL_MAXARG_LOOP(4, end);
      UNROLL_MAXARG_LOOP(2, end);
      UNROLL_MAXARG_LOOP(1, end);

      if(tid == 0) {
        binIndices[bestBinCostIdx] = sharedIndices[0];
        binValues[bestBinCostIdx]  = sharedValues[0];
      }
      __syncthreads();
    }
  }

  // final operation to restore blanked-out input values. They were blanked out for marking
  // already found values. Since we want input values to be invariant we restore here.
  // @TODO: The lack of constness here might be a problem for concurrent processing (which we currently don't have)
  for(int k = tid; k < K; k += blockDim.x) {
    int kthOutIdx = rowIdx * K + k;
    inValues[begin + outIndices[kthOutIdx]] = outValues[kthOutIdx];
  }
}

void TopK(Tensor outVal, Tensor outInd, Ptr<Allocator> allocator, const Tensor in, int k, int axis, bool descending) {
  
  ABORT_IF(axis != in->shape().size() - 1, "Currently only works for last axis");
  ABORT_IF(!isFloat(in->type()), "Input should be float type and not {}", in->type());
  ABORT_IF(outInd->type() != Type::uint32, "Output should be have type {}", Type::uint32);
  ABORT_IF(outVal->type() != in->type(), "Output should be have type {}", in->type());

  cudaSetDevice(outInd->getDeviceId().no);

  int cols = in->shape()[-1];               // e.g. in beam search that would be [beam * dimVoc]
  int rows = in->shape().elements() / cols; // e.g. in beam search that would be [time * batch]

  ABORT_IF(k > cols, "Cannot select more than {} elements for axis {}", cols, axis);

  float minimal = NumericLimits<float>(in->type()).lowest; // lowest if looking for max

  const int numBlocks = std::min(MAX_BINS, int(cols / (2 * BLOCK_SIZE)) + int(cols % (2 * BLOCK_SIZE) != 0));
  auto tempMemInd = allocator->alloc<IndexType>(rows * numBlocks);

  MemoryPiece::PtrType tempMemVal;
  if(in->type() == Type::float32) {
    tempMemVal = allocator->alloc<float>(rows * numBlocks);
    // first find the maximum value per row and block and save indices and values to temporary memory
    gMaxElement<<<numBlocks, // blocks
                  BLOCK_SIZE, // threads
                  BLOCK_SIZE * sizeof(float), // shared memory size
                  /* stream_ */ 0>>>(
      tempMemInd->data<IndexType>(), tempMemVal->data<float>(),
      in->data<float>(), rows, cols, minimal, descending);
    gMaxElementUpdate<<<rows,       // blocks ... seems we can have up to 2^31-1 of these, so we are safe?
                        BLOCK_SIZE, // threads
                        BLOCK_SIZE * sizeof(float),  // shared memory size
                        /* stream_ */ 0>>>(
      tempMemInd->data<IndexType>(), tempMemVal->data<float>(), 
      outInd->data<IndexType>(), outVal->data<float>(), 
      in->data<float>(), cols, k, numBlocks, minimal, descending);
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    tempMemVal = allocator->alloc<__half>(rows * numBlocks);
    // first find the maximum value per row and block and save indices and values to temporary memory
    gMaxElement<<<numBlocks, // blocks
                  BLOCK_SIZE, // threads
                  BLOCK_SIZE * sizeof(float), // shared memory size
                  /* stream_ */ 0>>>(
      tempMemInd->data<IndexType>(), tempMemVal->data<__half>(),
      in->data<__half>(), rows, cols, minimal, descending);
    gMaxElementUpdate<<<rows,       // blocks ... seems we can have up to 2^31-1 of these, so we are safe?
                        BLOCK_SIZE, // threads
                        BLOCK_SIZE * sizeof(float),  // shared memory size
                        /* stream_ */ 0>>>(
      tempMemInd->data<IndexType>(), tempMemVal->data<__half>(), 
      outInd->data<IndexType>(), outVal->data<__half>(), 
      in->data<__half>(), cols, k, numBlocks, minimal, descending);
#endif
  } else {
    ABORT("Topk not implemented for type {}", in->type());
  }

  allocator->free(tempMemInd);
  allocator->free(tempMemVal);
}

}
}
