/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <iostream>

#include "translator/nth_element.h"

#include <cuda.h>
#include "tensors/gpu/cuda_helpers.h"

namespace marian {

#define UNROLL_MAXARG_LOOP(n, max)       \
  if(tid < (n) && tid + (n) < (max)) {   \
    if(sdata[tid + (n)] > sdata[tid]) {  \
      sdata[tid] = sdata[tid + (n)];     \
      indices[tid] = indices[tid + (n)]; \
    }                                    \
  }

template <typename T>
__global__ void gMaxElement(float* d_out,
                            int* d_ind,
                            T* d_in, // this is the probs array, only one with type float or half
                            int numBatches,
                            int* batchFirstElementIdxs,
                            float disabledPathScore) // disabledPathScore is used to blank out found values, type-dependent
{
  extern __shared__ float sdata[];
  __shared__ int indices[512];

  int tid = threadIdx.x;

  for(int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    int begin = batchFirstElementIdxs[batchIdx];
    int end = batchFirstElementIdxs[batchIdx + 1];

    int i = begin + blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = disabledPathScore;

    if(i < end) {
      sdata[tid] = (float)d_in[i];
      indices[tid] = i;
    }

    if(i + blockDim.x < end) {
      float a = (float)d_in[i];
      float b = (float)d_in[i + blockDim.x];
      if(a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while(i + 2 * gridDim.x * blockDim.x < end) {
      i += 2 * gridDim.x * blockDim.x;

      float a = (float)d_in[i];
      if(a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if(i + blockDim.x < end) {
        float b = (float)d_in[i + blockDim.x];
        if(b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if(tid < s && tid + s < end) {
        if(sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
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
      d_out[blockIdx.x + batchIdx * gridDim.x] = sdata[0];
      d_ind[blockIdx.x + batchIdx * gridDim.x] = indices[0];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void gMaxElementUpdate(float* binCosts,
                                  int* binIdxs,
                                  T* probs, // should work well enough with half, uses float everywhere else
                                  int* batchFirstElements,
                                  float* outCosts,
                                  int* outIdxs,
                                  int* cumulativeBeamSizes,
                                  int NUM_BLOCKS,
                                  float disabledPathScore) {
  extern __shared__ float sdata[];
  __shared__ int indices[512];
  __shared__ float bestBinCost;
  __shared__ int bestBinCostIdx;

  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int N = batchFirstElements[batchIdx + 1] - batchFirstElements[batchIdx];
  int num_bins = int(N / (2 * 512)) + int(N % (2 * 512) != 0);
  if(num_bins > 500) {
    num_bins = 500;
  }

  for(int pos = cumulativeBeamSizes[batchIdx];
      pos < cumulativeBeamSizes[batchIdx + 1];
      ++pos) {
    int i = tid;

    sdata[tid] = disabledPathScore;

    if(i < num_bins) {
      sdata[tid] = binCosts[batchIdx * NUM_BLOCKS + i];
      indices[tid] = i;
    }

    if(i + blockDim.x < num_bins) {
      float a = binCosts[batchIdx * NUM_BLOCKS + i];
      float b = binCosts[batchIdx * NUM_BLOCKS + i + blockDim.x];
      if(a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while(i + 2 * blockDim.x < num_bins) {
      i += 2 * blockDim.x;

      float a = binCosts[batchIdx * NUM_BLOCKS + i];
      if(a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if(i + blockDim.x < num_bins) {
        float b = binCosts[batchIdx * NUM_BLOCKS + i + blockDim.x];
        if(b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if(tid < s && tid + s < num_bins) {
        if(sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
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
      bestBinCost = sdata[0];
      bestBinCostIdx = batchIdx * NUM_BLOCKS + indices[0];

      probs[binIdxs[bestBinCostIdx]] = disabledPathScore;

      outIdxs[pos] = binIdxs[bestBinCostIdx];
      outCosts[pos] = bestBinCost;
    }

    __syncthreads();

    i = batchFirstElements[batchIdx]
        + (bestBinCostIdx - batchIdx * NUM_BLOCKS) * (blockDim.x * 2) + tid;
    const int dist = num_bins * 2 * blockDim.x;

    sdata[tid] = disabledPathScore;

    if(i < batchFirstElements[batchIdx + 1]) {
      sdata[tid] = (float)probs[i];
      indices[tid] = i;
    }

    if(i + blockDim.x < batchFirstElements[batchIdx + 1]) {
      float a = (float)probs[i];
      float b = (float)probs[i + blockDim.x];
      if(a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while(i + dist < batchFirstElements[batchIdx + 1]) {
      i += dist;

      float a = (float)probs[i];
      if(a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if(i + blockDim.x < batchFirstElements[batchIdx + 1]) {
        float b = (float)probs[i + blockDim.x];
        if(b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for(int s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if(tid < s && tid + s < batchFirstElements[batchIdx + 1]) {
        if(sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UNROLL_MAXARG_LOOP(32, batchFirstElements[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(16, batchFirstElements[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(8, batchFirstElements[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(4, batchFirstElements[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(2, batchFirstElements[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(1, batchFirstElements[batchIdx + 1]);

    if(tid == 0) {
      binCosts[bestBinCostIdx] = sdata[0];
      binIdxs[bestBinCostIdx] = indices[0];
    }
    __syncthreads();
  }
}

__global__ void gGetValueByKey(float* d_in, float* d_out, int* indeces, int n) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < n) {
    int index = indeces[tid];
    d_out[tid] = d_in[index];
  }
}

class NthElementGPU {
public:
  NthElementGPU() = delete;
  NthElementGPU(const NthElementGPU& copy) = delete;

  NthElementGPU(size_t maxBeamSize,
                size_t maxBatchSize,
                DeviceId deviceId)
      : deviceId_(deviceId),
        maxBeamSize_(maxBeamSize), maxBatchSize_(maxBatchSize),
        NUM_BLOCKS(std::min(
            500,
            int(maxBeamSize* MAX_VOCAB_SIZE / (2 * BLOCK_SIZE))
                + int(maxBeamSize* MAX_VOCAB_SIZE % (2 * BLOCK_SIZE) != 0))) {
    // std::cerr << "NthElement::NthElement" << std::endl;

    cudaSetDevice(deviceId_.no);

    CUDA_CHECK(cudaMalloc((void**)&d_ind, maxBatchSize * NUM_BLOCKS * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, maxBatchSize * NUM_BLOCKS * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_res_idx, maxBatchSize * maxBeamSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_res,     maxBatchSize * maxBeamSize * sizeof(float)));

    CUDA_CHECK(cudaHostAlloc((void**)&h_res,     maxBeamSize * maxBatchSize * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_res_idx, maxBeamSize * maxBatchSize * sizeof(int), cudaHostAllocDefault));

    CUDA_CHECK(cudaMalloc((void**)&d_breakdown, maxBeamSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_batchPosition, (maxBatchSize + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cumBeamSizes,  (maxBatchSize + 1) * sizeof(int)));
  }

  ~NthElementGPU() {
    // No CUDA error checking as this is a destructor and we cannot do anything about errors anyway.
    cudaSetDevice(deviceId_.no);
    cudaFree(d_cumBeamSizes);
    cudaFree(d_batchPosition);
    cudaFree(d_breakdown);
    cudaFreeHost(h_res_idx);
    cudaFreeHost(h_res);
    cudaFree(d_res);
    cudaFree(d_res_idx);
    cudaFree(d_out);
    cudaFree(d_ind);
  }

private:
  template <typename T>
  void selectNBest(T* probs,
                   const std::vector<int>& batchFirstElementIdxs,
                   const std::vector<int>& cumulativeBeamSizes,
                   float disabledPathScore) {

    cudaSetDevice(deviceId_.no);
    CUDA_CHECK(cudaMemcpyAsync(d_batchPosition,
                               batchFirstElementIdxs.data(),
                               batchFirstElementIdxs.size() * sizeof(int),
                               cudaMemcpyHostToDevice,
                               /* stream_ */ 0));
    CUDA_CHECK(cudaMemcpyAsync(d_cumBeamSizes,
                               cumulativeBeamSizes.data(),
                               cumulativeBeamSizes.size() * sizeof(int),
                               cudaMemcpyHostToDevice,
                               /* stream_ */ 0));

    const int numBatches = batchFirstElementIdxs.size() - 1;

    gMaxElement<<<NUM_BLOCKS,
                  BLOCK_SIZE,
                  BLOCK_SIZE * sizeof(float), // shared memory size
                  /* stream_ */ 0>>>(
        d_out, d_ind, probs, numBatches, d_batchPosition, disabledPathScore);

    gMaxElementUpdate<<<numBatches,
                        BLOCK_SIZE,
                        BLOCK_SIZE * sizeof(float),  // shared memory size
                        /* stream_ */ 0>>>(d_out,
                                           d_ind,
                                           probs,
                                           d_batchPosition,
                                           d_res,
                                           d_res_idx,
                                           d_cumBeamSizes,
                                           NUM_BLOCKS,
                                           disabledPathScore);
  }

public:
  void getNBestList(Tensor scores,
                    size_t N,
                    std::vector<float>& outCosts,
                    std::vector<unsigned>& outKeys,
                    const bool isFirst) {
    cudaSetDevice(deviceId_.no);

    const auto vocabSize = scores->shape()[-1];
    const auto inputN    = scores->shape()[-2];
    const auto dimBatch  = scores->shape()[-4];
    ABORT_IF(inputN != (isFirst ? 1 : N), "Input tensor has wrong beam dim??"); // @TODO: Remove isFirst argument altogether
    ABORT_IF(vocabSize > MAX_VOCAB_SIZE, "GetNBestList(): actual vocab size {} exceeds MAX_VOCAB_SIZE of {}", vocabSize, MAX_VOCAB_SIZE);
    ABORT_IF(dimBatch > maxBatchSize_, "GetNBestList(): actual batch size {} exceeds initialization parameter {}", dimBatch, maxBatchSize_);
    ABORT_IF(std::max(N, (size_t)inputN) > maxBeamSize_, "GetNBestList(): actual beam size {} exceeds initialization parameter {}", N, maxBeamSize_);

    const std::vector<size_t> beamSizes(dimBatch, N);
    std::vector<int> cumulativeBeamSizes(beamSizes.size() + 1, 0);    
    std::vector<int> batchFirstElementIdxs(beamSizes.size() + 1, 0);

     for(size_t batchIdx = 0; batchIdx < beamSizes.size(); ++batchIdx) {
#if 1
      cumulativeBeamSizes[batchIdx + 1] = (batchIdx + 1) * (int)N;
      batchFirstElementIdxs[batchIdx + 1] += (batchIdx + 1) * inputN * vocabSize;
      ABORT_IF(cumulativeBeamSizes[batchIdx + 1] != cumulativeBeamSizes[batchIdx] + (int)N, "cumulativeBeamSizes wrong??");
      ABORT_IF((isFirst ? batchIdx + 1 : cumulativeBeamSizes[batchIdx + 1]) != (batchIdx + 1) * inputN, "inputN wrong??");
#else
      cumulativeBeamSizes[batchIdx + 1] = cumulativeBeamSizes[batchIdx] + beamSizes[batchIdx];
      ABORT_IF(cumulativeBeamSizes[batchIdx + 1] != (batchIdx + 1) * N, "cumulativeBeamSizes wrong??");
      batchFirstElementIdxs[batchIdx + 1]
          += ((isFirst) ? (batchIdx + 1) : cumulativeBeamSizes[batchIdx + 1]) * vocabSize;
      ABORT_IF((isFirst ? batchIdx + 1 : cumulativeBeamSizes[batchIdx + 1]) != (batchIdx + 1) * inputN, "inputN wrong??");
#endif
    }

    if(scores->type() == Type::float32) {
      float disabledPathScore = NumericLimits<float>(scores->type()).lowest;
      selectNBest(scores->data<float>(), batchFirstElementIdxs, cumulativeBeamSizes, disabledPathScore);
#if COMPILE_FP16
    } else if(scores->type() == Type::float16) {
      float disabledPathScore = NumericLimits<float>(scores->type()).lowest;
      selectNBest(scores->data<half>(), batchFirstElementIdxs, cumulativeBeamSizes, disabledPathScore);
#endif
    } else {
      ABORT("getNBestList not implemented for type {}", scores->type());
    }
    getPairs(dimBatch * N, outKeys, outCosts);
    ABORT_IF(cumulativeBeamSizes.back() != dimBatch * N, "cumulativeBeamSizes.back() wrong??");
  }

private:
  void getPairs(size_t number,
                std::vector<unsigned>& outKeys,
                std::vector<float>& outValues) {
    cudaSetDevice(deviceId_.no);
    CUDA_CHECK(cudaMemcpyAsync(h_res,
                               d_res,
                               number * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               /* stream_ */ 0));
    CUDA_CHECK(cudaMemcpyAsync(h_res_idx,
                               d_res_idx,
                               number * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               /* stream_ */ 0));
    cudaStreamSynchronize(/* stream_ */ 0);

    for(size_t i = 0; i < number; ++i) {
      outKeys.push_back(h_res_idx[i]);
      outValues.push_back(h_res[i]);
    }

    //lastN = number;
  }

  DeviceId deviceId_;

  const int MAX_VOCAB_SIZE = 500000;
  size_t maxBeamSize_;
  size_t maxBatchSize_;

  const int BLOCK_SIZE = 512;
  const int NUM_BLOCKS;

  int* d_ind;           // [maxBatchSize * NUM_BLOCKS]
  float* d_out;         // [maxBatchSize * NUM_BLOCKS]

  int* d_res_idx;       // [maxBatchSize * maxBeamSize]
  float* d_res;         // [maxBatchSize * maxBeamSize]

  int* h_res_idx;       // [maxBeamSize * maxBatchSize]
  float* h_res;         // [maxBeamSize * maxBatchSize]

  float* d_breakdown;   // [maxBeamSize]
  int* d_batchPosition; // [maxBatchSize + 1]
  int* d_cumBeamSizes;  // [maxBatchSize + 1]
  //size_t lastN;
};

// factory function
// Returns a lambda with the same signature as the getNBestList() function.
GetNBestListFn createGetNBestListGPUFn(size_t beamSize, size_t dimBatch, DeviceId deviceId) {
  auto nth = New<NthElementGPU>(beamSize, dimBatch, deviceId);
  return [nth](Tensor logProbs, size_t N, std::vector<float>& outCosts, std::vector<unsigned>& outKeys, const bool isFirst) {
    return nth->getNBestList(logProbs, N, outCosts, outKeys, isFirst);
  };
}

}  // namespace marian
