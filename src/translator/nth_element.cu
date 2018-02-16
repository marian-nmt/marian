#include <iostream>

#include "translator/nth_element.h"

#include "kernels/cuda_helpers.h"

namespace marian {

#define UNROLL_MAXARG_LOOP(n, max)       \
  if(tid < (n) && tid + (n) < (max)) {   \
    if(sdata[tid + (n)] > sdata[tid]) {  \
      sdata[tid] = sdata[tid + (n)];     \
      indices[tid] = indices[tid + (n)]; \
    }                                    \
  }

__global__ void gMaxElement(float* d_out,
                            int* d_ind,
                            float* d_in,
                            int numBatches,
                            int* batchFirstElementIdxs) {
  extern __shared__ float sdata[];
  __shared__ int indices[512];

  int tid = threadIdx.x;

  for(int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    int begin = batchFirstElementIdxs[batchIdx];
    int end = batchFirstElementIdxs[batchIdx + 1];

    int i = begin + blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = -3.40282e+38f;

    if(i < end) {
      sdata[tid] = d_in[i];
      indices[tid] = i;
    }

    if(i + blockDim.x < end) {
      float a = d_in[i];
      float b = d_in[i + blockDim.x];
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

      float a = d_in[i];
      if(a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if(i + blockDim.x < end) {
        float b = d_in[i + blockDim.x];
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

__global__ void gMaxElementUpdate(float* binCosts,
                                  int* binIdxs,
                                  float* probs,
                                  int* batchFirstElements,
                                  float* outCosts,
                                  int* outIdxs,
                                  int* cummulatedBeamSizes,
                                  int NUM_BLOCKS) {
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

  for(int pos = cummulatedBeamSizes[batchIdx];
      pos < cummulatedBeamSizes[batchIdx + 1];
      ++pos) {
    int i = tid;

    sdata[tid] = -3.40282e+38f;

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

      probs[binIdxs[bestBinCostIdx]] = -3.40282e+38f;

      outIdxs[pos] = binIdxs[bestBinCostIdx];
      outCosts[pos] = bestBinCost;
    }

    __syncthreads();

    i = batchFirstElements[batchIdx]
        + (bestBinCostIdx - batchIdx * NUM_BLOCKS) * (blockDim.x * 2) + tid;
    const int dist = num_bins * 2 * blockDim.x;

    sdata[tid] = -3.40282e+38f;

    if(i < batchFirstElements[batchIdx + 1]) {
      sdata[tid] = probs[i];
      indices[tid] = i;
    }

    if(i + blockDim.x < batchFirstElements[batchIdx + 1]) {
      float a = probs[i];
      float b = probs[i + blockDim.x];
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

      float a = probs[i];
      if(a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if(i + blockDim.x < batchFirstElements[batchIdx + 1]) {
        float b = probs[i + blockDim.x];
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

NthElement::NthElement(size_t maxBeamSize,
                       size_t maxBatchSize,
                       DeviceId deviceId)
    : deviceId_(deviceId),
      NUM_BLOCKS(std::min(
          500,
          int(maxBeamSize* MAX_VOCAB_SIZE / (2 * BLOCK_SIZE))
              + int(maxBeamSize* MAX_VOCAB_SIZE % (2 * BLOCK_SIZE) != 0))) {
  // std::cerr << "NthElement::NthElement" << std::endl;

  cudaSetDevice(deviceId_.no);

  CUDA_CHECK(
      cudaMalloc((void**)&d_ind, maxBatchSize * NUM_BLOCKS * sizeof(int)));

  CUDA_CHECK(
      cudaMalloc((void**)&d_out, maxBatchSize * NUM_BLOCKS * sizeof(float)));

  CUDA_CHECK(
      cudaMalloc((void**)&d_res_idx, maxBatchSize * maxBeamSize * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void**)&d_res, maxBatchSize * maxBeamSize * sizeof(float)));

  CUDA_CHECK(cudaHostAlloc((void**)&h_res,
                           maxBeamSize * maxBatchSize * sizeof(float),
                           cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void**)&h_res_idx,
                           maxBeamSize * maxBatchSize * sizeof(int),
                           cudaHostAllocDefault));

  CUDA_CHECK(cudaMalloc((void**)&d_breakdown, maxBeamSize * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void**)&d_batchPosition, (maxBatchSize + 1) * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc((void**)&d_cumBeamSizes, (maxBatchSize + 1) * sizeof(int)));
}

NthElement::~NthElement() {
  cudaSetDevice(deviceId_.no);
  
  CUDA_CHECK(cudaFree(d_ind));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_res_idx));
  CUDA_CHECK(cudaFree(d_res));
  CUDA_CHECK(cudaFreeHost(h_res));
  CUDA_CHECK(cudaFreeHost(h_res_idx));
  CUDA_CHECK(cudaFree(d_breakdown));
  CUDA_CHECK(cudaFree(d_batchPosition));
  CUDA_CHECK(cudaFree(d_cumBeamSizes));
}

void NthElement::getNBestList(float* probs,
                              const std::vector<int>& batchFirstElementIdxs,
                              const std::vector<int>& cummulatedBeamSizes) {
  cudaSetDevice(deviceId_.no);
  CUDA_CHECK(cudaMemcpyAsync(d_batchPosition,
                             batchFirstElementIdxs.data(),
                             batchFirstElementIdxs.size() * sizeof(int),
                             cudaMemcpyHostToDevice,
                             /* stream_ */ 0));
  CUDA_CHECK(cudaMemcpyAsync(d_cumBeamSizes,
                             cummulatedBeamSizes.data(),
                             cummulatedBeamSizes.size() * sizeof(int),
                             cudaMemcpyHostToDevice,
                             /* stream_ */ 0));

  const int numBatches = batchFirstElementIdxs.size() - 1;

  gMaxElement<<<NUM_BLOCKS,
                BLOCK_SIZE,
                BLOCK_SIZE * sizeof(float),
                /* stream_ */ 0>>>(
      d_out, d_ind, probs, numBatches, d_batchPosition);

  gMaxElementUpdate<<<numBatches,
                      BLOCK_SIZE,
                      BLOCK_SIZE * sizeof(float),
                      /* stream_ */ 0>>>(d_out,
                                         d_ind,
                                         probs,
                                         d_batchPosition,
                                         d_res,
                                         d_res_idx,
                                         d_cumBeamSizes,
                                         NUM_BLOCKS);
}

void NthElement::getNBestList(const std::vector<size_t>& beamSizes,
                              Tensor Probs,
                              std::vector<float>& outCosts,
                              std::vector<unsigned>& outKeys,
                              const bool isFirst) {
  cudaSetDevice(deviceId_.no);

  std::vector<int> cummulatedBeamSizes(beamSizes.size() + 1, 0);
  std::vector<int> batchFirstElementIdxs(beamSizes.size() + 1, 0);

  const size_t vocabSize = Probs->shape()[-1];

  for(size_t i = 0; i < beamSizes.size(); ++i) {
    cummulatedBeamSizes[i + 1] = cummulatedBeamSizes[i] + beamSizes[i];
    batchFirstElementIdxs[i + 1]
        += ((isFirst) ? (i + 1) : cummulatedBeamSizes[i + 1]) * vocabSize;
  }

  getNBestList(Probs->data(), batchFirstElementIdxs, cummulatedBeamSizes);
  GetPairs(cummulatedBeamSizes.back(), outKeys, outCosts);
}

void NthElement::GetPairs(size_t number,
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

  lastN = number;
}

void NthElement::getValueByKey(std::vector<float>& out, float* d_in) {
  cudaSetDevice(deviceId_.no);

  gGetValueByKey<<<1, lastN, 0, /* stream_ */ 0>>>(
      d_in, d_breakdown, h_res_idx, lastN);

  CUDA_CHECK(cudaMemcpyAsync(out.data(),
                             d_breakdown,
                             lastN * sizeof(float),
                             cudaMemcpyDeviceToHost,
                             /* stream_ */ 0));
  CUDA_CHECK(cudaStreamSynchronize(/* stream_ */ 0));
}
}
