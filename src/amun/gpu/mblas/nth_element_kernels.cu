#include "nth_element_kernels.h"

namespace amunmt {
namespace GPU {


#define SHARED_SIZE 512

__device__
void UnrollMaxArgLoop(uint n, uint max, uint tid, float *sdata, uint *indices)
{
  if (tid < (n) && tid + (n) < ( max ) ) {
    if (sdata[tid + ( n ) ] > sdata[tid]) {
      sdata[tid] = sdata[tid + ( n ) ];
      indices[tid] = indices[tid + ( n ) ];
    }
  }
}

__global__ void gMaxElement(mblas::MatrixWrapper<NthOut> out,
                            const mblas::MatrixWrapper<float> probsWrap,
                            const mblas::MatrixWrapper<uint> batchPositionWrap,
                            uint numBatches) {
  extern __shared__ float sdata[];
  __shared__ uint indices[SHARED_SIZE];

  uint tid = threadIdx.x;

  for (uint batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    uint begin = batchPositionWrap[batchIdx];
    uint end = batchPositionWrap[batchIdx + 1];

    uint i = begin + blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = -3.40282e+38f;

    if (i < end) {
      sdata[tid] = probsWrap[i];
      indices[tid] = i;
    }

    if (i + blockDim.x < end) {
      float a = probsWrap[i];
      float b = probsWrap[i + blockDim.x];
      if (a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + 2 * gridDim.x * blockDim.x < end) {
      i += 2 * gridDim.x * blockDim.x;

      float a = probsWrap[i];
      if (a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < end) {
        float b = probsWrap[i + blockDim.x];
        if (b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < end) {
        if (sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UnrollMaxArgLoop(32, end, tid, sdata, indices);
    UnrollMaxArgLoop(16, end, tid, sdata, indices);
    UnrollMaxArgLoop(8, end, tid, sdata, indices);
    UnrollMaxArgLoop(4, end, tid, sdata, indices);
    UnrollMaxArgLoop(2, end, tid, sdata, indices);
    UnrollMaxArgLoop(1, end, tid, sdata, indices);

    if (tid == 0) {
      out[blockIdx.x + batchIdx * gridDim.x] = {indices[0], sdata[0]};
    }
    __syncthreads();
  }
}

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut> out,
                                  mblas::MatrixWrapper<float> probsWrap,
                                  mblas::MatrixWrapper<NthOut> resNewWrap,
                                  const mblas::MatrixWrapper<uint> batchPositionWrap,
                                  const mblas::MatrixWrapper<uint> cumBeamSizesWrap,
                                  uint numBlocks) {
  extern __shared__ float sdata[];
  __shared__ uint indices[SHARED_SIZE];
  __shared__ float bestBinCost;
  __shared__ uint bestBinCostIdx;

  const uint tid = threadIdx.x;
  const uint batchIdx = blockIdx.x;
  const uint N = batchPositionWrap[batchIdx + 1] - batchPositionWrap[batchIdx];
  uint num_bins = uint(N / (2 * SHARED_SIZE)) + uint(N % (2 * SHARED_SIZE) != 0);
  //if (num_bins > 500) {
  //  num_bins = 500;
  //}

  for (uint pos = cumBeamSizesWrap[batchIdx]; pos < cumBeamSizesWrap[batchIdx + 1]; ++pos) {
    uint i = tid;

    sdata[tid] = -3.40282e+38f;

    if (i < num_bins) {
      sdata[tid] = out[batchIdx * numBlocks + i].score;
      indices[tid] = i;
    }

    if (i + blockDim.x < num_bins) {
      float a = out[batchIdx * numBlocks + i].score;
      float b = out[batchIdx * numBlocks + i + blockDim.x].score;
      if (a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + 2 * blockDim.x < num_bins) {
      i += 2 * blockDim.x;

      float a = out[batchIdx * numBlocks + i].score;
      if (a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < num_bins) {
        float b = out[batchIdx * numBlocks + i + blockDim.x].score;
        if (b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < num_bins) {
        if (sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UnrollMaxArgLoop(32, num_bins, tid, sdata, indices);
    UnrollMaxArgLoop(16, num_bins, tid, sdata, indices);
    UnrollMaxArgLoop(8, num_bins, tid, sdata, indices);
    UnrollMaxArgLoop(4, num_bins, tid, sdata, indices);
    UnrollMaxArgLoop(2, num_bins, tid, sdata, indices);
    UnrollMaxArgLoop(1, num_bins, tid, sdata, indices);

    if (tid == 0) {
      bestBinCost = sdata[0];
      bestBinCostIdx = batchIdx * numBlocks + indices[0];

      probsWrap[ out[bestBinCostIdx].ind ] = -3.40282e+38f;

      resNewWrap[pos].ind = out[bestBinCostIdx].ind;
      resNewWrap[pos].score = bestBinCost;
    }

    __syncthreads();

    i = batchPositionWrap[batchIdx] + (bestBinCostIdx - batchIdx * numBlocks) * (blockDim.x * 2) + tid;
    const uint dist = num_bins * 2 * blockDim.x;

    sdata[tid] = -3.40282e+38f;

    if (i < batchPositionWrap[batchIdx + 1]) {
      sdata[tid] = probsWrap[i];
      indices[tid] = i;
    }

    if (i + blockDim.x < batchPositionWrap[batchIdx + 1]) {
      float a = probsWrap[i];
      float b = probsWrap[i+blockDim.x];
      if (a > b) {
        sdata[tid] = a;
        indices[tid] = i;
      } else {
        sdata[tid] = b;
        indices[tid] = i + blockDim.x;
      }
    }

    while (i + dist < batchPositionWrap[batchIdx + 1]) {
      i += dist;

      float a = probsWrap[i];
      if (a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < batchPositionWrap[batchIdx + 1]) {
        float b = probsWrap[i + blockDim.x];
        if (b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (uint s = (blockDim.x >> 1); s > 32; s >>= 1) {
      if (tid < s && tid + s < batchPositionWrap[batchIdx + 1]) {
        if (sdata[tid + s] > sdata[tid]) {
          sdata[tid] = sdata[tid + s];
          indices[tid] = indices[tid + s];
        }
      }
      __syncthreads();
    }

    UnrollMaxArgLoop(32, batchPositionWrap[batchIdx + 1], tid, sdata, indices);
    UnrollMaxArgLoop(16, batchPositionWrap[batchIdx + 1], tid, sdata, indices);
    UnrollMaxArgLoop(8, batchPositionWrap[batchIdx + 1], tid, sdata, indices);
    UnrollMaxArgLoop(4, batchPositionWrap[batchIdx + 1], tid, sdata, indices);
    UnrollMaxArgLoop(2, batchPositionWrap[batchIdx + 1], tid, sdata, indices);
    UnrollMaxArgLoop(1, batchPositionWrap[batchIdx + 1], tid, sdata, indices);

    if (tid == 0) {
      out[bestBinCostIdx] = {indices[0], sdata[0]};
    }
    __syncthreads();
  }
}

__global__ void gGetValueByKey(mblas::MatrixWrapper<float> out,
                              const   mblas::MatrixWrapper<float> in,
                              uint* indices, uint n)
{
  uint tid = threadIdx.x  + blockDim.x * blockIdx.x;
  if (tid < n) {
    uint index = indices[tid];
    out[tid] = in[index];
  }
}

}
}

