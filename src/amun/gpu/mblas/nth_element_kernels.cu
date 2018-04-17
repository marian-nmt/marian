#include "nth_element_kernels.h"

using namespace std;

namespace amunmt {
namespace GPU {


#define SHARED_SIZE 512

__device__
void UnrollMaxArgLoop(unsigned n, unsigned max, unsigned tid, float *sdata, unsigned *indices)
{
  if (tid < (n) && tid + (n) < ( max ) ) {
    if (sdata[tid + ( n ) ] > sdata[tid]) {
      sdata[tid] = sdata[tid + ( n ) ];
      indices[tid] = indices[tid + ( n ) ];
    }
  }
}

__global__ void gMaxElement(mblas::VectorWrapper<NthOut> out,
                            const mblas::TensorWrapper<float> probsWrap,
                            const mblas::VectorWrapper<unsigned> batchPositionWrap,
                            unsigned numBatches) {
  extern __shared__ float sdata[];
  __shared__ unsigned indices[SHARED_SIZE];

  unsigned tid = threadIdx.x;

  for (unsigned batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    unsigned begin = batchPositionWrap[batchIdx];
    unsigned end = batchPositionWrap[batchIdx + 1];

    unsigned i = begin + blockIdx.x * (blockDim.x * 2) + tid;

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

    for (unsigned s = (blockDim.x >> 1); s > 32; s >>= 1) {
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

__global__ void gMaxElementUpdate(mblas::VectorWrapper<NthOut> out,
                                  mblas::TensorWrapper<float> probsWrap,
                                  mblas::VectorWrapper<NthOut> resWrap,
                                  const mblas::VectorWrapper<unsigned> batchPositionWrap,
                                  const mblas::VectorWrapper<unsigned> cumBeamSizesWrap,
                                  unsigned numBlocks) {
  extern __shared__ float sdata[];
  __shared__ unsigned indices[SHARED_SIZE];
  __shared__ float bestBinCost;
  __shared__ unsigned bestBinCostIdx;

  const unsigned tid = threadIdx.x;
  const unsigned batchIdx = blockIdx.x;
  const unsigned N = batchPositionWrap[batchIdx + 1] - batchPositionWrap[batchIdx];
  unsigned num_bins = unsigned(N / (2 * SHARED_SIZE)) + unsigned(N % (2 * SHARED_SIZE) != 0);
  //if (num_bins > 500) {
  //  num_bins = 500;
  //}

  for (unsigned pos = cumBeamSizesWrap[batchIdx]; pos < cumBeamSizesWrap[batchIdx + 1]; ++pos) {
    unsigned i = tid;

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

    for (unsigned s = (blockDim.x >> 1); s > 32; s >>= 1) {
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

      resWrap[pos].ind = out[bestBinCostIdx].ind;
      resWrap[pos].score = bestBinCost;
    }

    __syncthreads();

    i = batchPositionWrap[batchIdx] + (bestBinCostIdx - batchIdx * numBlocks) * (blockDim.x * 2) + tid;
    const unsigned dist = num_bins * 2 * blockDim.x;

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

    for (unsigned s = (blockDim.x >> 1); s > 32; s >>= 1) {
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

__global__ void gGetValueByKey(mblas::TensorWrapper<float> out,
                              const   mblas::TensorWrapper<float> in,
                              unsigned* indices, unsigned n)
{
  unsigned tid = threadIdx.x  + blockDim.x * blockIdx.x;
  if (tid < n) {
    unsigned index = indices[tid];
    out[tid] = in[index];
  }
}

}
}

