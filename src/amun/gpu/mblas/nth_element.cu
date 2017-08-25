#include <iostream>
#include "common/utils.h"
#include "matrix_wrapper.h"
#include "nth_element.h"
#include "matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

#define SHARED_SIZE 512

#define UNROLL_MAXARG_LOOP( n, max ) \
  if (tid < (n) && tid + (n) < ( max ) ) { \
    if (sdata[tid + ( n ) ] > sdata[tid]) { \
      sdata[tid] = sdata[tid + ( n ) ]; \
      indices[tid] = indices[tid + ( n ) ]; \
    } \
  }

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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

    UNROLL_MAXARG_LOOP(32, end);
    UNROLL_MAXARG_LOOP(16, end);
    UNROLL_MAXARG_LOOP(8, end);
    UNROLL_MAXARG_LOOP(4, end);
    UNROLL_MAXARG_LOOP(2, end);
    UNROLL_MAXARG_LOOP(1, end);

    if (tid == 0) {
      out[blockIdx.x + batchIdx * gridDim.x] = {indices[0], sdata[0]};
    }
    __syncthreads();
  }
}

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<NthOut> out,
                                  mblas::MatrixWrapper<float> probsWrap,
                                  mblas::MatrixWrapper<uint> batchPositionWrap,
                                  mblas::MatrixWrapper<NthOut> resNewWrap,
                                  mblas::MatrixWrapper<uint> cumBeamSizesWrap,
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

    UNROLL_MAXARG_LOOP(32, num_bins);
    UNROLL_MAXARG_LOOP(16, num_bins);
    UNROLL_MAXARG_LOOP(8, num_bins);
    UNROLL_MAXARG_LOOP(4, num_bins);
    UNROLL_MAXARG_LOOP(2, num_bins);
    UNROLL_MAXARG_LOOP(1, num_bins);

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

    UNROLL_MAXARG_LOOP(32, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(16, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(8, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(4, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(2, batchPositionWrap[batchIdx + 1]);
    UNROLL_MAXARG_LOOP(1, batchPositionWrap[batchIdx + 1]);

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

NthElement::NthElement(uint maxBeamSize, uint maxBatchSize)
: d_breakdown(maxBeamSize, 1, 1, 1)
, maxBeamSize_(maxBeamSize)
, maxBatchSize_(maxBatchSize)
{
  //cerr << "maxBatchSize=" << maxBatchSize << " maxBeamSize=" << maxBeamSize << endl;

  d_batchPosition.reserve(maxBatchSize + 1);
  d_cumBeamSizes.reserve(maxBatchSize + 1);

  d_res.reserve(maxBatchSize * maxBeamSize);
  h_res.reserve(maxBatchSize * maxBeamSize);
}

NthElement::~NthElement()
{
  //cerr << "FOO2" << endl;
}

void NthElement::getNBestList(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                  std::vector<float>& outCosts, std::vector<uint>& outKeys,
                  const bool isFirst) {
  /*
  cerr << "beamSizes=" << beamSizes.size() << endl;
  cerr << Debug(beamSizes, 2) << endl;
  cerr << "Probs=" << Probs.Debug(0) << endl;
  cerr << "outCosts=" << outCosts.size() << endl;
  cerr << "outKeys=" << outKeys.size() << endl;
  cerr << "isFirst=" << isFirst << endl;
  cerr << endl;
  */
  HostVector<uint> cummulatedBeamSizes(beamSizes.size() + 1);
  HostVector<uint> batchFirstElementIdxs(beamSizes.size() + 1);
  cummulatedBeamSizes[0] = 0;
  batchFirstElementIdxs[0] = 0;

  const uint vocabSize = Probs.dim(1);
  for (uint i = 0; i < beamSizes.size(); ++i) {

    cummulatedBeamSizes[i + 1] = cummulatedBeamSizes[i] + beamSizes[i];
    batchFirstElementIdxs[i + 1] = ((isFirst) ? (i + 1) : cummulatedBeamSizes[i + 1]) * vocabSize;
  }

  uint numHypos = cummulatedBeamSizes.back();
  d_res.NewSize(numHypos, 1, 1, 1);
  h_res.resize(numHypos);

  //cerr << endl;
  //cerr << "numHypos=" << numHypos << endl;
  //cerr << "beamSizes=" << Debug(beamSizes, 2) << endl;
  //cerr << "cummulatedBeamSizes=" << Debug(cummulatedBeamSizes, 2) << endl;
  //cerr << "batchFirstElementIdxs=" << Debug(batchFirstElementIdxs, 2) << endl;
  //cerr << "1Probs=" << Probs.Debug() << endl;

  getNBestList(Probs, batchFirstElementIdxs, cummulatedBeamSizes);

  //cerr << "2Probs=" << Probs.Debug() << endl;
  //cerr << "cummulatedBeamSizes.back()=" << cummulatedBeamSizes.back() << endl;
  //cerr << "cummulatedBeamSizes=" << Debug(cummulatedBeamSizes, 2) << endl;
  GetPairs(numHypos, outKeys, outCosts);

  //cerr << "outCosts=" << Debug(outCosts, 2) << endl;
  //cerr << "outKeys=" << Debug(outKeys, 2) << endl;
}

void NthElement::getNBestList(mblas::Matrix &probs,
                              const HostVector<uint>& batchFirstElementIdxs,
                              const HostVector<uint>& cummulatedBeamSizes)
{
  const uint vocabSize = probs.dim(1);
  const uint numBlocks = uint(maxBeamSize_ * vocabSize / (2 * BLOCK_SIZE)) + uint(maxBeamSize_ * vocabSize % (2 * BLOCK_SIZE) != 0);
  const uint numBatches = batchFirstElementIdxs.size() - 1;

  d_out.NewSize(maxBatchSize_ * numBlocks, 1, 1, 1);

  //cerr << "cummulatedBeamSizes=" << cummulatedBeamSizes.size() << endl;
  d_batchPosition.NewSize(batchFirstElementIdxs.size(), 1, 1, 1);
  d_cumBeamSizes.NewSize(cummulatedBeamSizes.size(), 1, 1, 1);
  assert(d_batchPosition.size() == d_cumBeamSizes.size());

  mblas::copy(thrust::raw_pointer_cast(batchFirstElementIdxs.data()),
              batchFirstElementIdxs.size(),
              d_batchPosition.data(),
              cudaMemcpyHostToDevice);
  mblas::copy(thrust::raw_pointer_cast(cummulatedBeamSizes.data()),
              cummulatedBeamSizes.size(),
              d_cumBeamSizes.data(),
              cudaMemcpyHostToDevice);

  mblas::MatrixWrapper<NthOut> outWrap(d_out);
  mblas::MatrixWrapper<float> probsWrap(probs);
  mblas::MatrixWrapper<uint> batchPositionWrap(d_batchPosition);
  mblas::MatrixWrapper<NthOut> resWrap(d_res, false);
  mblas::MatrixWrapper<uint> cumBeamSizesWrap(d_cumBeamSizes);

  gMaxElement<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
    (outWrap, probsWrap, batchPositionWrap, numBatches);

  gMaxElementUpdate<<<numBatches, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
    (outWrap, probsWrap, batchPositionWrap, resWrap, cumBeamSizesWrap,
     numBlocks);

  /*
  cerr << "numBlocks=" << numBlocks << endl;
  cerr << "numBatches=" << numBatches << endl;
  cerr << "threads=" << BLOCK_SIZE << endl;

  cerr << "outWrap=" << outWrap.Debug() << endl;

  cerr << "probsWrap=" << probsWrap.Debug() << endl;

  cerr << "batchPositionWrap=" << batchPositionWrap.Debug() << endl;
  cerr << mblas::Debug(d_batchPosition, 2) << endl;

  cerr << "resWrap=" << resWrap.Debug() << endl;
  cerr << mblas::Debug(d_res, 2) << endl;

  cerr << "cumBeamSizesWrap=" << cumBeamSizesWrap.Debug() << endl;
  //cerr << mblas::Debug(d_cumBeamSizes, 2) << endl;

  cerr << endl;
  */
}

void NthElement::GetPairs(uint number,
                    std::vector<uint>& outKeys,
                    std::vector<float>& outValues)
{
  mblas::copy(d_res.data(), d_res.size(), thrust::raw_pointer_cast(h_res.data()), cudaMemcpyDeviceToHost);
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()) );

  for (uint i = 0; i < number; ++i) {
    outKeys.push_back(h_res[i].ind);
    outValues.push_back(h_res[i].score);
  }
}

void NthElement::getValueByKey(std::vector<float>& out, const mblas::Matrix &d_in) const
{
  // need a model with multiple scorers to test this method
  assert(false);

  mblas::MatrixWrapper<float> breakdownWrap(d_breakdown);
  const mblas::MatrixWrapper<float> inWrap(d_in);

  //gGetValueByKey<<<1, lastN_, 0, stream_>>>
  //  (breakdownWrap, inWrap, h_res_idx, lastN_);

  HANDLE_ERROR( cudaMemcpyAsync(out.data(), d_breakdown.data(), h_res.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, mblas::CudaStreamHandler::GetStream()) );
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
}

}  // namespace GPU
} // namespace amunmt
