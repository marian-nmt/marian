#include <iostream>
#include "common/utils.h"
#include "matrix_wrapper.h"
#include "nth_element.h"
#include "matrix_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

#define UNROLL_MAXARG_LOOP( n, max ) \
  if (tid < (n) && tid + (n) < ( max ) ) { \
    if (sdata[tid + ( n ) ] > sdata[tid]) { \
      sdata[tid] = sdata[tid + ( n ) ]; \
      indices[tid] = indices[tid + ( n ) ]; \
    } \
  }

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void gMaxElement(mblas::MatrixWrapper<float> outWrap,
                            mblas::MatrixWrapper<int> indWrap,
                            const mblas::MatrixWrapper<float> probsWrap,
                            const mblas::MatrixWrapper<int> batchPositionWrap,
                            int numBatches) {
  extern __shared__ float sdata[];
  __shared__ int indices[512];

  int tid = threadIdx.x;

  for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    int begin = batchPositionWrap[batchIdx];
    int end = batchPositionWrap[batchIdx + 1];

    int i = begin + blockIdx.x * (blockDim.x * 2) + tid;

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

    for (int s = (blockDim.x >> 1); s > 32; s >>= 1) {
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
      outWrap[blockIdx.x + batchIdx * gridDim.x] = sdata[0];
      indWrap[blockIdx.x + batchIdx * gridDim.x] = indices[0];
    }
    __syncthreads();
  }
}

__global__ void gMaxElementUpdate(mblas::MatrixWrapper<float> outWrap,
                                  mblas::MatrixWrapper<int> indWrap,
                                  mblas::MatrixWrapper<float> probsWrap,
                                  mblas::MatrixWrapper<int> batchPositionWrap,
                                  mblas::MatrixWrapper<float> resWrap,
                                  mblas::MatrixWrapper<int> res_idxWrap,
                                  mblas::MatrixWrapper<int> cumBeamSizesWrap,
                                  int numBlocks) {
  extern __shared__ float sdata[];
  __shared__ int indices[512];
  __shared__ float bestBinCost;
  __shared__ int bestBinCostIdx;

  const int tid = threadIdx.x;
  const int batchIdx = blockIdx.x;
  const int N = batchPositionWrap[batchIdx + 1] - batchPositionWrap[batchIdx];
  int num_bins = int(N / (2 * 512)) + int(N % (2 * 512) != 0);
  if (num_bins > 500) {
    num_bins = 500;
  }

  for (int pos = cumBeamSizesWrap[batchIdx]; pos < cumBeamSizesWrap[batchIdx + 1]; ++pos) {
    int i = tid;

    sdata[tid] = -3.40282e+38f;

    if (i < num_bins) {
      sdata[tid] = outWrap[batchIdx * numBlocks + i];
      indices[tid] = i;
    }

    if (i + blockDim.x < num_bins) {
      float a = outWrap[batchIdx * numBlocks + i];
      float b = outWrap[batchIdx * numBlocks + i + blockDim.x];
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

      float a = outWrap[batchIdx * numBlocks + i];
      if (a > sdata[tid]) {
        sdata[tid] = a;
        indices[tid] = i;
      }

      if (i + blockDim.x < num_bins) {
        float b = outWrap[batchIdx * numBlocks + i + blockDim.x];
        if (b > sdata[tid]) {
          sdata[tid] = b;
          indices[tid] = i + blockDim.x;
        }
      }
    }

    __syncthreads();

    for (int s = (blockDim.x >> 1); s > 32; s >>= 1) {
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

      probsWrap[indWrap[bestBinCostIdx]] = -3.40282e+38f;

      res_idxWrap[pos] = indWrap[bestBinCostIdx];
      resWrap[pos] = bestBinCost;
    }

    __syncthreads();

    i = batchPositionWrap[batchIdx] + (bestBinCostIdx - batchIdx * numBlocks) * (blockDim.x * 2) + tid;
    const int dist = num_bins * 2 * blockDim.x;

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

    for (int s = (blockDim.x >> 1); s > 32; s >>= 1) {
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
      outWrap[bestBinCostIdx] = sdata[0];
      indWrap[bestBinCostIdx] = indices[0];
    }
    __syncthreads();
  }
}

__global__ void gGetValueByKey(mblas::MatrixWrapper<float> out,
                              const   mblas::MatrixWrapper<float> in,
                              int* indices, int n)
{
  int tid = threadIdx.x  + blockDim.x * blockIdx.x;
  if (tid < n) {
    int index = indices[tid];
    out[tid] = in[index];
  }
}

NthElement::NthElement(size_t maxBeamSize, size_t maxBatchSize, cudaStream_t& stream)
: stream_(stream)
, numBlocks_(std::min(500, int(maxBeamSize * 85000 / (2 * BLOCK_SIZE)) + int(maxBeamSize * 85000 % (2 * BLOCK_SIZE) != 0)))
, d_out(maxBatchSize * numBlocks_)
, d_ind(maxBatchSize * numBlocks_)
, d_breakdown(maxBeamSize)
, maxBeamSize_(maxBeamSize)
, maxBatchSize_(maxBatchSize)
{
  cerr << "FOO1" << endl;
  cerr << "maxBatchSize=" << maxBatchSize << " maxBeamSize=" << maxBeamSize << endl;

  d_batchPosition.reserve(maxBatchSize + 1);
  d_cumBeamSizes.reserve(maxBatchSize + 1);

  d_res_idx.resize(maxBatchSize * maxBeamSize);
  d_res.resize(maxBatchSize * maxBeamSize);

  HANDLE_ERROR( cudaHostAlloc((void**) &h_res, maxBeamSize * maxBatchSize* sizeof(float),
                              cudaHostAllocDefault) );
  HANDLE_ERROR( cudaHostAlloc((void**) &h_res_idx, maxBeamSize * maxBatchSize * sizeof(int),
                              cudaHostAllocDefault) );
}

NthElement::~NthElement()
{
  cerr << "FOO2" << endl;
  HANDLE_ERROR(cudaFreeHost(h_res));
  HANDLE_ERROR(cudaFreeHost(h_res_idx));
}

void NthElement::getNBestList(mblas::Matrix &probs, const std::vector<int>& batchFirstElementIdxs,
                              const std::vector<int>& cummulatedBeamSizes)
{
  cerr << "FOO3" << endl;
  cerr << "cummulatedBeamSizes=" << cummulatedBeamSizes.size() << endl;
  d_batchPosition.resize(batchFirstElementIdxs.size());
  d_cumBeamSizes.resize(cummulatedBeamSizes.size());
  assert(d_batchPosition.size() == d_cumBeamSizes.size());

  HANDLE_ERROR( cudaMemcpyAsync(thrust::raw_pointer_cast(d_batchPosition.data()), batchFirstElementIdxs.data(), batchFirstElementIdxs.size() * sizeof(int),
                                cudaMemcpyHostToDevice, stream_) );
  HANDLE_ERROR( cudaMemcpyAsync(thrust::raw_pointer_cast(d_cumBeamSizes.data()), cummulatedBeamSizes.data(), cummulatedBeamSizes.size() * sizeof(int),
                                cudaMemcpyHostToDevice, stream_) );

  const int numBatches = batchFirstElementIdxs.size() - 1;

  mblas::MatrixWrapper<float> outWrap(d_out);
  mblas::MatrixWrapper<int> indWrap(d_ind);
  mblas::MatrixWrapper<float> probsWrap(probs);
  mblas::MatrixWrapper<int> batchPositionWrap(d_batchPosition);
  mblas::MatrixWrapper<float> resWrap(d_res);
  mblas::MatrixWrapper<int> res_idxWrap(d_res_idx);
  mblas::MatrixWrapper<int> cumBeamSizesWrap(d_cumBeamSizes);

  gMaxElement<<<numBlocks_, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream_>>>
    (outWrap, indWrap, probsWrap, batchPositionWrap, numBatches);

  gMaxElementUpdate<<<numBatches, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream_>>>
    (outWrap, indWrap, probsWrap, batchPositionWrap, resWrap, res_idxWrap, cumBeamSizesWrap,
     numBlocks_);

  cerr << "numBlocks_=" << numBlocks_ << endl;
  cerr << "numBatches=" << numBatches << endl;
  cerr << "threads=" << BLOCK_SIZE << endl;

  cerr << "outWrap=" << outWrap.Debug() << endl;
  //cerr << mblas::Debug(d_out, 2) << endl;

  cerr << "indWrap=" << indWrap.Debug() << endl;
  //cerr << mblas::Debug(d_ind, 2) << endl;

  cerr << "probsWrap=" << probsWrap.Debug() << endl;

  cerr << "batchPositionWrap=" << batchPositionWrap.Debug() << endl;
  cerr << mblas::Debug(d_batchPosition, 2) << endl;

  cerr << "resWrap=" << resWrap.Debug() << endl;
  cerr << mblas::Debug(d_res, 2) << endl;

  cerr << "res_idxWrap=" << res_idxWrap.Debug() << endl;
  cerr << mblas::Debug(d_res_idx, 2) << endl;

  cerr << "cumBeamSizesWrap=" << cumBeamSizesWrap.Debug() << endl;
  cerr << mblas::Debug(d_cumBeamSizes, 2) << endl;

  cerr << endl;

}

void NthElement::getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                  std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                  const bool isFirst) {
  cerr << "FOO4" << endl;
  cerr << "beamSizes=" << beamSizes.size() << endl;
  cerr << Debug(beamSizes, 2) << endl;
  cerr << "outCosts=" << outCosts.size() << endl;
  cerr << "outKeys=" << outKeys.size() << endl;

  std::vector<int> cummulatedBeamSizes(beamSizes.size() + 1);
  std::vector<int> batchFirstElementIdxs(beamSizes.size() + 1);
  cummulatedBeamSizes[0] = 0;
  batchFirstElementIdxs[0] = 0;

  const size_t vocabSize = Probs.dim(1);
  for (size_t i = 0; i < beamSizes.size(); ++i) {

    cummulatedBeamSizes[i + 1] = cummulatedBeamSizes[i] + beamSizes[i];
    batchFirstElementIdxs[i + 1] = ((isFirst) ? (i + 1) : cummulatedBeamSizes[i + 1]) * vocabSize;
  }

  size_t numHypos = cummulatedBeamSizes.back();
  d_res_idx.resize(numHypos);
  d_res.resize(numHypos);

  //cerr << endl;
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

void NthElement::GetPairs(size_t number,
                    std::vector<unsigned>& outKeys,
                    std::vector<float>& outValues) {
  cerr << "FOO5:" << number << endl;

  HANDLE_ERROR( cudaMemcpyAsync(h_res, thrust::raw_pointer_cast(d_res.data()), number * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_) );
  HANDLE_ERROR( cudaMemcpyAsync(h_res_idx, thrust::raw_pointer_cast(d_res_idx.data()), number * sizeof(int),
                                cudaMemcpyDeviceToHost, stream_) );
  HANDLE_ERROR( cudaStreamSynchronize(stream_) );

  for (size_t i = 0; i < number; ++i) {
    outKeys.push_back(h_res_idx[i]);
    outValues.push_back(h_res[i]);
  }

  lastN = number;
}

void NthElement::getValueByKey(std::vector<float>& out, const mblas::Matrix &d_in) const
{
  cerr << "FOO6" << endl;
  mblas::MatrixWrapper<float> breakdownWrap(d_breakdown);
  const mblas::MatrixWrapper<float> inWrap(d_in);

  gGetValueByKey<<<1, lastN, 0, stream_>>>
    (breakdownWrap, inWrap, h_res_idx, lastN);

  HANDLE_ERROR( cudaMemcpyAsync(out.data(), thrust::raw_pointer_cast(d_breakdown.data()), lastN * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_) );
  HANDLE_ERROR( cudaStreamSynchronize(stream_));
}

}  // namespace GPU
} // namespace amunmt
