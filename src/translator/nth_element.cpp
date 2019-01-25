/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "translator/nth_element.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>

namespace marian {

class NthElementCPU {
  std::vector<int> h_res_idx;
  std::vector<float> h_res;
  size_t lastN;

public:
  NthElementCPU() = delete;
  NthElementCPU(const NthElementCPU& copy) = delete;

  NthElementCPU(size_t maxBeamSize, size_t maxBatchSize) {
    size_t maxSize = maxBeamSize * maxBatchSize;
    h_res.resize(maxSize);
    h_res_idx.resize(maxSize);
  }

private:
  void getNBestList(float* scores,
                    const std::vector<int>& batchFirstElementIdxs,
                    const std::vector<int>& cumulativeBeamSizes) {
    /* For each batch, select the max N elements, where N is the beam size for
     * this batch. Locally record these elements (their current value and index
     * in 'scores') before updating each element to a large negative value, such
     * that they won't be a maximum if we're called again on the same input.
     */

    int numProbs = batchFirstElementIdxs.back();
    std::vector<int> idxs(numProbs);
    std::iota(idxs.begin(), idxs.end(), 0);

    size_t numBatches = batchFirstElementIdxs.size() - 1;
    for(size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
      int pos = cumulativeBeamSizes[batchIdx];
      int beamSize = cumulativeBeamSizes[batchIdx + 1] - pos;

      std::vector<int>::iterator begin = idxs.begin() + batchFirstElementIdxs[batchIdx];
      std::vector<int>::iterator middle = begin + beamSize;
      std::vector<int>::iterator end = idxs.begin() + batchFirstElementIdxs[batchIdx + 1];
      std::partial_sort(
          begin, middle, end, [=](int a, int b) { return scores[a] > scores[b]; });

      while(begin != middle) {
        int idx = *begin++;
        h_res_idx[pos] = idx;
        h_res[pos] = scores[idx];
        scores[idx] = std::numeric_limits<float>::lowest();
        ++pos;
      }
    }
  }

public:
  void getNBestList(const std::vector<size_t>& beamSizes,
                                   Tensor scores,
                                   std::vector<float>& outPathScores,
                                   std::vector<unsigned>& outKeys,
                                   const bool isFirst) {
    std::vector<int> cumulativeBeamSizes(beamSizes.size() + 1, 0);
    std::vector<int> batchFirstElementIdxs(beamSizes.size() + 1, 0);

    auto vocabSize = scores->shape()[-1];
    for(int i = 0; i < beamSizes.size(); ++i) {
      cumulativeBeamSizes[i + 1] = cumulativeBeamSizes[i] + (int)beamSizes[i];
      batchFirstElementIdxs[i + 1]
          += (isFirst ? i + 1 : cumulativeBeamSizes[i + 1]) * vocabSize;
    }

    getNBestList(scores->data(), batchFirstElementIdxs, cumulativeBeamSizes);
    getPairs(cumulativeBeamSizes.back(), outKeys, outPathScores);
  }

private:
  void getPairs(size_t number,
                std::vector<unsigned>& outKeys,
                std::vector<float>& outValues) {
    std::copy(h_res_idx.begin(), h_res_idx.begin() + number, std::back_inserter(outKeys));
    std::copy(h_res    .begin(), h_res    .begin() + number, std::back_inserter(outValues));
    lastN = number;
  }

  void getValueByKey(std::vector<float>& out, float* d_in) {
    for(size_t i = 0; i < lastN; ++i) {
      out[i] = d_in[h_res_idx[i]];
    }
  }
};

#ifdef CUDA_FOUND
GetNBestListFn createGetNBestListGPUFn(size_t beamSize, size_t dimBatch, DeviceId deviceId); // in .cu file
#endif

// factory function
// Returns a lambda with the same signature as the getNBestList() function.
GetNBestListFn createGetNBestListFn(size_t beamSize, size_t dimBatch, DeviceId deviceId) {
#ifdef CUDA_FOUND
  if(deviceId.type == DeviceType::gpu)
    return createGetNBestListGPUFn(beamSize, dimBatch, deviceId);
#else
    deviceId; // (unused)
#endif
  auto nth = New<NthElementCPU>(beamSize, dimBatch);
  return [nth](const std::vector<size_t>& beamSizes,
      Tensor logProbs,
      std::vector<float>& outCosts,
      std::vector<unsigned>& outKeys,
      const bool isFirst) {
      return nth->getNBestList(beamSizes, logProbs, outCosts, outKeys, isFirst);
  };
}

}  // namespace marian
