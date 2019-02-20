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
  //size_t lastN_;

public:
  NthElementCPU() {}
  NthElementCPU(const NthElementCPU& copy) = delete;

private:
  // for each batch, select the max N elements, where N is the beam size for this batch.
  void selectNBest(const float* scores,
                   const std::vector<int>& batchFirstElementIdxs,
                   const std::vector<int>& cumulativeBeamSizes) {
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
          begin, middle, end, [&](int a, int b) { return scores[a] > scores[b]; });

      while(begin != middle) {
        int idx = *begin++;
        h_res_idx[pos] = idx;
        h_res[pos] = scores[idx];
        ++pos;
      }
    }
  }

public:
  void getNBestList(Tensor scores, // [dimBatch, 1, beamSize, dimVocab or dimShortlist]
                    size_t N,
                    std::vector<float>& outPathScores,
                    std::vector<unsigned>& outKeys,
                    const bool isFirst) {
    const auto vocabSize = scores->shape()[-1];
    const auto inputN    = scores->shape()[-2];
    const auto dimBatch  = scores->shape()[-4];
    ABORT_IF(inputN != (isFirst ? 1 : N), "Input tensor has wrong beam dim??"); // @TODO: Remove isFirst argument altogether

    std::vector<int> cumulativeBeamSizes(dimBatch + 1, 0);
    std::vector<int> batchFirstElementIdxs(dimBatch + 1, 0);

    for(int batchIdx = 0; batchIdx < dimBatch; ++batchIdx) {
      cumulativeBeamSizes[batchIdx + 1] = (batchIdx + 1) * (int)N;
      batchFirstElementIdxs[batchIdx + 1] += (batchIdx + 1) * inputN * vocabSize;
      ABORT_IF(cumulativeBeamSizes[batchIdx + 1] != cumulativeBeamSizes[batchIdx] + (int)N, "cumulativeBeamSizes wrong??");
      ABORT_IF((isFirst ? batchIdx + 1 : cumulativeBeamSizes[batchIdx + 1]) != (batchIdx + 1) * inputN, "inputN wrong??");
    }
    ABORT_IF(cumulativeBeamSizes.back() != dimBatch * N, "cumulativeBeamSizes.back() wrong??");

    size_t maxSize = N * dimBatch;
    h_res.resize(maxSize);
    h_res_idx.resize(maxSize);

    selectNBest(scores->data(), batchFirstElementIdxs, cumulativeBeamSizes);
    getPairs(/*cumulativeBeamSizes.back(),*/ outKeys, outPathScores);
  }

private:
  void getPairs(/*size_t number,*/
                std::vector<unsigned>& outKeys,
                std::vector<float>& outValues) {
    std::copy(h_res_idx.begin(), h_res_idx.end(), std::back_inserter(outKeys));
    std::copy(h_res    .begin(), h_res    .end(), std::back_inserter(outValues));
    //lastN_ = number;
  }

  //void getValueByKey(std::vector<float>& out, float* d_in) {
  //  for(size_t i = 0; i < lastN_; ++i) {
  //    out[i] = d_in[h_res_idx[i]];
  //  }
  //}
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
  deviceId; beamSize; dimBatch; // (unused)
#endif
  auto nth = New<NthElementCPU>();
  return [nth](Tensor logProbs, size_t N, std::vector<float>& outCosts, std::vector<unsigned>& outKeys, const bool isFirst) {
    return nth->getNBestList(logProbs, N, outCosts, outKeys, isFirst);
  };
}

}  // namespace marian
