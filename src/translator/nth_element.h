#pragma once

#include <algorithm>
#include <vector>

#include <cuda.h>
#include "tensors/tensor.h"

namespace marian {

class NthElement {
public:
  NthElement() = delete;
  NthElement(const NthElement& copy) = delete;
  NthElement(size_t maxBeamSize, size_t maxBatchSize /*, cudaStream_t stream*/);
  virtual ~NthElement();

  void getNBestList(float* probs,
                    const std::vector<int>& batchFirstElementIdxs,
                    const std::vector<int>& cummulatedBeamSizes);

  void getNBestList(const std::vector<size_t>& beamSizes,
                    Tensor Probs,
                    std::vector<float>& outCosts,
                    std::vector<unsigned>& outKeys,
                    const bool isFirst = false);

  void GetPairs(size_t number,
                std::vector<unsigned>& outKeys,
                std::vector<float>& outValues);

  void getValueByKey(std::vector<float>& out, float* d_in);

private:
  const int BLOCK_SIZE = 512;
  const int NUM_BLOCKS;
  // cudaStream_t stream_;
  int* d_ind;

  float* d_out;

  int* d_res_idx;
  float* d_res;

  int* h_res_idx;
  float* h_res;

  float* d_breakdown;
  int* d_batchPosition;
  int* d_cumBeamSizes;
  size_t lastN;
};
}
