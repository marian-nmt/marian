/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <vector>

#include "tensors/tensor.h"

namespace marian {

struct NthElement {
  virtual ~NthElement() {}

  virtual void getNBestList(float* probs,
                            const std::vector<int>& batchFirstElementIdxs,
                            const std::vector<int>& cummulatedBeamSizes)
      = 0;

  virtual void getNBestList(const std::vector<size_t>& beamSizes,
                            Tensor Probs,
                            std::vector<float>& outCosts,
                            std::vector<unsigned>& outKeys,
                            const bool isFirst = false)
      = 0;

  virtual void GetPairs(size_t number,
                        std::vector<unsigned>& outKeys,
                        std::vector<float>& outValues)
      = 0;

  virtual void getValueByKey(std::vector<float>& out, float* d_in) = 0;
};

class NthElementCPU : public NthElement {
  std::vector<int> h_res_idx;
  std::vector<float> h_res;
  size_t lastN;

public:
  NthElementCPU() = delete;
  NthElementCPU(const NthElementCPU& copy) = delete;
  NthElementCPU(size_t maxBeamSize, size_t maxBatchSize);

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
};

class NthElementGPU : public NthElement {
public:
  NthElementGPU() = delete;
  NthElementGPU(const NthElementGPU& copy) = delete;
  NthElementGPU(size_t maxBeamSize, size_t maxBatchSize, DeviceId deviceId);
  ~NthElementGPU();

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
  DeviceId deviceId_;

  const int MAX_VOCAB_SIZE = 100000;

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
