/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <vector>

#include "tensors/tensor.h"

namespace marian {

typedef std::function<void(const std::vector<size_t>& beamSizes,
                           Tensor logProbs,
                           std::vector<float>& outCosts,
                           std::vector<unsigned>& outKeys,
                           const bool isFirst)> GetNBestListFn;

GetNBestListFn createGetNBestListFn(size_t beamSize, size_t dimBatch, DeviceId deviceId);

struct /*interface*/ INthElement {
  virtual ~INthElement() {}
  virtual void getNBestList(const std::vector<size_t>& beamSizes,
                            Tensor logProbs,
                            std::vector<float>& outCosts,
                            std::vector<unsigned>& outKeys,
                            const bool isFirst)
      = 0;
};
}  // namespace marian
