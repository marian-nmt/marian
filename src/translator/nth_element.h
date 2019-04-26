/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "tensors/tensor.h"
#include <vector>

namespace marian {

typedef std::function<void(Tensor logProbs,
                           size_t N,
                           std::vector<float>& outCosts,
                           std::vector<unsigned>& outKeys,
                           const bool isFirst)> GetNBestListFn;

GetNBestListFn createGetNBestListFn(size_t beamSize, size_t dimBatch, DeviceId deviceId);
}  // namespace marian
