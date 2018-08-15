#pragma once

#include "common/definitions.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {

/**
 * Class implementing exponential smoothing for graph groups.
 */
class ExponentialSmoothing {
public:
  ExponentialSmoothing(float decay = 0.0f)
      : mvAvg_{decay > 0}, mvDecay_{decay} {}

protected:
  void updateAvgParams(Tensor paramsAvg, Tensor params, size_t batches) {
    using namespace functional;
    float decay = std::max(mvDecay_,
                           1.f - (float)(batches + 1) / (float)(batches + 10));
    Element(_1 = ((1.f - decay) * _1) + (decay * _2), paramsAvg, params);
  }

  bool mvAvg_{false};
  float mvDecay_{1e-4f};
};
}  // namespace marian
