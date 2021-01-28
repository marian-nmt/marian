#pragma once

#include "common/definitions.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
#include "common/options.h"

namespace marian {

/**
 * Class implementing exponential smoothing for graph groups.
 * The smoothed parameters themselves are not stored in here.
 */
class ExponentialSmoothing {
public:
    ExponentialSmoothing(Ptr<Options> options) {
      mvDecayBy_ = options->get<float>("exponential-smoothing", 0);
      refBatchTrgWords_ = options->get<size_t>("mini-batch-words-ref", 0); // adjust as if our MB size (in target labels) was this value
      mvAvg_ = (mvDecayBy_ > 0);
    }

protected:
  void updateAvgParams(Tensor paramsAvg, Tensor params, size_t batches, size_t actualBatchTrgWords);

  bool mvAvg_{false};
  float mvDecayBy_{1e-4f};     // decay prior model by this factor
  size_t refBatchTrgWords_{0}; // mvDecayBy_ is specified for this batch size (in target words) (0 means not specified)
};
}  // namespace marian
