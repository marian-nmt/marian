#pragma once

#include "common/definitions.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
#include "optimizers/optimizers.h"

namespace marian {

/**
 * Class implementing exponential smoothing for graph groups.
 * The smoothed parameters themselves are not stored in here.
 */
class ExponentialSmoothing {
public:
    ExponentialSmoothing(Ptr<Options> options) {
      mvDecayBy_ = options->get<float>("exponential-smoothing");
      refBatchTrgWords_ = options->get<size_t>("mini-batch-words-ref"); // adjust as if our MB size (in target labels) was this value
      mvAvg_ = (mvDecayBy_ > 0);
    }

protected:
  void updateAvgParams(Tensor paramsAvg, Tensor params, size_t batches, size_t actualBatchTrgWords = OptimizerBase::mbSizeNotProvided) {
    double beta = 1. - mvDecayBy_;
    // correction term if batch size is different from what mvDecayBy_ was specified for
    if (refBatchTrgWords_) {
      LOG_ONCE(info, "Exponential smoothing gets automatically adjusted as if update size was {} target words", refBatchTrgWords_);
      ABORT_IF(actualBatchTrgWords == OptimizerBase::mbSizeNotProvided,
               "This graph-group type does not support reference batch size specification for exponential-smoothing");
      beta = pow(beta, (double)actualBatchTrgWords / (double)refBatchTrgWords_);
      // If actual size differs from reference, then try to estimate the equivalent number of batches.
      // E.g. if MB size is growing over time, then this is an overestimate, which would diminish the
      // effect overly quickly, but in a range where that should be OK.
      batches = std::max(batches, batches * actualBatchTrgWords / refBatchTrgWords_); // @BUGBUG: Does not consider that batch size is changing
    }
    // reduce effect of decay parameter in early training stages
    float decayBy = std::max(1.f - (float)beta,
                             1.f - (float)(batches + 1) / (float)(batches + 10));
    using namespace functional;
    Element(_1 = ((1.f - decayBy) * _1) + (decayBy * _2), paramsAvg, params);
  }

  bool mvAvg_{false};
  float mvDecayBy_{1e-4f};     // decay prior model by this factor
  size_t refBatchTrgWords_{0}; // mvDecayBy_ is specified for this batch size (in target words) (0 means not specified)
};
}  // namespace marian
