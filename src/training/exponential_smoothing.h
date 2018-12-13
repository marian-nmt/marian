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
      auto args = options->get<std::vector<float>>("exponential-smoothing");
      ABORT_IF(args.size() < 1 || args.size() > 2, "exponential-smoothing parameter must be one or two numbers");
      mvDecayBy_ = args[0];
      if (args.size() > 1)
        refBatchTrgWords_ = (size_t)args[1];
      mvAvg_ = (mvDecayBy_ > 0);
    }

protected:
  void updateAvgParams(Tensor paramsAvg, Tensor params, size_t batches, size_t actualBatchTrgWords = OptimizerBase::mbSizeNotProvided) {
    double beta = 1. - mvDecayBy_;
    // correction term if batch size is different from what mvDecayBy_ was specified for
    if (refBatchTrgWords_) {
      ABORT_IF(actualBatchTrgWords == OptimizerBase::mbSizeNotProvided,
               "This graph-group type does not support reference batch size specification for exponential-smoothing");
      beta = pow(beta, (double)actualBatchTrgWords / (double)refBatchTrgWords_);
    }
    // reduce effect of decay parameter in early training stages
    float decayBy = std::max(1.f - (float)beta,
                             1.f - (float)(batches + 1) / (float)(batches + 10));
    using namespace functional;
    Element(_1 = ((1.f - decayBy) * _1) + (decayBy * _2), paramsAvg, params);
  }

  bool mvAvg_{false};
  float mvDecayBy_{1e-4f};     // decay prior model by this factor
  size_t refBatchTrgWords_{0}; // mvDecayBy_ is specified for this batch size (in target words)
};
}  // namespace marian
