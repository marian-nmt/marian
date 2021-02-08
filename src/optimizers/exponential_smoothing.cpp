#include "common/definitions.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
#include "common/options.h"
#include "optimizers/optimizers.h"
#include "optimizers/exponential_smoothing.h"

namespace marian {

void ExponentialSmoothing::updateAvgParams(Tensor paramsAvg, Tensor params, size_t batches, size_t actualBatchTrgWords) {
  double beta = 1. - mvDecayBy_;

  // correction term if batch size is different from what mvDecayBy_ was specified for
  if (refBatchTrgWords_ > 0 && actualBatchTrgWords > refBatchTrgWords_) {
    LOG_ONCE(info, "Exponential smoothing gets automatically adjusted as if update size was {} target words", refBatchTrgWords_);
    beta = pow(beta, (double)actualBatchTrgWords / (double)refBatchTrgWords_);
    batches = (size_t)((double)batches * (double)actualBatchTrgWords / (double)refBatchTrgWords_);
  }

  // reduce effect of decay parameter in early training stages
  float decayBy = std::max(1.f - (float)beta,
                           1.f - (float)(batches + 1) / (float)(batches + 10));
  using namespace functional;
  Element(_1 = ((1.f - decayBy) * _1) + (decayBy * _2), paramsAvg, params);
}

}  // namespace marian
