#include "layers/loss.h"

namespace marian {

Ptr<LabelwiseLoss> newLoss(Ptr<Options> options, bool inference) {
  float smoothing = inference ? 0.f : options->get<float>("label-smoothing");
  std::string costType = options->get<std::string>("cost-type", "ce-mean");
  if(costType == "ce-mean" || costType == "cross-entropy") {
    return New<CrossEntropyLoss>(smoothing);
  } else if(costType == "ce-mean-words") {
    return New<CrossEntropyLoss>(smoothing);
  } else if(costType == "ce-sum") {
    return New<CrossEntropyLoss>(smoothing);
  } else if(costType == "perplexity") {
    return New<CrossEntropyLoss>(smoothing);
  } else if(costType == "ce-rescore") {
    return New<RescorerLoss>();
  } else if(costType == "ce-rescore-mean") {
    return New<RescorerLoss>();
  } else {  // same as ce-mean
    return New<CrossEntropyLoss>(smoothing);
  }
}

Ptr<MultiRationalLoss> newMultiLoss(Ptr<Options> options) {
    std::string multiLossType = options->get<std::string>("multi-loss-type", "sum");
    if(multiLossType == "sum")
      return New<SumMultiRationalLoss>();
    else if(multiLossType == "scaled")
      return New<ScaledMultiRationalLoss>();
    else if(multiLossType == "normal")
      return New<MeanMultiRationalLoss>();
    else
      ABORT("Unknown multi-loss-type {}", multiLossType);

    return nullptr;
}

}  // namespace marian
