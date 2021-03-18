#include "layers/loss.h"

namespace marian {

// @TODO, simplify this. Currently here for back-compat
Ptr<LabelwiseLoss> newLoss(Ptr<Options> options, bool inference) {
  float smoothing = inference ? 0.f : options->get<float>("label-smoothing");
  float factorWeight = options->get<float>("factor-weight", 1.0f);
  std::string costType = options->get<std::string>("cost-type", "ce-mean");
  bool unlikelihood = options->get<bool>("unlikelihood-loss", false);

  if(costType == "ce-rescore") {  // per-batch-item scores (while ce-mean reduces over batch)
    bool wordScores = options->get<bool>("word-scores", false);
    return New<RescorerLoss>(wordScores);
  } else if(unlikelihood) {
    ABORT_IF(
        !options->hasAndNotEmpty("data-weighting")
            && options->get<std::string>("data-weighting-type") != "word",
        "Unlikelihood loss training requires error annotation in form of per-target-label scores");
    return New<SequenceUnlikelihoodLoss>(
        smoothing, factorWeight);  // this is a mix of CE-loss and unlikelihood less depending on
                                   // values given for data-weighting
  } else {  // same as ce-mean  --@TODO: better check all allowed values, and fail for invalid ones.
            // E.g. what about ce-sum?
    return New<CrossEntropyLoss>(smoothing, factorWeight);
  }
}

// see loss.h for detailed explanations of each class
Ptr<MultiRationalLoss> newMultiLoss(Ptr<Options> options) {
  std::string multiLossType = options->get<std::string>("multi-loss-type", "sum");
  if(multiLossType == "sum")  // sum of sums
    return New<SumMultiRationalLoss>();
  else if(multiLossType == "scaled")  // sum of scaled sums, first element is reference scale
    return New<ScaledMultiRationalLoss>();
  else if(multiLossType == "mean")  // sum of means
    return New<MeanMultiRationalLoss>();
  else
    ABORT("Unknown multi-loss-type {}", multiLossType);
}

}  // namespace marian
