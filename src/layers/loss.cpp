#include "layers/loss.h"

namespace marian {

Ptr<LossBase> LossFactory(Ptr<Options> options, bool inference) {
  std::string costType = options->get<std::string>("cost-type", "ce-mean");
  float smoothing = inference ? 0.f : options->get<float>("label-smoothing");
  if(costType == "ce-mean" || costType == "cross-entropy") {
    return New<CrossEntropyMeanLoss>(smoothing);
  } else if(costType == "ce-mean-words") {
    return New<CrossEntropyMeanWordsLoss>(smoothing);
  } else if(costType == "ce-sum") {
    return New<CrossEntropySumLoss>(smoothing);
  } else if(costType == "perplexity") {
    return New<PerplexityLoss>(smoothing);
  } else if(costType == "ce-rescore") {
    return New<CrossEntropyRescoreLoss>(smoothing);
  } else {  // same as ce-mean
    return New<CrossEntropyMeanLoss>(smoothing);
  }
}

Expr LossBase::getCrossEntropy(Expr logits,
                               Expr indices,
                               Expr mask,
                               Expr weights) {
  using namespace keywords;

  auto ce = cross_entropy(logits, indices);

  if(smoothing_ > 0) {
    // @TODO: add this to CE kernels instead
    auto ceq = mean(logsoftmax(logits), axis = -1);
    ce = (1 - smoothing_) * ce - smoothing_ * ceq;
  }

  if(mask)
    ce = ce * mask;

  if(weights)
    ce = ce * weights;

  return ce;
}

// Implement individual loss functions...
// axes:
//  - time axis (words): -3
//  - batch axis (sentences): -2
Expr CrossEntropyMeanLoss::getCost(Expr logits,
                                   Expr indices,
                                   Expr mask,
                                   Expr weights) {
  using namespace keywords;

  auto ce = getCrossEntropy(logits, indices, mask, weights);
  auto cost = mean(sum(ce, axis = -3), axis = -2);

  return cost;
}

Expr CrossEntropyMeanWordsLoss::getCost(Expr logits,
                                        Expr indices,
                                        Expr mask,
                                        Expr weights) {
  using namespace keywords;

  auto ce = getCrossEntropy(logits, indices, mask, weights);
  auto cost = sum(sum(ce, axis = -3), axis = -2)
              / sum(sum(mask, axis = -3), axis = -2);

  return cost;
}

Expr CrossEntropySumLoss::getCost(Expr logits,
                                  Expr indices,
                                  Expr mask,
                                  Expr weights) {
  using namespace keywords;

  auto ce = getCrossEntropy(logits, indices, mask, weights);
  auto cost = sum(sum(ce, axis = -3), axis = -2);

  return cost;
}

Expr PerplexityLoss::getCost(Expr logits,
                             Expr indices,
                             Expr mask,
                             Expr weights) {
  using namespace keywords;

  auto ce = getCrossEntropy(logits, indices, mask, weights);
  auto cost = exp(sum(sum(ce, axis = -3), axis = -2)
                  / sum(sum(mask, axis = -3), axis = -2));

  return cost;
}

Expr CrossEntropyRescoreLoss::getCost(Expr logits,
                                      Expr indices,
                                      Expr mask,
                                      Expr weights) {
  using namespace keywords;

  auto ce = getCrossEntropy(logits, indices, mask, weights);
  auto cost = -sum(ce, axis = -3);

  return cost;
}
}
