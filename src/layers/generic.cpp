#include "layers/generic.h"

namespace marian {

Expr Cost(Expr logits,
          Expr indices,
          Expr mask,
          std::string costType,
          float smoothing,
          Expr weights) {
  using namespace keywords;

  auto ce = cross_entropy(logits, indices);

  if(weights)
    ce = weights * ce;
  
  if(smoothing > 0) {
    // @TODO: add this to CE kernels instead
    auto ceq = mean(logsoftmax(logits), axis = -1);
    ce = (1 - smoothing) * ce - smoothing * ceq;
  }

  if(mask)
    ce = ce * mask;

  auto costSum = sum(ce, axis = -3);
  
  Expr cost;
  if(costType == "ce-mean" || costType == "cross-entropy") {
    cost = mean(costSum, axis = -2);
  } else if(costType == "ce-mean-words") {
    cost = sum(costSum, axis = -2) / sum(sum(mask, axis = -3), axis = -2);
  } else if(costType == "ce-sum") {
    cost = sum(costSum, axis = -2);
  } else if(costType == "perplexity") {
    cost = exp(sum(costSum, axis = -2) / sum(sum(mask, axis = -3), axis = -2));
  } else if(costType == "ce-rescore") {
    cost = -costSum;
  } else {  // same as ce-mean
    cost = mean(costSum, axis = -2);
  }

  return cost;
}
}
