#include "layers/generic.h"

namespace marian {

Expr Cost(Expr logits,
          Expr indices,
          Expr mask,
          std::string costType,
          float smoothing) {
  using namespace keywords;

  auto ce = cross_entropy(logits, indices);

  if(smoothing > 0) {
    // @TODO: add this to CE kernels instead
    auto ceq = mean(logsoftmax(logits), axis = 1);
    ce = (1 - smoothing) * ce - smoothing * ceq;
  }

  if(mask)
    ce = ce * mask;

  Expr cost;
  if(costType == "ce-mean" || costType == "cross-entropy") {
    cost = mean(sum(ce, axis = 2), axis = 0);
  } else if(costType == "ce-mean-words") {
    cost
        = sum(sum(ce, axis = 2), axis = 0) / sum(sum(mask, axis = 2), axis = 0);
  } else if(costType == "ce-sum") {
    cost = sum(sum(ce, axis = 2), axis = 0);
  } else if(costType == "perplexity") {
    cost = exp(sum(sum(ce, axis = 2), axis = 0)
               / sum(sum(mask, axis = 2), axis = 0));
  } else if(costType == "ce-rescore") {
    cost = -sum(ce, axis = 2);
  } else {  // same as ce-mean
    cost = mean(sum(ce, axis = 2), axis = 0);
  }

  return cost;
}
}
