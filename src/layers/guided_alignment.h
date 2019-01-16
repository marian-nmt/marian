#pragma once

#include "layers/loss.h"

namespace marian {

static inline RationalLoss guidedAlignmentCost(Ptr<ExpressionGraph> graph,
                                               Ptr<data::CorpusBatch> batch,
                                               Ptr<Options> options,
                                               Expr attention) {

  // @TODO: change "cost" to "loss"
  std::string guidedLossType = options->get<std::string>("guided-alignment-cost");
  float guidedScalar = options->get<float>("guided-alignment-weight");
  
  float epsilon = 1e-6f;
  Expr alignment = constant_like(attention, inits::from_vector(batch->getGuidedAlignment()));
  Expr alignmentLoss; // sum up loss over all attention/alignment positions
  if(guidedLossType == "mse") {
    alignmentLoss = sum(flatten(square(attention - alignment))) / 2.f;
  } else if(guidedLossType == "mult") {
    alignmentLoss = -log(sum(flatten(attention * alignment)) + epsilon);
  } else if(guidedLossType == "ce") {
    alignmentLoss = -sum(flatten(alignment * log(attention + epsilon)));
  } else {
    ABORT("Unknown alignment cost type: {}", guidedLossType);
  }
  
  alignmentLoss = guidedScalar * alignmentLoss; // weigh by scalar

  // every position is a label as they should all agree
  float numLabels = alignment->shape().elements();
  
  // create label node
  Expr labels = graph->constant({1}, inits::from_value(numLabels));
  labels = guidedScalar * labels; // also weigh by scalar so labels and cost are in the same domain

  return RationalLoss(alignmentLoss, labels);
}

}  // namespace marian
