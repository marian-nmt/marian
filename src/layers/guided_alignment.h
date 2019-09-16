#pragma once

#include "layers/loss.h"

namespace marian {

static inline RationalLoss guidedAlignmentCost(Ptr<ExpressionGraph> graph,
                                               Ptr<data::CorpusBatch> batch,
                                               Ptr<Options> options,
                                               Expr attention) {

  // @TODO: there should be positional masking here ... on the other hand, positions that are not 
  // in a sentence should always agree (both being 0). Lack of masking affects label count only which is 
  // probably negligible?

  // @TODO: change "cost" to "loss"
  std::string guidedLossType = options->get<std::string>("guided-alignment-cost");
  float guidedScalar = options->get<float>("guided-alignment-weight");
  
  float epsilon = 1e-6f;
  Expr alignment = constant_like(attention, inits::fromVector(batch->getGuidedAlignment()));
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

  // every position is a label as they should all agree, see caveat at the top.
  size_t numLabels = alignment->shape().elements();
  
  // Create label node, also weigh by scalar so labels and cost are in the same domain.
  // Fractional label counts are OK
  return RationalLoss(alignmentLoss, guidedScalar * numLabels);
}

}  // namespace marian
