#pragma once

#include "layers/loss.h"
#include "common/logging.h"

namespace marian {

static inline RationalLoss guidedAlignmentCost(Ptr<ExpressionGraph> /*graph*/,
                                               Ptr<data::CorpusBatch> batch,
                                               Ptr<Options> options,
                                               Expr attention) { // [beam depth=1, max src length, batch size, tgt length]

  std::string guidedLossType = options->get<std::string>("guided-alignment-cost");  // @TODO: change "cost" to "loss"
  float guidedLossWeight = options->get<float>("guided-alignment-weight");

  const auto& shape = attention->shape(); // [beam depth=1, max src length, batch size, tgt length]
  float epsilon = 1e-6f;
  Expr alignmentLoss; // sum up loss over all attention/alignment positions
  size_t numLabels;
  if(guidedLossType == "ce") {
    // normalizedAlignment is multi-hot, but ce requires normalized probabilities, so need to normalize to P(s|t)
    auto dimBatch    = shape[-2];
    auto dimTrgWords = shape[-1];
    auto dimSrcWords = shape[-3];
    ABORT_IF(shape[-4] != 1, "Guided alignments with beam??");
    auto normalizedAlignment = batch->getGuidedAlignment(); // [dimSrcWords, dimBatch, dimTrgWords] flattened, matches shape of 'attention'
    auto srcBatch = batch->front();
    const auto& srcMask = srcBatch->mask();
    ABORT_IF(shape.elements() != normalizedAlignment.size(), "Attention-matrix and alignment shapes differ??");
    ABORT_IF(dimBatch != batch->size() || dimTrgWords != batch->widthTrg() || dimSrcWords != batch->width(), "Attention-matrix and batch shapes differ??");
    auto locate = [=](size_t s, size_t b, size_t t) { return ((s * dimBatch) + b) * dimTrgWords + t; };
    for (size_t b = 0; b < dimBatch; b++) {
      for (size_t t = 0; t < dimTrgWords; t++) {
        for (size_t s = 0; s < dimSrcWords; s++)
          ABORT_IF(locate(s, b, t) != batch->locateInGuidedAlignments(b, s, t), "locate() and locateInGuidedAlignments() differ??");
        // renormalize the alignment such that it sums up to 1
        float sum = 0;
        for (size_t s = 0; s < dimSrcWords; s++)
          sum += srcMask[srcBatch->locate(b, s)] * normalizedAlignment[locate(s, b, t)]; // these values are 0 or 1
        if (sum != 0 && sum != 1)
          for (size_t s = 0; s < dimSrcWords; s++)
            normalizedAlignment[locate(s, b, t)] /= sum;
      }
    }
    auto alignment = constant_like(attention, std::move(normalizedAlignment));
    alignmentLoss = -sum(flatten(alignment * log(attention + epsilon)));
    numLabels = batch->back()->batchWords();
    ABORT_IF(numLabels > shape.elements() / shape[-3], "Num labels of guided alignment cost is off??");
  } else {
    auto alignment = constant_like(attention, batch->getGuidedAlignment());
    if(guidedLossType == "mse")
      alignmentLoss = sum(flatten(square(attention - alignment))) / 2.f;
    else if(guidedLossType == "mult") // @TODO: I don't know what this criterion is for. Can we remove it?
      alignmentLoss = -log(sum(flatten(attention * alignment)) + epsilon);
    else
       ABORT("Unknown alignment cost type: {}", guidedLossType);
    // every position is a label as they should all agree
    // @TODO: there should be positional masking here ... on the other hand, positions that are not
    // in a sentence should always agree (both being 0). Lack of masking affects label count only which is
    // probably negligible?
    numLabels = shape.elements();
  }

  // Create label node, also weigh by scalar so labels and cost are in the same domain.
  // Fractional label counts are OK. But only if combined as "sum".
  // @TODO: It is ugly to check the multi-loss type here, but doing this right requires
  // a substantial rewrite of the multi-loss architecture, which is planned anyways.
  std::string multiLossType = options->get<std::string>("multi-loss-type", "sum");
  if (multiLossType == "sum")         // sum of sums
    return RationalLoss(guidedLossWeight * alignmentLoss, guidedLossWeight * numLabels);
  else
    return RationalLoss(guidedLossWeight * alignmentLoss, (float)numLabels);
}

}  // namespace marian
