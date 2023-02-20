#pragma once

#include "layers/loss.h"
#include "common/logging.h"

namespace marian {

static inline const std::tuple<std::vector<IndexType>, std::vector<float>> 
guidedAlignmentToSparse(Ptr<data::CorpusBatch> batch) {
  int trgWords = (int)batch->back()->batchWidth();  
  int dimBatch = (int)batch->size();

  typedef std::tuple<size_t, float> BiPoint;
  std::vector<BiPoint> byIndex;
  byIndex.reserve(dimBatch * trgWords);

  for(size_t b = 0; b < dimBatch; ++b) {
    auto guidedAlignmentFwd = batch->getGuidedAlignment()[b]; // this copies
    guidedAlignmentFwd.normalize(/*reverse=*/false); // normalize forward
    for(size_t i = 0; i < guidedAlignmentFwd.size(); ++i) {
      auto pFwd = guidedAlignmentFwd[i];
      IndexType idx = (IndexType)(pFwd.srcPos * dimBatch * trgWords + b * trgWords + pFwd.tgtPos);
      byIndex.push_back({idx, pFwd.prob});
    }
  }

  std::sort(byIndex.begin(), byIndex.end(), [](const BiPoint& a, const BiPoint& b) { return std::get<0>(a) < std::get<0>(b); });
  std::vector<IndexType> indices; std::vector<float> valuesFwd; 
  indices.reserve(byIndex.size()); 
  valuesFwd.reserve(byIndex.size()); 
  for(auto& p : byIndex) {
    indices.push_back((IndexType)std::get<0>(p));
    valuesFwd.push_back(std::get<1>(p));
  }

  return {indices, valuesFwd};
}

static inline RationalLoss guidedAlignmentCost(Ptr<ExpressionGraph> graph,
                                               Ptr<data::CorpusBatch> batch,
                                               Ptr<Options> options,
                                               Expr attention) { // [beam depth=1, max src length, batch size, tgt length]
  std::string guidedLossType = options->get<std::string>("guided-alignment-cost");  // @TODO: change "cost" to "loss"
  // @TODO: It is ugly to check the multi-loss type here, but doing this right requires
  // a substantial rewrite of the multi-loss architecture, which is planned anyways.
  std::string multiLossType = options->get<std::string>("multi-loss-type", "sum");
  
  // We dropped support for other losses which are not possible to implement with sparse labels.
  // They were most likely not used anyway.
  ABORT_IF(guidedLossType != "ce", "Only alignment loss type 'ce' is supported");

  float guidedLossWeight = options->get<float>("guided-alignment-weight");
  const auto& [indices, values] = guidedAlignmentToSparse(batch);
  
  Expr alignmentLoss;
  size_t numLabels = indices.size(); // can be zero
  if(indices.empty()) {
    removeAsRoot(stopGradient(attention)); // unused, hence make sure we don't polute the backwards operations
    alignmentLoss = graph->zeros({1});
    numLabels = multiLossType == "sum" ? 0 : 1;
  } else {
    float epsilon           = 1e-6f;
    auto alignmentIndices   = graph->indices(indices);
    auto alignmentValues    = graph->constant({(int)values.size()}, inits::fromVector(values));
    auto attentionAtAligned = cols(flatten(attention), alignmentIndices);
    alignmentLoss           = -sum(cast(alignmentValues * log(attentionAtAligned + epsilon), Type::float32));
  }
  // Create label node, also weigh by scalar so labels and cost are in the same domain.
  // Fractional label counts are OK. But only if combined as "sum".
  if (multiLossType == "sum") // sum of sums
    return RationalLoss(guidedLossWeight * alignmentLoss, guidedLossWeight * numLabels);
  else
    return RationalLoss(guidedLossWeight * alignmentLoss, (float)numLabels);
}

}  // namespace marian
