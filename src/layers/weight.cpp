#include "layers/weight.h"

namespace marian {

Ptr<WeightingBase> WeightingFactory(Ptr<Options> options) {
  ABORT_IF(!options->hasAndNotEmpty("data-weighting"),
           "No data-weighting specified in options");
  return New<DataWeighting>(options->get<std::string>("data-weighting-type"));
}

Expr DataWeighting::getWeights(Ptr<ExpressionGraph> graph,
                               Ptr<data::CorpusBatch> batch) {
  ABORT_IF(batch->getDataWeights().empty(),
           "Vector of weights is unexpectedly empty!");
  bool sentenceWeighting = weightingType_ == "sentence";
  int dimBatch = (int)batch->size();
  int dimWords = sentenceWeighting ? 1 : (int)batch->back()->batchWidth();

  // This would abort anyway in fromVector(...), but has clearer error message
  // here for this particular case
  ABORT_IF(batch->getDataWeights().size() != dimWords * dimBatch, 
           "Number of sentence/word-level weights ({}) does not match tensor size ({})",
           batch->getDataWeights().size(), dimWords * dimBatch);

  auto weights = graph->constant({1, dimWords, dimBatch, 1},
                                 inits::fromVector(batch->getDataWeights()));
  return weights; // [1, dimWords, dimBatch, 1] in case of word-level weights or
                  // [1,        1, dimBatch, 1] in case of sentence-level weights
}
}  // namespace marian
