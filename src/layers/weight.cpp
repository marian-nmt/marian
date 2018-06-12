#include "layers/weight.h"

namespace marian {

Ptr<WeightingBase> WeightingFactory(Ptr<Options> options) {
  if(options->has("data-weighting"))
    return New<DataWeighting>(options->get<std::string>("data-weighting-type"));
}

Expr DataWeighting::getWeights(Ptr<ExpressionGraph> graph,
                               Ptr<data::CorpusBatch> batch) {
  ABORT_IF(batch->getDataWeights().empty(),
           "Vector of weights is unexpectedly empty!");
  bool sentenceWeighting = weightingType_ == "sentence";
  int dimBatch = batch->size();
  int dimWords = sentenceWeighting ? 1 : batch->back()->batchWidth();
  auto weights = graph->constant({1, dimWords, dimBatch, 1},
                                 inits::from_vector(batch->getDataWeights()));
  return weights;
}
}
