#include "layers/weight.h"

namespace marian {

Ptr<WeightingBase> WeightingFactory(Ptr<Options> options) {
  ABORT_IF(!options->has("data-weighting"),
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
  auto weights = graph->constant({1, dimWords, dimBatch, 1},
                                 inits::from_vector(batch->getDataWeights()));
  return weights;
}
}  // namespace marian
