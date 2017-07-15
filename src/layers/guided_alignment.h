#include "data/batch_generator.h"
#include "data/corpus.h"

#include "common/config.h"
#include "graph/expression_graph.h"
#include "layers/generic.h"
#include "layers/param_initializers.h"

namespace marian {

Expr guidedAlignmentCost(Ptr<ExpressionGraph> graph,
                         Ptr<data::CorpusBatch> batch,
                         Ptr<Config> options,
                         Expr att) {
  using namespace keywords;

  int dimBatch = att->shape()[0];
  int dimSrc = att->shape()[2];
  int dimTrg = att->shape()[3];

  auto aln = graph->constant(
      {dimBatch, 1, dimSrc, dimTrg},
      keywords::init = inits::from_vector(batch->getGuidedAlignment()));

  std::string guidedCostType
      = options->get<std::string>("guided-alignment-cost");

  Expr alnCost;
  float eps = 1e-6;
  if(guidedCostType == "mse") {
    alnCost = sum(flatten(square(att - aln))) / (2 * dimBatch);
  } else if(guidedCostType == "mult") {
    alnCost = -log(sum(flatten(att * aln)) + eps) / dimBatch;
  } else if(guidedCostType == "ce") {
    alnCost = -sum(flatten(aln * log(att + eps))) / dimBatch;
  } else {
    UTIL_THROW2("Unknown alignment cost type");
  }

  float guidedScalar = options->get<float>("guided-alignment-weight");
  return guidedScalar * alnCost;
}
}
