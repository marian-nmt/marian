#pragma once

#include "common/options.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "graph/node_initializers.h"

namespace marian {

class WeightingBase {
public:
  WeightingBase(){};
  virtual Expr getWeights(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch)
      = 0;
  virtual void debugWeighting(std::vector<float> weightedMask,
                              std::vector<float> freqMask,
                              Ptr<data::CorpusBatch> batch){};
};

class DataWeighting : public WeightingBase {
protected:
  std::string weightingType_;

public:
  DataWeighting(std::string weightingType)
      : WeightingBase(), weightingType_(weightingType){};
  Expr getWeights(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch);
};

Ptr<WeightingBase> WeightingFactory(Ptr<Options> options);
}
