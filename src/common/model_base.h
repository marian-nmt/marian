#pragma once

#include <string>

#include "graph/expression_graph.h"
#include "data/batch.h"

namespace marian {
namespace models {

class ModelBase {
public:
  virtual void load(Ptr<ExpressionGraph>, const std::string&) = 0;

  virtual void save(Ptr<ExpressionGraph>, const std::string&) = 0;

  virtual void save(Ptr<ExpressionGraph>, const std::string&, bool) = 0;

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) = 0;

  virtual Expr buildToScore(Ptr<ExpressionGraph> graph,
                            Ptr<data::Batch> batch,
                            bool clearGraph = true) {
    return build(graph, batch, clearGraph);
  }

  virtual Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph) = 0;
};

}
}
