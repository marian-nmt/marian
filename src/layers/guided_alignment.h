#pragma once


#include "data/corpus.h"
#include "common/options.h"
#include "graph/expression_graph.h"

namespace marian {

Expr guidedAlignmentCost(Ptr<ExpressionGraph> graph,
                         Ptr<data::CorpusBatch> batch,
                         Ptr<Options> options,
                         Expr att);

}
