#pragma once

#include "graph/expression_graph.h"

namespace marian {

namespace cpu {

void suppressWords(Expr logProbs, Expr wordIndices);
}

namespace gpu {

void suppressWords(Expr logProbs, Expr wordIndices);
}

void suppressWords(Expr logProbs, Expr wordIndices);
}  // namespace marian
