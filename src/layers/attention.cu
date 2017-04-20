#include "attention.h"

namespace marian {

Expr attOps(Expr va, Expr context, Expr state, Expr coverage) {
  std::vector<Expr> nodes{va, context, state};
  if(coverage)
    nodes.push_back(coverage);

  int dimBatch = context->shape()[0];
  int dimWords = context->shape()[2];
  int dimBeam  = state->shape()[3];
  return reshape(Expression<AttentionNodeOp>(nodes),
                 {dimWords, dimBatch, 1, dimBeam});
}

}