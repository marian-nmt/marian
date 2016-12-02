#include "node_operators_unary.h"
#include "expression_graph.h"

namespace marian {
  void UnaryNodeOp::remove_children_from_top_nodes() {
    graph_->remove_top_node(a_);
  }

    // We're caching the logsoftmax probabilities here because we'll need them for
  // the backward computation.
  void CrossEntropyPickNodeOp::forward() {
    // C = sum(-B * logsoftmax(A))
    if(!probs_)
      graph_->tensor(probs_, a_->val()->shape());

    CudnnLogSoftmax(probs_, a_->val());

    if(!result_)
      graph_->tensor(result_, a_->val()->shape());

    Pick(_1 = -_2 * _3, result_, probs_, picks_);
    Sum(val_, result_, 1);
  }


}
