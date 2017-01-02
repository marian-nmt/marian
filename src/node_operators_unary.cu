#include "node_operators_unary.h"
#include "expression_graph.h"

namespace marian {
  void UnaryNodeOp::remove_children_from_top_nodes() {
    graph_->remove_top_node(a_);
  }

  //// We're caching the logsoftmax probabilities here because we'll need them for
  //// the backward computation.
  //void CrossEntropyPickNodeOp::forward() {
  //  // C = sum(-B * logsoftmax(A))
  //  if(!probs_)
  //    graph_->tensor(probs_, a_->val()->shape());
  //    // @TODO: this should be cached in a_->grad()
  //
  //  CudnnLogSoftmax(probs_, a_->val());
  //
  //  PickReduce(-_1 * _2, val_, probs_, picks_);
  //}


}
