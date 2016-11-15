#include "node_operators_binary.h"
#include "expression_graph.h"

namespace marian {
  void BinaryNodeOp::remove_children_from_top_nodes() {
    graph_->remove_top_node(a_);
    graph_->remove_top_node(b_);
  }

  // We're caching the logsoftmax probabilities here because we'll need them for
  // the backward computation.
  void CrossEntropyNodeOp::forward() {
    // C = sum(-B * logsoftmax(A))
    if(!probs_)
      graph_->tensor(probs_, a_->val()->shape());

    CudnnLogSoftmax(probs_, a_->val());

    if(!result_)
      graph_->tensor(result_, a_->val()->shape());
    Element(_1 = -_2 * _3, result_, b_->val(), probs_);
    Sum(val_, result_, 1);
  }

  void NaryNodeOp::remove_children_from_top_nodes() {
    for(auto child : children_)
      graph_->remove_top_node(child);
  }

}
