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
    // C = -dot(B, logsoftmax(A)).
    if(!probs_)
      probs_ = graph_->tensor(a_->val()->shape());
    probs_->set(0.0f);

    CudnnLogSoftmax(probs_, a_->val());
    if(!result_)
      result_ = graph_->tensor(a_->val()->shape());
    Element(_1 = -_2 * _3, result_, b_->val(), probs_);
    SumRowwise(result_, val_);
  }


}
