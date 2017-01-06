#include "node_operators_unary.h"
#include "expression_graph.h"

namespace marian {
  void UnaryNodeOp::remove_children_from_top_nodes() {
    graph_->remove_top_node(a_);
  }

   void SoftmaxNodeOp::remove_mask_from_top_nodes() {
    if(mask_)
      graph_->remove_top_node(mask_);
  }
}
