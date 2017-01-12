#include "node.h"
#include "tensor_operators.h"
#include "expression_graph.h"

namespace marian {

size_t Node::allocate() {
  size_t elements = 0;
  if(!val_) {
    graph_->tensor(val_, shape_);
    elements = val_->shape().elements();
  }
  return elements;
}

void Node::init_dependent() {
  if(!adj_)
    graph_->tensor(adj_, shape_);
  adj_->set(1);
}

void Node::set_zero_adjoint() {
  if(!adj_)
    graph_->tensor(adj_, shape_);
  adj_->set(0);
}

}
