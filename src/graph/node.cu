#include "graph/expression_graph.h"
#include "graph/node.h"
#include "kernels/tensor_operators.h"

namespace marian {

size_t Node::allocate() {
  size_t elements = 0;
  if(!val_) {
    graph_->tensor(val_, shape_);
    elements = val_->shape().elements();
  }
  return elements;
}

void Node::free() {
  if(val_)
    graph_->free(val_);
  if(adj_)
    graph_->free(adj_);
}

void Node::init_dependent() {
  if(!adj_) {
    graph_->tensor(adj_, shape_);
    adj_->set(1);
  }
}

void Node::set_zero_adjoint() {
  if(!adj_) {
    graph_->tensor(adj_, shape_);
    adj_->set(0);
  }
}

}
