#include "graph/backend_gpu.h"
#include "graph/expression_graph.h"
#include "graph/node.h"

namespace marian {

size_t Node::allocate() {
  size_t elements = 0;
  if(!val_) {
    graph()->tensor(val_, shape_);
    elements = val_->shape().elements();
  }
  return elements;
}

void Node::free() {
  if(graph()) {
    if(val_)
      graph()->free(val_);
    if(adj_)
      graph()->free(adj_);
  }
}

void Node::init_dependent() {
  if(!adj_) {
    graph()->tensor(adj_, shape_);
    adj_->set(1);
  }
}

void Node::set_zero_adjoint() {
  if(!adj_) {
    graph()->tensor(adj_, shape_);
    adj_->set(0);
  }
}

float Node::scalar() {
  return val_->scalar();
}

Ptr<Backend> Node::getBackend() {
  return graph()->getBackend();
}

void NaryNodeOp::remove_children_from_top_nodes() {
  for(auto child : children_)
    graph()->remove_top_node(child);
}
}
