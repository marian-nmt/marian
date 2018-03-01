#include "expression_graph.h"
#include "node_operators.h"

namespace marian {

size_t ConstantNode::allocate() {
  // @TODO params
  size_t elements = 0;
  if(!val_) {
    graph()->tensor(val_, shape_);
    elements = val_->shape().elements();
  }
  return elements;
}

void ConstantNode::init() {
  if(!initialized_) {
    (*init_)(val_);
    initialized_ = true;
  }
  init_.reset();
}

size_t ParamNode::allocate() {
  // @TODO params
  size_t elements = 0;
  if(!val_) {
    graph()->tensor(val_, shape_);
    elements = val_->shape().elements();
  }
  return elements;
}

void ParamNode::init() {
  if(!initialized_) {
    (*init_)(val_);
    initialized_ = true;
  }
  init_.reset();
}
}
