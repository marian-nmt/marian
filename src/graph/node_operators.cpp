#include "node_operators.h"
#include "expression_graph.h"

#include "tensors/tensor_operators.h"
#include "tensors/cpu/sharp/sse_gemm.h"

namespace marian {

size_t ConstantNode::allocate() {
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
