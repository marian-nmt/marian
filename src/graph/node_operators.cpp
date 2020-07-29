#include "node_operators.h"
#include "expression_graph.h"

#include "tensors/tensor_operators.h"

namespace marian {

ConstantNode::ConstantNode(Ptr<ExpressionGraph> graph,
                          const Shape& shape,
                          const Ptr<inits::NodeInitializer>& init,
                          Type valueType)
    : Node(graph, shape, valueType),
      init_(init),
      initialized_(false) {
  init_->setAllocator(graph->allocator());
  setTrainable(false);
}

void ConstantNode::allocate() {
  if(!val_) {
    graph()->allocateForward(this);
  }
}

void ConstantNode::init() {
  if(!initialized_) {
    init_->apply(val_);
    initialized_ = true;
  }
  init_.reset();
}

ParamNode::ParamNode(Ptr<ExpressionGraph> graph,
                     const Shape& shape,
                     const Ptr<inits::NodeInitializer>& init,
                     bool fixed)
    : ParamNode(graph, shape, init, Type::float32, fixed) {}

ParamNode::ParamNode(Ptr<ExpressionGraph> graph,
                     const Shape& shape,
                     const Ptr<inits::NodeInitializer>& init,
                     Type valueType,
                     bool fixed)
    : Node(graph, shape, valueType),
      init_(init),
      initialized_(false) {
  init_->setAllocator(graph->allocator());
  setTrainable(!fixed);
  setMemoize(graph->isInference());
}

void ParamNode::init() {
  if(!initialized_) {
    init_->apply(val_);
    initialized_ = true;
  }
  init_.reset();
}
}  // namespace marian
