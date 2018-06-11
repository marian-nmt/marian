#include "graph/node.h"
#include "graph/expression_graph.h"
#include "tensors/backend.h"
#include "graph/auto_tuner.h"

namespace marian {

size_t Node::allocate() {
  size_t elements = 0;
  if(!val_) {
    graph()->allocateForward(shared_from_this());
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

/**
 * Initialization for backward step of top node
 * in computation graph. Allocates memory and sets gradient
 * to 1 (df/df == 1).
 */
void Node::init_dependent() {
  if(!adj_) {
    graph()->allocateBackward(shared_from_this());
    adj_->set(1.f);
  }
}

/**
 * Initialization for backward step of any non-top node
 * in computation graph. Allocates memory and sets gradient
 * to 0 for further accumulation of gradients from all
 * parents.
 */
void Node::set_zero_adjoint() {
  if(!adj_) {
    graph()->allocateBackward(shared_from_this());
    adj_->set(0.f);
  }
}

float Node::scalar() {
  return val_->scalar();
}

Ptr<Backend> Node::getBackend() {
  return graph()->getBackend();
}

void Node::forward() {
  if(recorder_)
    recorder_->start(recorderHash_);

  runForward(forwardOps());

  if(recorder_)
    recorder_->stop(recorderHash_, recorderStop_);

}

void Node::backward() {
  if(recorder_)
    recorder_->start(recorderHash_);

  runBackward(backwardOps());

  if(recorder_ && recorderStop_)
    recorder_->stop(recorderHash_, recorderStop_);
}

void Node::record(Ptr<AutoTunerRecorder> recorder, size_t recorderHash, bool stop) {
  recorder_ = recorder;
  recorderHash_ = recorderHash;
  recorderStop_ = stop;
}

void NaryNodeOp::remove_children_from_top_nodes() {
  for(auto child : children_)
    graph()->remove_top_node(child);
}
}
