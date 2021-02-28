#include "graph/node.h"
#include "graph/auto_tuner.h"
#include "graph/expression_graph.h"
#include "tensors/backend.h"

namespace marian {

void Node::allocate() {
  if(!val_) {
    graph()->allocateForward(this);
  }
}

void Node::free() {
  if(destroy_) { // don't free views, @TODO: better naming
    //std::cerr << "Freeing" << std::endl;
    if(graph()) {
      if(val_) {
        graph()->free(val_);
        val_ = nullptr;
      }
      if(adj_) {
        graph()->free(adj_);
        adj_ = nullptr;
      }
    }
  }
}

void Node::init_dependent() {
  if(!adj_) {
    graph()->allocateBackward(this);
    adj_->set(1.f);
  }
}

void Node::set_zero_adjoint() {
  if(!adj_) {
    graph()->allocateBackward(this);
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

void Node::record(Ptr<AutoTunerRecorder> recorder,
                  size_t recorderHash,
                  bool stop) {
  recorder_ = recorder;
  recorderHash_ = recorderHash;
  recorderStop_ = stop;
}
}  // namespace marian
