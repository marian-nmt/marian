#pragma once

#include <map>
#include <memory>

#include "common/config.h"
#include "graph/expression_graph.h"
#include "optimizers/clippers.h"
#include "tensors/tensor.h"
#include "training/training_state.h"

namespace marian {

class OptimizerBase : public TrainingObserver {
public:
  OptimizerBase(float eta, Ptr<ClipperBase> clipper = nullptr)
      : eta_(eta), clipper_(clipper) {}

  void update(Ptr<ExpressionGraph> graph, float multiplyFactor = 1.0f) {
    Tensor p = graph->params()->vals();
    Tensor g = graph->params()->grads();

    update(p, g, multiplyFactor);
  }

  void update(Tensor params, Tensor grads, float multiplyFactor = 1.0f) {
    if(clipper_)
      clipper_->clip(grads);

    // In case we want to add a multiply factor to our learning rate
    multiplyFactor_ = multiplyFactor;
    updateImpl(params, grads);
  }

  virtual void actAfterEpoch(TrainingState& state) { eta_ = state.eta; }
  virtual void actAfterBatches(TrainingState& state) { eta_ = state.eta; }
  virtual void actAfterStalled(TrainingState& state) { eta_ = state.eta; }

  void setParams(const std::vector<float>& params) { parseParams(params); }

protected:
  virtual void updateImpl(Tensor params, Tensor grads) = 0;
  virtual void parseParams(const std::vector<float>& params) = 0;

  // Learning rate
  float eta_;
  // Compensates for larger batch
  float multiplyFactor_;
  // Clip gradient norm
  Ptr<ClipperBase> clipper_;
};

class Sgd : public OptimizerBase {
public:
  Sgd(float eta, Ptr<ClipperBase> clipper = nullptr)
      : OptimizerBase(eta, clipper) {}

private:
  void updateImpl(Tensor params, Tensor grads);

  virtual void parseParams(const std::vector<float>& params) {}
};

// @TODO: Add serialization for historic gradients and parameters
class Adagrad : public OptimizerBase {
public:
  Adagrad(float eta, Ptr<ClipperBase> clipper = nullptr)
      : OptimizerBase(eta, clipper) {}

  virtual void actAfterEpoch(TrainingState& state) {
    OptimizerBase::actAfterEpoch(state);
    if(state.reset)
      resetStats();
  }

  virtual void actAfterBatches(TrainingState& state) {
    OptimizerBase::actAfterBatches(state);
    if(state.reset)
      resetStats();
  }

  virtual void actAfterStalled(TrainingState& state) {
    OptimizerBase::actAfterStalled(state);
    if(state.reset)
      resetStats();
  }

private:
  void updateImpl(Tensor params, Tensor grads);
  void resetStats();

  virtual void parseParams(const std::vector<float>& params) {
    if(params.size() > 0)
      eps_ = params[0];
  }

  float eps_ = 1e-8;
  Ptr<TensorAllocator> alloc_;
  Tensor gt_;
};

// @TODO: Add serialization for historic gradients and parameters
// https://arxiv.org/pdf/1412.6980v8.pdf
class Adam : public OptimizerBase {
public:
  Adam(float eta, Ptr<ClipperBase> clipper = nullptr)
      : OptimizerBase(eta, clipper), t_(0) {}

private:
  void updateImpl(Tensor params, Tensor grads);

  virtual void actAfterEpoch(TrainingState& state) {
    OptimizerBase::actAfterEpoch(state);
    if(state.reset)
      resetStats();
  }

  virtual void actAfterBatches(TrainingState& state) {
    OptimizerBase::actAfterBatches(state);
    if(state.reset)
      resetStats();
  }

  virtual void actAfterStalled(TrainingState& state) {
    OptimizerBase::actAfterStalled(state);
    if(state.reset)
      resetStats();
  }

private:
  void resetStats();

  virtual void parseParams(const std::vector<float>& params) {
    if(params.size() > 0)
      beta1_ = params[0];
    if(params.size() > 1)
      beta2_ = params[1];
    if(params.size() > 2)
      eps_ = params[2];
  }

  float beta1_ = 0.9;
  float beta2_ = 0.999;
  float eps_ = 1e-8;
  size_t t_;

  Ptr<TensorAllocator> alloc_;
  Tensor mt_;
  Tensor vt_;
};

template <class Algorithm>
Ptr<OptimizerBase> Optimizer(float eta,
                             Ptr<ClipperBase> clipper = nullptr,
                             std::vector<float> params = {}) {
  auto opt = Ptr<OptimizerBase>(new Algorithm(eta, clipper));
  opt->setParams(params);
  return opt;
}

Ptr<OptimizerBase> Optimizer(Ptr<Config> options);
}
