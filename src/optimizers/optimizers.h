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
  template <typename... Args>
  OptimizerBase(Ptr<Config> options, float eta, Args... args)
      : clipper_(Get(keywords::clip, nullptr, args...)), eta_(eta), multiply_factor(1.0f) {
    scale_lr = options->get<bool>("batch-flexible-lr");
    batch_target_num_words = options->get<float>("batch-normal-words");
  }

  void update(Ptr<ExpressionGraph> graph, size_t batch_size=0, size_t batch_words=0) {
    Tensor p = graph->params()->vals();
    Tensor g = graph->params()->grads();

    if (scale_lr) {
      multiply_factor = batch_words/batch_target_num_words;
    }

    update(p, g);
  }

  void update(Tensor params, Tensor grads) {
    if(clipper_)
      clipper_->clip(grads);
    updateImpl(params, grads);
  }

  void actAfterEpoch(TrainingState& state) {
    eta_ = state.eta;
  }
  void actAfterBatches(TrainingState& state) {
    eta_ = state.eta;
  }
  void actAfterStalled(TrainingState& state) {
    eta_ = state.eta;
  }

protected:
  virtual void updateImpl(Tensor params, Tensor grads) = 0;

  Ptr<ClipperBase> clipper_;
  float eta_;
  float multiply_factor; //Compensates for larger batch
  float batch_target_num_words;
  bool scale_lr;
};

class Sgd : public OptimizerBase {
public:
  template <typename... Args>
  Sgd(Ptr<Config> options, float eta, Args... args) : OptimizerBase(options, eta, args...) {}

private:
  void updateImpl(Tensor params, Tensor grads);
};

// @TODO: Add serialization for historic gradients and parameters
class Adagrad : public OptimizerBase {
public:
  template <typename... Args>
  Adagrad(Ptr<Config> options, float eta, Args... args)
      : OptimizerBase(options, eta, args...), eps_(Get(keywords::eps, 1e-8, args...)) {}

private:
  void updateImpl(Tensor params, Tensor grads);

  float eps_;
  Ptr<TensorAllocator> alloc_;
  Tensor gt_;
};

// @TODO: Add serialization for historic gradients and parameters
// https://arxiv.org/pdf/1412.6980v8.pdf
class Adam : public OptimizerBase {
public:
  template <typename... Args>
  Adam(Ptr<Config> options, float eta, Args... args)
      : OptimizerBase(options, eta, args...),
        beta1_(Get(keywords::beta1, 0.9, args...)),
        beta2_(Get(keywords::beta2, 0.999, args...)),
        eps_(Get(keywords::eps, 1e-8, args...)),
        t_(0) {}

  void updateImpl(Tensor params, Tensor grads);

private:
  float beta1_;
  float beta2_;
  float eps_;
  size_t t_;

  Ptr<TensorAllocator> mtAlloc_;
  Tensor mt_;
  Ptr<TensorAllocator> vtAlloc_;
  Tensor vt_;
};

template <class Algorithm, typename... Args>
Ptr<OptimizerBase> Optimizer(Ptr<Config> options, float eta, Args&&... args) {
  return Ptr<OptimizerBase>(new Algorithm(options, eta, args...));
}

Ptr<OptimizerBase> Optimizer(Ptr<Config> options);
}
