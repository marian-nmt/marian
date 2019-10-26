#pragma once

#include "common/options.h"
#include "graph/expression_graph.h"
#include "optimizers/clippers.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "training/training_state.h"

#include <algorithm>
#include <map>
#include <memory>

namespace marian {

/**
 * Base class for optimizers.
 */
class OptimizerBase : public TrainingObserver {
public:
  OptimizerBase(float eta, size_t refMBWordsParam, Ptr<ClipperBase> clipper)
      : eta_(eta), refMBWordsParam_(refMBWordsParam), clipper_(clipper) {

    // automatic learning-rate adjustment
    // If users provide, in addition to the hyper-parameters, a reference minibatch size,
    // that these hyper-parameters were originally tuned for, then the learning-rate gets
    // adjusted accordingly. Note: Requires user to also use ce-sum criterion.
    if (refMBWordsParam_ != 0)
      LOG(info, "[optimizers] Learning rate gets automatically adjusted as if minibatch size was {}", refMBWordsParam_);
  }

  static constexpr size_t mbSizeNotProvided = SIZE_MAX;

  void update(Ptr<ExpressionGraph> graph, size_t mbSize = mbSizeNotProvided) {
    Tensor p = graph->params()->vals();
    Tensor g = graph->params()->grads();

    update(p, g, mbSize);
  }

  void update(Tensor params, Tensor grads, size_t mbSize = mbSizeNotProvided) {
    if(clipper_)
      clipper_->clip(grads); //@BUGBUG: take into account actual mini-batch size since gradients are not normalized

    size_t refMBWords = refMBWordsParam_;
    if (refMBWords == 0) { // optimizer not configured to use hyper-parameter auto-adjustment
      refMBWords = mbSize = 1; // neutral settings that keep the standard behavior
    }
    else { // optimizer is configured to auto-adjust hyper-parameters
      ABORT_IF(mbSize == mbSizeNotProvided, "Using rational optimizer auto-adjustment with trainer that does not provide MB size");
      // note: this behavior is only meaningful if using the ce-sum criterion
    }

    updateImpl(params, grads, mbSize, refMBWords);
  }

  virtual void init(TrainingState& state) override {
    eta_ = state.eta;
  }
  virtual void actAfterLoaded(TrainingState& state) override {
    eta_ = state.eta;
  }
  virtual void actAfterEpoch(TrainingState& state) override {
    eta_ = state.eta;
    if(state.reset)
      resetStats();
  }
  virtual void actAfterBatches(TrainingState& state) override {
    eta_ = state.eta;
    if(state.reset)
      resetStats();
  }
  virtual void actAfterStalled(TrainingState& state) override {
    eta_ = state.eta;
    if(state.reset)
      resetStats();
  }

  virtual void setParams(const std::vector<float>& params) = 0;

  typedef std::function<void(size_t /*localDeviceIndex*/,
                             std::vector<float>::const_iterator /*begin*/,
                             std::vector<float>::const_iterator /*end*/)> ScatterStateSetFunc;
  typedef std::function<std::vector<float>(size_t /*localDeviceIndex*/)> GatherStateGetFunc;

  typedef std::function<void(const std::vector<float>& /*data*/, const ScatterStateSetFunc& /*setFn*/)> ScatterStateFunc;
  typedef std::function<std::vector<float>(const GatherStateGetFunc& /*getFn*/)> GatherStateFunc;

  virtual void load(const std::string& /*name*/,
                    const std::vector<Ptr<OptimizerBase>>& /*opts*/,
                    const std::vector<Ptr<Backend>>& /*backends*/,
                    const ScatterStateFunc& /*scatterFn*/) {}
  virtual void save(const std::string& /*name*/,
                    const std::vector<Ptr<OptimizerBase>>& /*opts*/,
                    const GatherStateFunc& /*gatherFn*/,
                    bool /*isMainProcess*/ = true) {}

protected:
  virtual void updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) = 0;
  virtual void resetStats() = 0;

  // Learning rate
  float eta_;
  // Reference MB size. This enables automatic adjustment of optimizer hyper-parameters to MB size.
  size_t refMBWordsParam_{0}; // 0 means no adjustment
  // Clip gradient norm
  Ptr<ClipperBase> clipper_;
};

/**
 * @brief Stochastic gradient descent optimizer.
 */
class Sgd : public OptimizerBase {
public:
  Sgd(float eta, size_t refMBWordsParam = 0, Ptr<ClipperBase> clipper = nullptr)
      : OptimizerBase(eta, refMBWordsParam, clipper) {}

  virtual void setParams(const std::vector<float>& /*params*/) override {}
private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) override;

  virtual void resetStats() override {}
};

/**
 * @brief Adagrad optimizer
 *
 * http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
 */
class Adagrad : public OptimizerBase {
public:
  Adagrad(float eta, size_t refMBWordsParam = 0, Ptr<ClipperBase> clipper = nullptr)
      : OptimizerBase(eta, refMBWordsParam, clipper) {}

  void load(const std::string& name,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const std::vector<Ptr<Backend>>& backends,
            const ScatterStateFunc& scatterFn) override;
  void save(const std::string& name,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const GatherStateFunc& gatherFn,
            bool /*isMainProcess*/ = true) override;

  void setParams(const std::vector<float>& params) override {
    if(params.size() > 0)
      eps_ = params[0];
  }

private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) override;
  void resetStats() override;

  float eps_ = 1e-8f;
  Ptr<TensorAllocator> alloc_;
  Tensor gt_;
};

/**
 * @brief Adam optimizer
 *
 * https://arxiv.org/pdf/1412.6980v8.pdf
 *
 * with Frank's modifications for automatic hyper-parameter adjustment.
 */
class Adam : public OptimizerBase {
public:
  Adam(float eta, size_t refMBWordsParam = 0, Ptr<ClipperBase> clipper = nullptr)
      : OptimizerBase(eta, refMBWordsParam, clipper) {}

  void load(const std::string& name,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const std::vector<Ptr<Backend>>& backends,
            const ScatterStateFunc& scatterFn) override;
  void save(const std::string& name,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const GatherStateFunc& gatherFn,
            bool isMainProcess = true) override;

private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) override;
  void resetStats() override;

  // Adam parameters:
  // [beta1, beta2, eps, w, refMBWords]
  virtual void setParams(const std::vector<float>& params) override {
    if(params.size() > 0)
      beta1_ = params[0];
    if(params.size() > 1)
      beta2_ = params[1];
    if(params.size() > 2)
      eps_ = params[2];

    // weighted decay for AdamW, to be explored, disabled by default
    if(params.size() > 3)
      w_ = params[3]; // default (disabled): 0
  }

  // hyper-parameters
  float beta1_ = 0.9f;
  float beta2_ = 0.999f;
  float eps_ = 1e-8f;
  float w_ = 0.0f;

  // CPU-side running accumulators
  double denom1_ = 0;
  double denom2_ = 0;

  // GPU-side running accumulators
  Ptr<TensorAllocator> alloc_;
  Tensor mt_;
  Tensor vt_;
};

template <class Algorithm>
Ptr<OptimizerBase> Optimizer(float eta, size_t refMBWordsParam = 0,
                             Ptr<ClipperBase> clipper = nullptr,
                             std::vector<float> params = {}) {
  auto opt = Ptr<OptimizerBase>(new Algorithm(eta, refMBWordsParam, clipper));
  opt->setParams(params);
  return opt;
}

Ptr<OptimizerBase> Optimizer(Ptr<Options> options);
}  // namespace marian
