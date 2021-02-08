#pragma once

#include "common/options.h"
#include "graph/expression_graph.h"
#include "optimizers/clippers.h"
#include "optimizers/exponential_smoothing.h"
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
class OptimizerBase : public TrainingObserver, public ExponentialSmoothing {
public:
  OptimizerBase(Ptr<Options> options)
  : ExponentialSmoothing(options),
    options_(options),
    eta_(options_->get<float>("learn-rate")),
    refMBWordsParam_(options_->get<size_t>("mini-batch-words-ref", 0)),
    normalizedGradient_{options_->get<bool>("normalize-gradient", false)} // @TODO: get rid of this if we manage to confirm that it does not help with fp16 training
  {

    auto precisions = options_->get<std::vector<std::string>>("precision", {"float32", "float32"});
    ABORT_IF(precisions.size() < 2, "No optimizer precision type specified??");
    
    auto paramType = typeFromString(precisions[0]);
    optimizerType_ = typeFromString(precisions[1]);

    // if true model for forward/backward uses a different type than the optimizer
    castOptimizerType_ = paramType != optimizerType_;
    
    // automatic learning-rate adjustment
    // If users provide, in addition to the hyper-parameters, a reference minibatch size,
    // that these hyper-parameters were originally tuned for, then the learning-rate gets
    // adjusted accordingly. Note: Requires user to also use ce-sum criterion.
    if (refMBWordsParam_ != 0)
      LOG_ONCE(info, "[optimizers] Learning rate gets automatically adjusted as if minibatch size was {}", refMBWordsParam_);
  }

  virtual ~OptimizerBase() {}

  float update(Ptr<ExpressionGraph> graph, size_t mbSize, float costScaleFactor = 1.f) {
    Tensor p = graph->params()->vals();
    Tensor g = graph->params()->grads();

    return update(p, g, mbSize, costScaleFactor);
  }

  float update(Tensor params, Tensor grads, size_t mbSize, float costScaleFactor = 1.f);

  virtual void init(TrainingState& state) override {
    eta_ = state.eta;
    batchesSeen_ = state.batches;
  }

  virtual void actAfterLoaded(TrainingState& state) override {
    eta_ = state.eta;
    batchesSeen_ = state.batches;
  }

  virtual void actAfterEpoch(TrainingState& state) override {
    eta_ = state.eta;
    batchesSeen_ = state.batches;

    if(state.reset)
      resetStats();
  }

  virtual void actAfterBatches(TrainingState& state) override {
    eta_ = state.eta;
    batchesSeen_ = state.batches;

    if(state.reset)
      resetStats();
  }

  virtual void actAfterStalled(TrainingState& state) override {
    eta_ = state.eta;
    batchesSeen_ = state.batches;

    if(state.reset)
      resetStats();
  }

  virtual void setParams(const std::vector<float>& params) = 0;

  typedef std::function<void(size_t /*localDeviceIndex*/,
                             const char* /*begin*/,
                             const char* /*end*/)> ScatterStateSetFunc;
  typedef std::function<io::Item(size_t /*localDeviceIndex*/)> GatherStateGetFunc;

  typedef std::function<void(const io::Item& /*data*/, const ScatterStateSetFunc& /*setFn*/)> ScatterStateFunc;

  typedef std::function<io::Item(const GatherStateGetFunc& /*getFn*/)> GatherStateFunc;

  virtual void load(std::vector<io::Item>& /*items*/,
                    const std::vector<Ptr<OptimizerBase>>& /*opts*/,
                    const std::vector<Ptr<Backend>>& /*backends*/,
                    const ScatterStateFunc& /*scatterFn*/,
                    bool isMainProcess);

  virtual void save(std::vector<io::Item>& /*items*/,
                    const std::vector<Ptr<OptimizerBase>>& /*opts*/,
                    const GatherStateFunc& /*gatherFn*/,
                    bool isMainProcess);

  // This function swaps out the current optimizer parameters with the smoothed version (provided smoothing is enabled).
  // Usually we will call this twice, to swap in and to swap out.
  void swapWithSmoothed(Tensor params);

  // return stateful optimizer shards, for base that's only averaged parameters
  virtual std::vector<Tensor> getShards() { 
    if(avg_)
      return { avg_ }; 
    else
      return { };
  }

protected:
  virtual void updateImpl(Tensor params, Tensor grads, size_t actualMBSize) = 0;
  virtual void resetStats() = 0;

  Ptr<Options> options_;

  float eta_;                      // Learning rate
  size_t refMBWordsParam_{0};      // reference MB size. This enables automatic adjustment of optimizer hyper-parameters to MB size. 0 means no adjustment
  size_t batchesSeen_{0};          // updates seen so far
  bool normalizedGradient_{false}; // has the gradient been normalized by MB size? @TODO: get rid of this if we manage to confirm that it does not help with fp16 training

  Type optimizerType_{Type::float32};
  bool castOptimizerType_{false};

  Ptr<Clipper> clipper_;  // Clip gradient norm

  Ptr<TensorAllocator> baseAlloc_;
  Ptr<Allocator> alloc_;

  Tensor avg_;

  Tensor pm_;
  Tensor gd_;
};

/**
 * @brief Stochastic gradient descent optimizer.
 */
class Sgd : public OptimizerBase {
public:
  Sgd(Ptr<Options> options) : OptimizerBase(options) {}

  void load(std::vector<io::Item>& /*items*/,
            const std::vector<Ptr<OptimizerBase>>& /*opts*/,
            const std::vector<Ptr<Backend>>& /*backends*/,
            const ScatterStateFunc& /*scatterFn*/,
            bool isMainProcess) override;

  void save(std::vector<io::Item>& items,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const GatherStateFunc& gatherFn,
            bool isMainProcess) override;

  virtual void setParams(const std::vector<float>& /*params*/) override {}
private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize) override;

  virtual void resetStats() override {}
};

/**
 * @brief Adagrad optimizer
 *
 * http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
 */
class Adagrad : public OptimizerBase {
public:
  Adagrad(Ptr<Options> options) : OptimizerBase(options) {}

  void load(std::vector<io::Item>& /*items*/,
            const std::vector<Ptr<OptimizerBase>>& /*opts*/,
            const std::vector<Ptr<Backend>>& /*backends*/,
            const ScatterStateFunc& /*scatterFn*/,
            bool isMainProcess) override;

  void save(std::vector<io::Item>& items,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const GatherStateFunc& gatherFn,
            bool isMainProcess) override;

  void setParams(const std::vector<float>& params) override {
    if(params.size() > 0)
      eps_ = params[0];
  }

  std::vector<Tensor> getShards() override { 
    auto shards = OptimizerBase::getShards();
    shards.push_back(gt_);
    return shards;
  }

private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize) override;
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
  Adam(Ptr<Options> options) : OptimizerBase(options) {}

  void load(std::vector<io::Item>& /*items*/,
            const std::vector<Ptr<OptimizerBase>>& /*opts*/,
            const std::vector<Ptr<Backend>>& /*backends*/,
            const ScatterStateFunc& /*scatterFn*/,
            bool isMainProcess) override;

  void save(std::vector<io::Item>& items,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const GatherStateFunc& gatherFn,
            bool isMainProcess) override;

  std::vector<Tensor> getShards() override { 
    auto shards = OptimizerBase::getShards();
    shards.push_back(mt_);
    shards.push_back(vt_);
    return shards;
  }

private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize) override;
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

Ptr<OptimizerBase> Optimizer(Ptr<Options> options);
}  // namespace marian
