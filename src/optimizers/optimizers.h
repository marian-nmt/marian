#pragma once

#include <map>
#include <memory>

#include "common/config.h"
#include "graph/expression_graph.h"
#include "optimizers/clippers.h"
#include "tensors/tensor.h"
#include "training/training_state.h"

namespace marian {

/**
 * Base class for optimizers.
 */
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
  virtual void actAfterLoaded(TrainingState& state) {
    eta_ = state.eta;
    multiplyFactor_ = state.factor;
  }

  void setParams(const std::vector<float>& params) { parseParams(params); }

  virtual void load(const std::string& name,
                    std::vector<Ptr<OptimizerBase>> opts,
                    std::vector<size_t> devices) {}
  virtual void save(const std::string& name,
                    std::vector<Ptr<OptimizerBase>> opts,
                    std::vector<size_t> devices) {}

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

  void load(const std::string& name,
            std::vector<Ptr<OptimizerBase>> opts,
            std::vector<size_t> devices) {
    if(!boost::filesystem::exists(name))
      return;

    // @TODO: implement multi-gpu setting
    if(opts.size() > 1)
      return;

    LOG(info, "Loading Adam parameters from {}", name);

    auto opt = std::dynamic_pointer_cast<Adam>(opts.front());

    auto numpy = cnpy::npz_load(name);
    for(auto it : numpy) {
      auto name = it.first;
      cnpy::NpyArray& np = it.second;

      // get the size of mt_ and vt_
      int size = 1;
      for(size_t i = 0; i < np.shape.size(); ++i)
        size *= np.shape[i];

      // reserve memory for momentums
      if(!opt->mt_ || !opt->vt_) {
        if(!opt->alloc_)
          opt->alloc_ = New<TensorAllocator>(devices.front());

        opt->alloc_->reserveExact(2 * size);
        opt->alloc_->allocate(opt->mt_, {1, size});
        opt->alloc_->allocate(opt->vt_, {1, size});
      }

      // extract data into a vector
      std::vector<float> npv(size);
      std::copy((float*)np.data, (float*)np.data + size, npv.begin());

      // set tensors
      if(name == "mt_")
        opt->mt_->set(npv);
      if(name == "vt_")
        opt->vt_->set(npv);
    }
  }

  void save(const std::string& name,
            std::vector<Ptr<OptimizerBase>> opts,
            std::vector<size_t> devices) {
    // @TODO: implement multi-gpu setting
    if(opts.size() > 1)
      return;

    LOG(info, "Saving Adam parameters to {}", name);

    auto opt = std::dynamic_pointer_cast<Adam>(opts.front());

    // the shape is the same for mt_ and vt_
    unsigned dim = opt->mt_->shape().size();
    unsigned* shape = new unsigned[dim];
    for(int i = 0; i < dim; ++i)
      shape[i] = opt->mt_->shape()[i];

    std::vector<float> vMt;
    opt->mt_->get(vMt);
    cnpy::npz_save(name, "mt_", vMt.data(), shape, dim, "w");

    std::vector<float> vVt;
    opt->vt_->get(vVt);
    cnpy::npz_save(name, "vt_", vVt.data(), shape, dim, "a");

    delete[] shape;
  }

private:
  float beta1_ = 0.9;
  float beta2_ = 0.999;
  float eps_ = 1e-8;
  size_t t_;

  Ptr<TensorAllocator> alloc_;
  Tensor mt_;
  Tensor vt_;

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

  void resetStats();

  virtual void parseParams(const std::vector<float>& params) {
    if(params.size() > 0)
      beta1_ = params[0];
    if(params.size() > 1)
      beta2_ = params[1];
    if(params.size() > 2)
      eps_ = params[2];
  }
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
