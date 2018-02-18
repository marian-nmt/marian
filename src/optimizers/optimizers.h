#pragma once

#include <map>
#include <memory>
#include <algorithm>

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
                    std::vector<DeviceId> devices) {}
  virtual void save(const std::string& name,
                    std::vector<Ptr<OptimizerBase>> opts,
                    size_t totalSize) {}

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
            std::vector<DeviceId> devices) {
    if(!boost::filesystem::exists(name))
      return;

    LOG(info, "Loading Adam parameters from {}", name);

    std::vector<float> vMt;
    std::vector<float> vVt;
    size_t totalSize = 0;

    auto numpy = cnpy::npz_load(name);
    for(auto it : numpy) {
      auto name = it.first;
      cnpy::NpyArray& np = it.second;

      // get the size of mt_ and vt_, they are the same
      totalSize = np.shape[1];

      // extract data into vectors
      if(name == "adam_mt") {
        vMt.resize(totalSize);
        std::copy((float*)np.data, (float*)np.data + totalSize, vMt.begin());
      }
      if(name == "adam_vt") {
        vVt.resize(totalSize);
        std::copy((float*)np.data, (float*)np.data + totalSize, vVt.begin());
      }
    }

    if(vMt.empty() || vVt.empty()) {
      LOG(info, "[warn] Adam parameters not found in .npz file");
      return;
    }

    size_t shardSize = ceil(totalSize / (float)devices.size());

    size_t id = 0;
    for(auto optBase : opts) {
      auto opt = std::dynamic_pointer_cast<Adam>(optBase);

      int size = std::min(shardSize, totalSize);
      totalSize -= size;

      if(!opt->mt_ || !opt->vt_) {
        if(!opt->alloc_)
          opt->alloc_ = New<TensorAllocator>(devices[id]);

        opt->alloc_->reserveExact(2 * sizeof(float) * size);
        opt->alloc_->allocate(opt->mt_, {1, size});
        opt->alloc_->allocate(opt->vt_, {1, size});
      }

      int shift = id * shardSize;
      std::vector<float> tmpMt(vMt.begin() + shift, vMt.begin() + shift + size);
      opt->mt_->set(tmpMt);
      std::vector<float> tmpVt(vVt.begin() + shift, vVt.begin() + shift + size);
      opt->mt_->set(tmpVt);

      id++;
    }
  }

  void save(const std::string& name,
            std::vector<Ptr<OptimizerBase>> opts,
            size_t totalSize) {
    LOG(info, "Saving Adam parameters to {}", name);

    std::vector<float> vMt;
    std::vector<float> vVt;

    for(auto optBase : opts) {
      auto opt = std::dynamic_pointer_cast<Adam>(optBase);

      std::vector<float> tmpMt;
      opt->mt_->get(tmpMt);
      vMt.insert(vMt.end(), tmpMt.begin(), tmpMt.end());

      std::vector<float> tmpVt;
      opt->vt_->get(tmpVt);
      vVt.insert(vVt.end(), tmpVt.begin(), tmpVt.end());
    }

    // truncate to the real size
    if(totalSize < vMt.size()) {
      vMt.resize(totalSize);
      vVt.resize(totalSize);
    }

    // the shape is the same for mt_ and vt_
    unsigned* shape = new unsigned[2];
    shape[0] = 1;
    shape[1] = vMt.size();

    cnpy::npz_save(name, "adam_mt", vMt.data(), shape, 2, "w");
    cnpy::npz_save(name, "adam_vt", vVt.data(), shape, 2, "a");

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
