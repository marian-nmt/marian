#pragma once

#include <map>
#include <memory>

#include "kernels/tensor_operators.h"
#include "training/config.h"
#include "optimizers/clippers.h"

namespace marian {

class OptimizerBase {
  public:
    template <typename ...Args>
    OptimizerBase(float eta, Args... args)
    : clipper_(Get(keywords::clip, nullptr, args...)),
      eta_(eta) {}

    float backpropUpdate(Ptr<ExpressionGraph> graph) {
      graph->forward();
      float cost = graph->topNode()->scalar();
      graph->backprop();
      update(graph);
      return cost;
    }

    void update(Ptr<ExpressionGraph> graph) {
      Tensor p = graph->params().vals();
      Tensor g = graph->params().grads();
      update(p, g);
    }

    void update(Tensor params, Tensor grads) {
      if(clipper_)
        clipper_->clip(grads);
      updateImpl(params, grads);
    }

    void updateSchedule() {
      //eta_ *= 0.5;
      //LOG(info, "Changing learning rate to {}", eta_);
    }

  protected:

    virtual void updateImpl(Tensor params, Tensor grads) = 0;

    Ptr<ClipperBase> clipper_;
    float eta_;
};

class Sgd : public OptimizerBase {
  public:
    template <typename ...Args>
    Sgd(float eta, Args... args)
    : OptimizerBase(eta, args...) {}

  private:
    void updateImpl(Tensor params, Tensor grads) {
      Element(_1 -= eta_ * _2, params, grads);
    }
};

// @TODO: Add serialization for historic gradients and parameters
class Adagrad : public OptimizerBase {
  public:
    template <typename ...Args>
    Adagrad(float eta, Args ...args)
    : OptimizerBase(eta, args...),
      eps_(Get(keywords::eps, 1e-8, args...))
    {}

  private:
    void updateImpl(Tensor params, Tensor grads) {
      if(!alloc_)
        alloc_ = New<TensorAllocator>(params->getDevice());

      if(!gt_) {
        int totalSize = params->size();
        alloc_->reserveExact(totalSize);
        alloc_->allocate(gt_, {1, totalSize});
        gt_->set(0);
      }

      Element(_1 += (_2 * _2),
              gt_, grads);

      Element(_1 -= (eta_ / (Sqrt(_2) + eps_)) * _3,
              params, gt_, grads);
    }

    float eps_;
    Ptr<TensorAllocator> alloc_;
    Tensor gt_;
};


// @TODO: Add serialization for historic gradients and parameters
// https://arxiv.org/pdf/1412.6980v8.pdf
class Adam : public OptimizerBase {
  public:
    template <typename ...Args>
    Adam(float eta, Args ...args)
    : OptimizerBase(eta, args...),
      beta1_(Get(keywords::beta1, 0.9, args...)),
      beta2_(Get(keywords::beta2, 0.999, args...)),
      eps_(Get(keywords::eps, 1e-8, args...)),
      t_(0)
    {}

    void updateImpl(Tensor params, Tensor grads) {

      if(!mtAlloc_)
        mtAlloc_ = New<TensorAllocator>(params->getDevice());
      if(!vtAlloc_)
        vtAlloc_ = New<TensorAllocator>(params->getDevice());

      if(!mt_) {
        int totalSize = params->size();
        mtAlloc_->reserveExact(totalSize);
        mtAlloc_->allocate(mt_, {1, totalSize});
        mt_->set(0);

        vtAlloc_->reserveExact(totalSize);
        vtAlloc_->allocate(vt_, {1, totalSize});
        vt_->set(0);
      }

      t_++;
      float denom1 = 1 - std::pow(beta1_, t_);
      float denom2 = 1 - std::pow(beta2_, t_);

      Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2),
              mt_, grads);
      Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)),
              vt_, grads);

      Element(_1 -= eta_ * (_2 / denom1) / (Sqrt(_3 / denom2) + eps_),
              params, mt_, vt_);
    }

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

template <class Algorithm, typename ...Args>
Ptr<OptimizerBase> Optimizer(Args&& ...args) {
  return Ptr<OptimizerBase>(new Algorithm(args...));
}

Ptr<OptimizerBase> Optimizer(Ptr<Config> options) {

  Ptr<ClipperBase> clipper = nullptr;
  float clipNorm = options->get<double>("clip-norm");
  if(clipNorm > 0)
    clipper = Clipper<Norm>(clipNorm);

  float lrate = options->get<double>("learn-rate");

  std::string opt = options->get<std::string>("optimizer");

  if(opt == "sgd") {
    return Optimizer<Sgd>(lrate, keywords::clip=clipper);
  }
  else if(opt == "adagrad") {
    return Optimizer<Adagrad>(lrate, keywords::clip=clipper);
  }
  else if(opt == "adam") {
    return Optimizer<Adam>(lrate, keywords::clip=clipper);
  }
  else {
    UTIL_THROW2("Unknown optimizer: " << opt);
  }
}

}
