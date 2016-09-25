#pragma once

#include <map>
#include <boost/any.hpp>
#include "tensor_operators.h"

namespace marian {

// @TODO: modify computation graph to group all paramters in single matrix object.
// This will allow to perform a single large SGD update per batch. Currently there
// are as many updates as different parameters.

class Sgd {
  public:
    Sgd(float eta=0.01) : eta_(eta) {}

    void operator()(ExpressionGraph& graph, BatchPtr batch) {
      graph.backprop(*batch);

      for(auto& param : graph.params())
        Element(_1 -= eta_ * _2,
                param.val(), param.grad());
    }

  private:
    float eta_;
};

// @TODO: Add serialization for historic gradients and parameters
class Adagrad {
  public:
    Adagrad(float eta=0.01, float eps=1e-8)
    : eta_(eta), eps_(eps) {}

    void operator()(ExpressionGraph& graph, BatchPtr batch) {
      graph.backprop(*batch);

      if(gt_.size() < graph.params().size())
        for(auto& param : graph.params())
          gt_.emplace_back(Tensor(param.grad().shape(), 0));

      auto gtIt = gt_.begin();
      for(auto& param : graph.params()) {
        Element(_1 += (_2 * _2),
                *gtIt, param.grad());
        Element(_1 -= (eta_ / (Sqrt(_2) + eps_)) * _3,
                param.val(), *gtIt, param.grad());
        gtIt++;
      }
    }

  private:
    float eta_;
    float eps_;
    std::vector<Tensor> gt_;
};


// @TODO: Add serialization for historic gradients and parameters
// https://arxiv.org/pdf/1412.6980v8.pdf
class Adam {
  public:
    Adam(float eta=0.001, float beta1=0.9, float beta2=0.999, float eps=1e-8)
    : eta_(eta), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {}

    void operator()(ExpressionGraph& graph, BatchPtr batch) {
      graph.backprop(*batch);

      if(mt_.size() < graph.params().size()) {
        for(auto& param : graph.params()) {
          mt_.emplace_back(Tensor(param.grad().shape(), 0));
          vt_.emplace_back(Tensor(param.grad().shape(), 0));
        }
      }

      t_++;
      float denom1 = 1 - pow(beta1_, t_);
      float denom2 = 1 - pow(beta2_, t_);

      auto mtIt = mt_.begin();
      auto vtIt = vt_.begin();

      for(auto& param : graph.params()) {
        Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2),
                *mtIt, param.grad());
        Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)),
                *vtIt, param.grad());
        Element(_1 -= eta_ * (_2 / denom1) / (Sqrt(_3 / denom2) + eps_),
                param.val(), *mtIt, *vtIt);
        mtIt++; vtIt++;
      }
    }

  private:
    float eta_;
    float beta1_;
    float beta2_;
    float eps_;
    size_t t_;
    std::vector<Tensor> mt_;
    std::vector<Tensor> vt_;
};

}
