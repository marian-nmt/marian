#pragma once

#include <map>
#include <boost/any.hpp>
#include "tensor_operators.h"

namespace marian {

class Sgd {
  public:
    Sgd(float eta=0.1) : eta_(eta) {}
    
    void operator()(ExpressionGraph& graph, int batchSize) {
      graph.backprop(batchSize);
      
      for(auto& param : graph.params())
        Element(_1 -= eta_ * _2,
                param.val(), param.grad());
    }
    
  private:
    float eta_;
};

class Adagrad {
  public:
    Adagrad(float eta=0.1) : eta_(eta) {}
    
    void operator()(ExpressionGraph& graph, int batchSize) {
      float fudgeFactor = 1e-6;
      graph.backprop(batchSize);
      
      if(history_.size() < graph.params().size())
        for(auto& param : graph.params())
          history_.emplace_back(Tensor(param.grad().shape(), 0));
      
      auto it = history_.begin();
      for(auto& param : graph.params()) {    
        Element(_1 += _2 * _2, *it, param.grad());
        Element(_1 -= eta_ / (fudgeFactor + Sqrt(_2)) * _3,
                param.val(), *it, param.grad());
        it++;
      }
    }
    
  private:
    float eta_;
    std::vector<Tensor> history_;
};

}