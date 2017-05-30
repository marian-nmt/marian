#pragma once

#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "layers/param_initializers.h"

namespace marian {

  class Layer {
    protected:
      std::string name_;
      std::vector<Expr> params_;

    public:
      Layer(const std::string& name) : name_(name) {}

      virtual const decltype(params_)& getParams() {
        return params_;
      }

      virtual const std::string& getName() {
        return name_;
      }
  };

  class DenseNew : public Layer {
    private:
      int outDim_;
      act activation_;
      bool layerNorm_;

    public:
      template <class ...Args>
      DenseNew(const std::string name,
            int outDim,
            Args ...args)
       : Layer(name),
         outDim_(outDim),
         activation_(Get(keywords::activation,
                         act::linear,
                         args...)),
         layerNorm_(Get(keywords::normalize,
                        false, args...)) {}

      template <class ...Args>
      Expr operator()(Args ... args) {
        std::vector<Expr> inputs{args...};

        UTIL_THROW_IF2(inputs.empty(), "No inputs");

        auto g = inputs.back()->graph();

        auto in = concatenate(inputs, keywords::axis=1);

        auto W = g->param(name_ + "_W", {in->shape()[1], outDim_},
                          keywords::init=inits::glorot_uniform);
        auto b = g->param(name_ + "_b", {1, outDim_},
                            keywords::init=inits::zeros);

        params_ = { W, b };

        Expr out;
        if(layerNorm_) {
          auto gamma = g->param(name_ + "_gamma", {1, outDim_},
                                keywords::init=inits::from_value(1.0));

          params_.push_back(gamma);
          out = layer_norm(dot(in, W), gamma, b);
        }
        else {
          out = affine(in, W, b);
        }

        switch (activation_) {
          case act::linear :
            return out;
          case act::tanh :
            return tanh(out);
          case act::logit :
            return logit(out);
          case act::ReLU :
            return relu(out);
          default:
            return out;
        }
      }
  };


  class Dense : public Layer {
    private:
      int outDim_;
      act activation_;
      bool layerNorm_;

    public:
      template <class ...Args>
      Dense(const std::string name,
            int outDim,
            Args ...args)
       : Layer(name),
         outDim_(outDim),
         activation_(Get(keywords::activation,
                         act::linear,
                         args...)),
         layerNorm_(Get(keywords::normalize,
                        false, args...)) {}

      Expr operator()(Expr in) {
        auto g = in->graph();
        auto W = g->param(name_ + "_W", {in->shape()[1], outDim_},
                          keywords::init=inits::glorot_uniform);
        auto b = g->param(name_ + "_b", {1, outDim_},
                            keywords::init=inits::zeros);

        params_ = { W, b };

        Expr out;
        if(layerNorm_) {
          auto gamma = g->param(name_ + "_gamma", {1, outDim_},
                                keywords::init=inits::from_value(1.0));

          params_.push_back(gamma);
          out = layer_norm(dot(in, W), gamma, b);
        }
        else {
          out = affine(in, W, b);
        }

        switch (activation_) {
          case act::linear :
            return out;
          case act::tanh :
            return tanh(out);
          case act::logit :
            return logit(out);
          case act::ReLU :
            return relu(out);
          default:
            return out;
        }
      }

      template <class ...Args>
      Expr operator()(Args ... args) {
        std::vector<Expr> inputs{args...};

        UTIL_THROW_IF2(inputs.empty(), "No inputs");

        auto g = inputs[0]->graph();

        params_ = {};
        std::vector<Expr> outputs;
        size_t i = 0;
        for(auto&& in : inputs) {
          auto W = g->param(name_ + "_W" + std::to_string(i),
                            {in->shape()[1], outDim_},
                            keywords::init=inits::glorot_uniform);
          auto b = g->param(name_ + "_b" + std::to_string(i),
                              {1, outDim_},
                              keywords::init=inits::zeros);
          params_.push_back(W);
          params_.push_back(b);

          if(layerNorm_) {
            auto gamma = g->param(name_ + "_gamma" + std::to_string(i), {1, outDim_},
                                  keywords::init=inits::from_value(1.0));

            params_.push_back(gamma);
            outputs.push_back(layer_norm(dot(in, W), gamma, b));
          }
          else {
            outputs.push_back(affine(in, W, b));
          }
          i++;
        }

        switch (activation_) {
          case act::linear :
            return plus(outputs);
          case act::tanh :
            return tanh(outputs);
          case act::logit :
            return logit(outputs);
          case act::ReLU :
            return relu(outputs);
          default:
            return plus(outputs);
        }

      }
  };


  class DenseTied : public Layer {
    private:
      Expr tiedWeights_;
      int outDim_;
      act activation_;
      bool layerNorm_;

    public:
      template <class ...Args>
      DenseTied(const std::string name,
                Expr tiedWeights,
                int outDim,
                Args ...args)
       : Layer(name),
         tiedWeights_(tiedWeights),
         outDim_(outDim),
         activation_(Get(keywords::activation,
                         act::linear,
                         args...)),
         layerNorm_(Get(keywords::normalize,
                        false, args...)) {}

      Expr operator()(Expr in) {
        auto g = in->graph();

        auto b = g->param(name_ + "_b", {1, outDim_},
                            keywords::init=inits::zeros);

        params_ = { tiedWeights_, b };

        Expr out;
        if(layerNorm_) {
          auto gamma = g->param(name_ + "_gamma", {1, outDim_},
                                keywords::init=inits::from_value(1.0));

          params_.push_back(gamma);
          out = layer_norm(dot(in, tiedWeights_), gamma, b);
        }
        else {
          out = affine(in, tiedWeights_, b);
        }

        switch (activation_) {
          case act::linear :
            return out;
          case act::tanh :
            return tanh(out);
          case act::logit :
            return logit(out);
          case act::ReLU :
            return relu(out);
          default:
            return out;
        }
      }
  };

  class DenseWithFilter : public Layer {
    private:
      int outDim_;
      act activation_;
      bool layerNorm_;
      std::vector<size_t> filter_;

    public:
      template <class ...Args>
      DenseWithFilter(const std::string name,
                      int outDim,
                      const std::vector<size_t>& filter,
                      Args ...args)
       : Layer(name),
         outDim_(outDim),
         filter_(filter),
         activation_(Get(keywords::activation,
                         act::linear,
                         args...)),
         layerNorm_(Get(keywords::normalize,
                        false, args...)) {}

      Expr operator()(Expr in) {
        auto g = in->graph();

        auto W = cols(g->param(name_ + "_W", {in->shape()[1], outDim_},
                               keywords::init=inits::glorot_uniform),
                      filter_);
        auto b = cols(g->param(name_ + "_b", {1, outDim_},
                               keywords::init=inits::zeros),
                      filter_);

        params_ = { W, b };

        Expr out;
        if(layerNorm_) {
          auto gamma = cols(g->param(name_ + "_gamma", {1, outDim_},
                                     keywords::init=inits::from_value(1.0)),
                            filter_);

          params_.push_back(gamma);
          out = layer_norm(dot(in, W), gamma, b);
        }
        else {
          out = affine(in, W, b);
        }

        switch (activation_) {
          case act::linear :
            return out;
          case act::tanh :
            return tanh(out);
          case act::logit :
            return logit(out);
          case act::ReLU :
            return relu(out);
          default:
            return out;
        }
      }

      template <class ...Args>
      Expr operator()(Args ... args) {
        std::vector<Expr> inputs{args...};

        UTIL_THROW_IF2(inputs.empty(), "No inputs");

        auto g = inputs[0]->graph();

        params_ = {};
        std::vector<Expr> outputs;
        size_t i = 0;
        for(auto&& in : inputs) {
          auto W = cols(g->param(name_ + "_W" + std::to_string(i),
                                 {in->shape()[1], outDim_},
                                  keywords::init=inits::glorot_uniform),
                        filter_);
          auto b = cols(g->param(name_ + "_b" + std::to_string(i),
                                 {1, outDim_},
                                 keywords::init=inits::zeros),
                        filter_);

          params_.push_back(W);
          params_.push_back(b);

          if(layerNorm_) {
            auto gamma = cols(g->param(name_ + "_gamma" + std::to_string(i), {1, outDim_},
                                       keywords::init=inits::from_value(1.0)),
                              filter_);

            params_.push_back(gamma);
            outputs.push_back(layer_norm(dot(in, W), gamma, b));
          }
          else {
            outputs.push_back(affine(in, W, b));
          }
          i++;
        }

        switch (activation_) {
          case act::linear :
            return plus(outputs);
          case act::tanh :
            return tanh(outputs);
          case act::logit :
            return logit(outputs);
          case act::ReLU :
            return relu(outputs);
          default:
            return plus(outputs);
        }

      }
  };

  class Embedding : public Layer {
    private:
      int dimVoc_;
      int dimEmb_;
      std::function<void(Tensor)> init_;

    public:
      template <typename ...Args>
      Embedding(const std::string name,
                int dimVoc,
                int dimEmb,
                Args ... args)
       : Layer(name),
         dimVoc_(dimVoc),
         dimEmb_(dimEmb) {
        init_ = Get(keywords::init,
                    inits::glorot_uniform,
                    args...);
      }

      Expr operator()(Ptr<ExpressionGraph> graph) {
        params_ = {
          graph->param(name_, {dimVoc_, dimEmb_}, keywords::init=init_)
        };
        return params_.back();
      }
  };

  class CrossEntropyCost : public Layer {
    public:
      CrossEntropyCost(const std::string name)
       : Layer(name) {}

      template <typename ...Args>
      Expr operator()(Expr in, Expr picks, Args ...args) {
        auto mask = Get(keywords::mask, nullptr, args...);

        auto ce = cross_entropy(in, picks);

        if(mask)
          ce = ce * mask;

        auto cost = mean(sum(ce, keywords::axis=2),
                         keywords::axis=0);
        return cost;
      }

  };
}
