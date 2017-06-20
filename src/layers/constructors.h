#pragma once

#include "layers/factory.h"

namespace marian {
  namespace mlp {
    enum struct act : int { linear, tanh, logit, ReLU };
  }
}

YAML_REGISTER_TYPE(marian::mlp::act, int)

namespace marian {
namespace mlp {

class Layer {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

public:
  Layer(Ptr<ExpressionGraph> graph, Ptr<Options> options)
   : graph_(graph), options_(options)
  {}

  virtual Expr apply(const std::vector<Expr>&) = 0;
  virtual Expr apply(Expr) = 0;
};


struct LayerFactory : public Factory<Layer> {
  LayerFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}
  LayerFactory(const LayerFactory&) = default;
  LayerFactory(LayerFactory&&) = default;

  virtual ~LayerFactory() {}

  template <typename Cast>
  inline Ptr<Cast> as() {
    return std::dynamic_pointer_cast<Cast>(shared_from_this());
  }

  template <typename Cast>
  inline bool is() {
    return as<Cast>() != nullptr;
  }

  virtual Ptr<Layer> construct() = 0;
};

template <class T>
struct SimpleLayerFactory : public LayerFactory {
  SimpleLayerFactory(Ptr<ExpressionGraph> graph) : LayerFactory(graph) {}

  Ptr<Layer> construct() {
    return New<T>(graph_, options_);
  }
};

class Dense : public Layer {
private:
  std::vector<Expr> params_;

public:
  Dense(Ptr<ExpressionGraph> graph, Ptr<Options> options)
   : Layer(graph, options) {}

  Expr apply(const std::vector<Expr>& inputs) {
    UTIL_THROW_IF2(inputs.empty(), "No inputs");
    auto name = options_->get<std::string>("prefix");
    auto dim  = options_->get<int>("dim");

    auto layerNorm = options_->get<bool>("normalization", false);
    auto activation = options_->get<act>("activation", act::linear);

    auto g = graph_;

    params_ = {};
    std::vector<Expr> outputs;
    size_t i = 0;
    for(auto&& in : inputs) {
      auto W = g->param(name + "_W" + std::to_string(i),
                        {in->shape()[1], dim},
                        keywords::init = inits::glorot_uniform);
      auto b = g->param(name + "_b" + std::to_string(i),
                        {1, dim},
                        keywords::init = inits::zeros);
      params_.push_back(W);
      params_.push_back(b);

      if(layerNorm) {
        auto gamma = g->param(name + "_gamma" + std::to_string(i),
                              {1, dim},
                              keywords::init = inits::from_value(1.0));

        params_.push_back(gamma);
        outputs.push_back(layer_norm(dot(in, W), gamma, b));
      } else {
        outputs.push_back(affine(in, W, b));
      }
      i++;
    }

    switch(activation) {
      case act::linear: return plus(outputs);
      case act::tanh: return tanh(outputs);
      case act::logit: return logit(outputs);
      case act::ReLU: return relu(outputs);
      default: return plus(outputs);
    }
  };

  Expr apply(Expr input) {
    auto g = graph_;

    auto name = options_->get<std::string>("prefix");
    auto dim  = options_->get<int>("dim");

    auto layerNorm = options_->get<bool>("normalization", false);
    auto activation = options_->get<act>("activation", act::linear);

    auto W = g->param(name + "_W",
                      {input->shape()[1], dim},
                      keywords::init = inits::glorot_uniform);
    auto b = g->param(name + "_b", {1, dim}, keywords::init = inits::zeros);

    params_ = {W, b};

    Expr out;
    if(layerNorm) {
      auto gamma = g->param(name + "_gamma",
                            {1, dim},
                            keywords::init = inits::from_value(1.0));

      params_.push_back(gamma);
      out = layer_norm(dot(input, W), gamma, b);
    } else {
      out = affine(input, W, b);
    }

    switch(activation) {
      case act::linear: return out;
      case act::tanh: return tanh(out);
      case act::logit: return logit(out);
      case act::ReLU: return relu(out);
      default: return out;
    }
  }
};

typedef Accumulator<SimpleLayerFactory<Dense>> dense;

class MLP {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

  std::vector<Ptr<Layer>> layers_;

public:
  MLP(Ptr<ExpressionGraph> graph, Ptr<Options> options)
   : graph_(graph), options_(options)
  {}

  template <typename ...Args>
  Expr apply(Args ...args) {
    std::vector<Expr> av = {args...};
    auto output = layers_[0]->apply(av);

    for(int i = 1; i < layers_.size(); ++i)
      output = layers_[i]->apply(output);

    return output;
  }

  void push_back(Ptr<Layer> layer) {
    layers_.push_back(layer);
  }
};

class MLPFactory : public Factory<MLP> {
private:
  std::vector<Ptr<LayerFactory>> layers_;

public:
  MLPFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  Ptr<MLP> construct() {
    auto mlp = New<MLP>(graph_, options_);
    for(auto layer : layers_) {
      layer->getOptions()->merge(options_);
      mlp->push_back(layer->construct());
    }
    return mlp;
  }

  Ptr<MLP> operator->() {
    return construct();
  }

  template <class LF>
  Accumulator<MLPFactory> push_back(const LF& lf) {
    layers_.push_back(New<LF>(lf));
    return Accumulator<MLPFactory>(*this);
  }
};

typedef Accumulator<MLPFactory> mlp;

}
}