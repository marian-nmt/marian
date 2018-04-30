#pragma once

#include "layers/factory.h"
#include "layers/generic.h"

namespace marian {
namespace mlp {

/**
 * Base class for layer factories, can be used in a multi-layer network factory.
 */
struct LayerFactory : public Factory {
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

/**
 * Dense layer factory, can be used in a multi-layer network factory.
 */
class DenseFactory : public LayerFactory {
public:
  DenseFactory(Ptr<ExpressionGraph> graph) : LayerFactory(graph) {}

  Ptr<Layer> construct() {
    auto dense = New<Dense>(graph_, options_);
    return dense;
  }

  DenseFactory clone() {
    DenseFactory aClone(graph_);
    aClone.options_->merge(options_);
    return aClone;
  }
};

// @TODO: change naming convention
typedef Accumulator<DenseFactory> dense;

/**
 * Factory for output layers, can be used in a multi-layer network factory.
 */
class OutputFactory : public LayerFactory {
protected:
  std::vector<std::pair<std::string, std::string>> tiedParamsTransposed_;
  Ptr<data::Shortlist> shortlist_;

public:
  OutputFactory(Ptr<ExpressionGraph> graph) : LayerFactory(graph) {}

  Accumulator<OutputFactory> tie_transposed(const std::string& param,
                                           const std::string& tied) {
    tiedParamsTransposed_.push_back({param, tied});
    return Accumulator<OutputFactory>(*this);
  }

  Accumulator<OutputFactory> set_shortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
    return Accumulator<OutputFactory>(*this);
  }

  Ptr<Layer> construct() {
    auto output = New<Output>(graph_, options_);
    for(auto& p : tiedParamsTransposed_)
      output->tie_transposed(p.first, p.second);
    output->set_shortlist(shortlist_);
    return output;
  }

  OutputFactory clone() {
    OutputFactory aClone(graph_);
    aClone.options_->merge(options_);
    aClone.tiedParamsTransposed_ = tiedParamsTransposed_;
    aClone.shortlist_ = shortlist_;
    return aClone;
  }
};

// @TODO: change naming convention
typedef Accumulator<OutputFactory> output;

/**
 * Multi-layer network, holds and applies layers.
 */
class MLP {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

  std::vector<Ptr<Layer>> layers_;

public:
  MLP(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : graph_(graph), options_(options) {}

  template <typename... Args>
  Expr apply(Args... args) {
    std::vector<Expr> av = {args...};

    Expr output;
    if(av.size() == 1)
      output = layers_[0]->apply(av[0]);
    else
      output = layers_[0]->apply(av);

    for(int i = 1; i < layers_.size(); ++i)
      output = layers_[i]->apply(output);

    return output;
  }

  void push_back(Ptr<Layer> layer) { layers_.push_back(layer); }
};

/**
 * Multi-layer network factory. Can hold layer factories. Used
 * to accumulate options for later lazy construction.
 */
class MLPFactory : public Factory {
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

  Ptr<MLP> operator->() { return construct(); }

  template <class LF>
  Accumulator<MLPFactory> push_back(const LF& lf) {
    layers_.push_back(New<LF>(lf));
    return Accumulator<MLPFactory>(*this);
  }
};

// @TODO: change naming convention.
typedef Accumulator<MLPFactory> mlp;
}
}
