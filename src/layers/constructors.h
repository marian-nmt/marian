#pragma once

#include "layers/factory.h"
#include "layers/generic.h"

namespace marian {
namespace mlp {

/**
 * Base class for layer factories, can be used in a multi-layer network factory.
 */
struct LayerFactory : public Factory {
  LayerFactory() : Factory() {}
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

  virtual Ptr<IUnaryLayer> construct(Ptr<ExpressionGraph> graph) = 0;
};

/**
 * Dense layer factory, can be used in a multi-layer network factory.
 */
class DenseFactory : public LayerFactory {
public:
  Ptr<IUnaryLayer> construct(Ptr<ExpressionGraph> graph) override {
    return New<Dense>(graph, options_);
  }

  DenseFactory clone() {
    DenseFactory aClone;
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
  std::string tiedTransposedName_;
  Ptr<data::Shortlist> shortlist_;

public:
  Accumulator<OutputFactory> tieTransposed(const std::string& tied) {
    tiedTransposedName_ = tied;
    return Accumulator<OutputFactory>(*this);
  }

  Accumulator<OutputFactory> setShortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
    return Accumulator<OutputFactory>(*this);
  }

  Ptr<IUnaryLayer> construct(Ptr<ExpressionGraph> graph) override {
    auto output = New<Output>(graph, options_);
    output->tieTransposed(graph->get(tiedTransposedName_));
    output->setShortlist(shortlist_);
    return output;
  }

  OutputFactory clone() {
    OutputFactory aClone;
    aClone.options_->merge(options_);
    aClone.tiedTransposedName_ = tiedTransposedName_;
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

  std::vector<Ptr<IUnaryLayer>> layers_;

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

    for(size_t i = 1; i < layers_.size(); ++i)
      output = layers_[i]->apply(output);

    return output;
  }

  void push_back(Ptr<IUnaryLayer> layer) { layers_.push_back(layer); }
};

/**
 * Multi-layer network factory. Can hold layer factories. Used
 * to accumulate options for later lazy construction.
 */
class MLPFactory : public Factory {
private:
  std::vector<Ptr<LayerFactory>> layers_;

public:
  Ptr<MLP> construct(Ptr<ExpressionGraph> graph) {
    auto mlp = New<MLP>(graph, options_);
    for(auto layer : layers_) {
      layer->getOptions()->merge(options_);
      mlp->push_back(layer->construct(graph));
    }
    return mlp;
  }

  template <class LF>
  Accumulator<MLPFactory> push_back(const LF& lf) {
    layers_.push_back(New<LF>(lf));
    return Accumulator<MLPFactory>(*this);
  }
};

// @TODO: change naming convention.
typedef Accumulator<MLPFactory> mlp;
}  // namespace mlp
}  // namespace marian
