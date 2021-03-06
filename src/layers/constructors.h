#pragma once

#include "layers/embedding.h"
#include "layers/factory.h"
#include "layers/generic.h"
#include "layers/output.h"

namespace marian {
namespace mlp {

/**
 * Base class for layer factories, can be used in a multi-layer network factory.
 */
struct LayerFactory : public Factory {
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
struct LogitLayerFactory : public Factory {
  using Factory::Factory;
  virtual Ptr<IUnaryLogitLayer> construct(Ptr<ExpressionGraph> graph) = 0;
};

// @TODO: In the long run, I hope we can get rid of the abstract factories altogether.
class OutputFactory : public LogitLayerFactory {
  using LogitLayerFactory::LogitLayerFactory;

protected:
  std::string tiedTransposedName_;
  Ptr<data::Shortlist> shortlist_;

public:
  Accumulator<OutputFactory> tieTransposed(const std::string& tied) {
    tiedTransposedName_ = tied;
    return Accumulator<OutputFactory>(*this);
  }

  void setShortlist(Ptr<data::Shortlist> shortlist) { shortlist_ = shortlist; }

  Ptr<IUnaryLogitLayer> construct(Ptr<ExpressionGraph> graph) override {
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
class MLP : public IUnaryLogitLayer, public IHasShortList {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

  std::vector<Ptr<IUnaryLayer>> layers_;

public:
  MLP(Ptr<ExpressionGraph> graph, Ptr<Options> options) : graph_(graph), options_(options) {}

  Expr apply(const std::vector<Expr>& av) override {
    Expr output;
    if(av.size() == 1)
      output = layers_[0]->apply(av[0]);
    else
      output = layers_[0]->apply(av);

    for(size_t i = 1; i < layers_.size(); ++i)
      output = layers_[i]->apply(output);

    return output;
  }

  Logits applyAsLogits(const std::vector<Expr>& av) override {
    // same as apply() except  for the last layer, we invoke applyAsLogits(), which has a different
    // return type
    auto lastLayer = std::dynamic_pointer_cast<IUnaryLogitLayer>(layers_.back());
    ABORT_IF(
        !lastLayer,
        "MLP::applyAsLogits() was called on an MLP whose last layer is not an IUnaryLogitLayer");
    if(layers_.size() == 1) {
      if(av.size() == 1)
        return lastLayer->applyAsLogits(av[0]);
      else
        return lastLayer->applyAsLogits(av);
    } else {
      Expr output;
      if(av.size() == 1)
        output = layers_[0]->apply(av[0]);
      else
        output = layers_[0]->apply(av);
      for(size_t i = 1; i < layers_.size() - 1; ++i)
        output = layers_[i]->apply(output);
      return lastLayer->applyAsLogits(output);
    }
  }

  Expr apply(Expr e) override { return apply(std::vector<Expr>{e}); }
  Logits applyAsLogits(Expr e) override { return applyAsLogits(std::vector<Expr>{e}); }

  void push_back(Ptr<IUnaryLayer> layer) { layers_.push_back(layer); }
  void push_back(Ptr<IUnaryLogitLayer> layer) { layers_.push_back(layer); }

  void setShortlist(Ptr<data::Shortlist> shortlist) override final {
    auto p = tryAsHasShortlist();
    ABORT_IF(
        !p,
        "setShortlist() called on an MLP with an output layer that does not support short lists");
    p->setShortlist(shortlist);
  }

  void clear() override final {
    auto p = tryAsHasShortlist();
    if(p)
      p->clear();
  }

private:
  Ptr<IHasShortList> tryAsHasShortlist() const {
    return std::dynamic_pointer_cast<IHasShortList>(layers_.back());
  }
};

/**
 * Multi-layer network factory. Can hold layer factories. Used
 * to accumulate options for later lazy construction.
 */
class MLPFactory : public Factory {
  using Factory::Factory;

private:
  std::vector<Ptr<LayerFactory>> layers_;

public:
  Ptr<MLP> construct(Ptr<ExpressionGraph> graph) {
    auto mlp = New<MLP>(graph, options_);
    for(auto layer : layers_) {
      layer->mergeOpts(options_);
      mlp->push_back(layer->construct(graph));
    }
    return mlp;
  }

  template <class LF>
  Accumulator<MLPFactory> push_back(const LF& lf) {
    layers_.push_back(New<LF>(lf));
    return Accumulator<MLPFactory>(*this);
  }

  // Special case for last layer, which may be a IUnaryLogitLayer. Requires some hackery,
  // which will go away if we get rid of the abstract factories, and instead just construct
  // all layers immediately, which is my long-term goal for Marian.
private:
  template <class WrappedFactory>
  class AsLayerFactory : public LayerFactory {
    WrappedFactory us;

  public:
    AsLayerFactory(const WrappedFactory& wrapped) : us(wrapped) {}
    Ptr<IUnaryLayer> construct(Ptr<ExpressionGraph> graph) override final {
      auto p = std::static_pointer_cast<IUnaryLayer>(us.construct(graph));
      ABORT_IF(!p, "Attempted to cast a Factory to LayerFactory that isn't one");
      return p;
    }
  };
  template <class WrappedFactory>
  static inline AsLayerFactory<WrappedFactory> asLayerFactory(const WrappedFactory& wrapped) {
    return wrapped;
  }

public:
  Accumulator<MLPFactory> push_back(const Accumulator<OutputFactory>& lf) {
    push_back(AsLayerFactory<OutputFactory>(lf));
    // layers_.push_back(New<AsLayerFactory<OutputFactory>>(asLayerFactory((OutputFactory&)lf)));
    return Accumulator<MLPFactory>(*this);
  }
};

// @TODO: change naming convention.
typedef Accumulator<MLPFactory> mlp;
}  // namespace mlp

typedef ConstructingFactory<Embedding> EmbeddingFactory;
typedef ConstructingFactory<ULREmbedding> ULREmbeddingFactory;

typedef Accumulator<EmbeddingFactory> embedding;
typedef Accumulator<ULREmbeddingFactory> ulr_embedding;
}  // namespace marian
