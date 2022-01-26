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
  /**
   * Construct a layer instance in a given graph.
   * @param graph a shared pointer a graph
   * @return a shared pointer to the layer object
   */
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

/**
 * A convenient typedef for constructing a MLP dense layer.
 * @TODO: change naming convention
 */
typedef Accumulator<DenseFactory> dense;

/**
 * Base factory for output layers, can be used in a multi-layer network factory.
 */
struct LogitLayerFactory : public Factory {
  using Factory::Factory;
  virtual Ptr<IUnaryLogitLayer> construct(Ptr<ExpressionGraph> graph) = 0;
};

/**
 * Implementation of Output layer factory, can be used in a multi-layer network factory.
 * @TODO: In the long run, I hope we can get rid of the abstract factories altogether.
 */
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

/**
 * A convenient typedef for constructing a MLP output layer.
 * @TODO: change naming convention
 */
typedef Accumulator<OutputFactory> output;

/** Multi-layer network, holds and applies layers. */
class MLP : public IUnaryLogitLayer, public IHasShortList {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

  std::vector<Ptr<IUnaryLayer>> layers_;

public:
  /**
   * Construct a MLP container in the graph.
   * @param graph The expression graph.
   * @param options The options used for this mlp container.
   */
  MLP(Ptr<ExpressionGraph> graph, Ptr<Options> options) : graph_(graph), options_(options) {}
  /**
   * Apply/Link a vector of mlp layers (with the given inputs) to the expression graph.
   * @param av The vector of input expressions
   * @return The expression holding the mlp container
   */
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
  /**
   * Apply/Link a vector of mlp layers (with the given inputs) to the expression graph.
   * @param av The vector of input expressions
   * @return The expression holding the mlp container as a
   *         <a href=https://en.wikipedia.org/wiki/Logit>Logits</a> object
   */
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
  /**
   * Apply/Link a mlp layer (with the given input) to the expression graph.
   * @param e The input expression
   * @return The expression holding the mlp container
   */
  Expr apply(Expr e) override { return apply(std::vector<Expr>{e}); }
  /**
   * Apply/Link a mlp layer (with the given input) to the expression graph.
   * @param e The input expression
   * @return The expression holding the mlp container as a
   *         <a href=https://en.wikipedia.org/wiki/Logit>Logits</a> object
   */
  Logits applyAsLogits(Expr e) override { return applyAsLogits(std::vector<Expr>{e}); }
  /**
   * Stack a mlp layer to the mlp container.
   * @param layer The mlp layer
   */
  void push_back(Ptr<IUnaryLayer> layer) { layers_.push_back(layer); }
  /**
   * Stack a mlp layer with <a href=https://en.wikipedia.org/wiki/Logit>Logits</a> object to the mlp container.
   * @param layer The mlp layer with <a href=https://en.wikipedia.org/wiki/Logit>Logits</a> object
   */
  void push_back(Ptr<IUnaryLogitLayer> layer) { layers_.push_back(layer); }
  /**
   * Set shortlisted words to the mlp container.
   * @param shortlist The given shortlisted words
   */
  void setShortlist(Ptr<data::Shortlist> shortlist) override final {
    auto p = tryAsHasShortlist();
    ABORT_IF(
        !p,
        "setShortlist() called on an MLP with an output layer that does not support short lists");
    p->setShortlist(shortlist);
  }
  /** Remove shortlisted words from the mlp container. */
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
 * Multi-layer network factory. Can hold layer factories.
 * Used to accumulate options for later lazy construction.
 */
class MLPFactory : public Factory {
  using Factory::Factory;

private:
  std::vector<Ptr<LayerFactory>> layers_;

public:
  /**
   * Create a MLP container instance in the expression graph.
   * Used to accumulate options for later lazy construction.
   * @param graph The expression graph
   * @return The shared pointer to the MLP container
   */
  Ptr<MLP> construct(Ptr<ExpressionGraph> graph) {
    auto mlp = New<MLP>(graph, options_);
    for(auto layer : layers_) {
      layer->mergeOpts(options_);
      mlp->push_back(layer->construct(graph));
    }
    return mlp;
  }
  /**
   * Stack a layer to the mlp container.
   * @param lf The layer
   * @return The Accumulator object holding the mlp container
   */
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
  /**
   * Stack a mlp output layer to the mlp container.
   * @param lf The mlp output layer
   * @return The Accumulator object holding the mlp container
   */
  Accumulator<MLPFactory> push_back(const Accumulator<OutputFactory>& lf) {
    push_back(AsLayerFactory<OutputFactory>(lf));
    // layers_.push_back(New<AsLayerFactory<OutputFactory>>(asLayerFactory((OutputFactory&)lf)));
    return Accumulator<MLPFactory>(*this);
  }
};


/**
 * A convenient typedef for constructing MLP layers.
 * @TODO: change naming convention.
 */
typedef Accumulator<MLPFactory> mlp;
}  // namespace mlp

typedef ConstructingFactory<Embedding> EmbeddingFactory;
typedef ConstructingFactory<ULREmbedding> ULREmbeddingFactory;

/** A convenient typedef for constructing a standard embedding layers. */
typedef Accumulator<EmbeddingFactory> embedding;
/** A convenient typedef for constructing ULR word embedding layers. */
typedef Accumulator<ULREmbeddingFactory> ulr_embedding;
}  // namespace marian
