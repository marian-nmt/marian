#pragma once

#include "common/definitions.h"
#include "graph/expression_operators.h"
#include "marian.h"

#include "data/shortlist.h"
#include "layers/factory.h"

namespace marian {
namespace mlp {
/** Activation functions for MLP layers. */
enum struct act : int { linear, tanh, sigmoid, ReLU, LeakyReLU, PReLU, swish };
}  // namespace mlp
}  // namespace marian

namespace marian {

/**
 * Base class for a layer.
 * Each layer consists of LayerBase and IXXXLayer which defines one or more apply()
 * functions for the respective layer type (different layers may require different signatures).
 * This base class contains configuration info for creating parameters and executing apply().
 */
class LayerBase {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

public:
  LayerBase(Ptr<ExpressionGraph> graph, Ptr<Options> options) : graph_(graph), options_(options) {}

  template <typename T>
  T opt(const std::string key) const {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string key, const T& defaultValue) const {
    return options_->get<T>(key, defaultValue);
  }
};

/** Simplest layer interface: Unary function. */
struct IUnaryLayer {
  virtual ~IUnaryLayer() {}
  /** Link a node as the input for this layer. */
  virtual Expr apply(Expr) = 0;
  /** Link a list of nodes as the inputs for this layer. */
  virtual Expr apply(const std::vector<Expr>& es) {
    ABORT_IF(es.size() > 1, "Not implemented");  // simple stub
    return apply(es.front());
  }
};

/** Shortlist interface for layers. */
struct IHasShortList {
  virtual void setShortlist(Ptr<data::Shortlist> shortlist) = 0;
  virtual void clear() = 0;
};

/** Embedding from corpus sub-batch to (emb, mask). */
struct IEmbeddingLayer {
  virtual std::tuple<Expr /*embeddings*/, Expr /*mask*/> apply(
      Ptr<data::SubBatch> subBatch) const = 0;

  virtual Expr apply(const Words& embIdx, const Shape& shape) const = 0;

  // alternative from indices directly
  virtual Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const = 0;
  virtual ~IEmbeddingLayer() {}
};

/**
 * Base class for Encoder and Decoder classes.
 * Have embeddings and a batch index (=stream index).
 */
class EncoderDecoderLayerBase : public LayerBase {
protected:
  const std::string prefix_;
  const bool embeddingFix_;
  const float dropoutEmbeddings_;  // this drops out full embedding vectors
  const bool inference_;
  const size_t batchIndex_;
  mutable std::vector<Ptr<IEmbeddingLayer>> embeddingLayers_;  // (lazily created)

  EncoderDecoderLayerBase(Ptr<ExpressionGraph> graph,
                          Ptr<Options> options,
                          const std::string& prefix,
                          size_t batchIndex,
                          float dropoutEmbeddings,
                          bool embeddingFix)
      : LayerBase(graph, options),
        prefix_(options->get<std::string>("prefix", prefix)),
        embeddingFix_(embeddingFix),
        dropoutEmbeddings_(dropoutEmbeddings),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", batchIndex)) {}

  virtual ~EncoderDecoderLayerBase() {}

private:
  Ptr<IEmbeddingLayer> createEmbeddingLayer() const;
  Ptr<IEmbeddingLayer> createULREmbeddingLayer() const;

public:
  /**
   * Get all embedding layer(s).
   * It lazily creates the embedding layer on first call.
   * This is lazy mostly because the constructors of the consuming objects are not
   * guaranteed presently to have access to their graph.
   * @param ulr whether to use ULREmbedding layer. false by default.
   * @return a shared pointer to the embedding layer
   */
  Ptr<IEmbeddingLayer> getEmbeddingLayer(bool ulr = false) const;
};

/**
 *  The namespace mlp.
 *  Declare class Dense and all the available functions for creating
 *  <a href=https://en.wikipedia.org/wiki/Multilayer_perceptron>multilayer perceptron (MLP)</a>
 *  network.
 */
namespace mlp {

/**
 * Base class for a fully connected layer.
 * Implement the operations `output = activation(input * weight + bias)`.
 */
class Dense : public LayerBase, public IUnaryLayer {
public:
  /**
   * Construct a dense layer in the graph.
   * @param graph The expression graph.
   * @param options The options used for this dense layer.
   */
  Dense(Ptr<ExpressionGraph> graph, Ptr<Options> options) : LayerBase(graph, options) {}
  /**
   * Apply/Link a vector of dense layers (with the given inputs) to the expression graph.
   * @param inputs The vector of the input expressions
   * @return The expression holding the dense layers
   */
  Expr apply(const std::vector<Expr>& inputs) override {
    ABORT_IF(inputs.empty(), "No inputs");

    auto name = opt<std::string>("prefix");
    auto dim = opt<int>("dim");

    auto useLayerNorm = opt<bool>("layer-normalization", false);
    auto useNematusNorm = opt<bool>("nematus-normalization", false);
    auto activation = (act)opt<int>("activation", (int)act::linear);

    auto g = graph_;

    std::vector<Expr> outputs;
    size_t i = 0;

    std::string num;
    for(auto&& in : inputs) {
      if(inputs.size() > 1)
        num = std::to_string(i);

      Expr W = g->param(name + "_W" + num, {in->shape()[-1], dim}, inits::glorotUniform());
      Expr b = g->param(name + "_b" + num, {1, dim}, inits::zeros());

      if(useLayerNorm) {
        if(useNematusNorm) {
          auto ln_s = g->param(name + "_ln_s" + num, {1, dim}, inits::fromValue(1.f));
          auto ln_b = g->param(name + "_ln_b" + num, {1, dim}, inits::zeros());

          outputs.push_back(layerNorm(affine(in, W, b), ln_s, ln_b, NEMATUS_LN_EPS));
        } else {
          auto gamma = g->param(name + "_gamma" + num, {1, dim}, inits::fromValue(1.0));

          outputs.push_back(layerNorm(dot(in, W), gamma, b));
        }
      } else {
        outputs.push_back(affine(in, W, b));
      }
      i++;
    }

    // clang-format off
    switch(activation) {
      case act::linear:    return plus(outputs);
      case act::tanh:      return tanh(outputs);
      case act::sigmoid:   return sigmoid(outputs);
      case act::ReLU:      return relu(outputs);
      case act::LeakyReLU: return leakyrelu(outputs);
      case act::PReLU:     return prelu(outputs);
      case act::swish:     return swish(outputs);
      default:             return plus(outputs);
    }
    // clang-format on
  };
  /**
   * Apply/Link this dense layer (with the given input) to the expression graph.
   * @param input The input expression
   * @return The expression holding the dense layer
   */
  Expr apply(Expr input) override { return apply(std::vector<Expr>({input})); }
};

}  // namespace mlp

// --- a few layers with built-in parameters created on the fly, without proper object
// @TODO: change to a proper layer object

static inline std::function<Expr(Expr)> activationByName(const std::string& actName) {
  if (actName == "relu")
    return (ActivationFunction*)relu;
  else if (actName == "swish")
    return (ActivationFunction*)swish;
  else if (actName == "gelu")
    return (ActivationFunction*)gelu;
  else if (actName == "sigmoid")
    return (ActivationFunction*)sigmoid;
  else if (actName == "") // return identity function if activation name is empty
    return [](Expr x) { return x; };
  ABORT("Invalid activation name '{}'", actName);
}

// like affine() but with built-in parameters, activation, and dropout
static inline Expr denseInline(Expr x,
                               std::string prefix,
                               std::string suffix,
                               int outDim,
                               Ptr<inits::NodeInitializer> initFn = inits::glorotUniform(),
                               std::string actName = "",
                               float dropProb = 0.0f) {
  auto graph = x->graph();

  auto W = graph->param(prefix + "_W" + suffix, {x->shape()[-1], outDim}, initFn);
  auto b = graph->param(prefix + "_b" + suffix, {1, outDim}, inits::zeros());

  if(actName == "relu") {
    x = affineWithRelu(x, W, b); // speed optimization for inference, @TODO: handle better in future layer framework
  } else {
    x = affine(x, W, b);
    x = activationByName(actName)(x);
  }
  x = dropout(x, dropProb);  // @TODO: check for infernce?
  return x;
}

static inline Expr layerNorm(Expr x, std::string prefix, std::string suffix = std::string()) {
  int dimModel = x->shape()[-1];
  auto scale = x->graph()->param(prefix + "_ln_scale" + suffix, {1, dimModel}, inits::ones());
  auto bias = x->graph()->param(prefix + "_ln_bias" + suffix, {1, dimModel}, inits::zeros());
  return marian::layerNorm(x, scale, bias, 1e-6f);
}

static inline Expr rmsNorm(Expr x, std::string prefix, std::string suffix = std::string()) {
  int dimModel = x->shape()[-1];
  auto scale = x->graph()->param(prefix + "_rms_scale" + suffix, {1, dimModel}, inits::ones());
  return marian::rmsNorm(x, scale, nullptr, 1e-6f);
}

}  // namespace marian
