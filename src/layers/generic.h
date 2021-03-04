#pragma once

#include "marian.h"

#include "data/shortlist.h"
#include "layers/factory.h"

namespace marian { namespace mlp {
  /**
   * @brief Activation functions
   */
  enum struct act : int { linear, tanh, sigmoid, ReLU, LeakyReLU, PReLU, swish };
}}

namespace marian {

// Each layer consists of LayerBase and IXXXLayer which defines one or more apply()
// functions for the respective layer type (different layers may require different signatures).
// This base class contains configuration info for creating parameters and executing apply().
class LayerBase {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

public:
  LayerBase(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : graph_(graph), options_(options) {}

  template <typename T>
  T opt(const std::string key) const {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string key, const T& defaultValue) const {
    return options_->get<T>(key, defaultValue);
  }
};

// Simplest layer interface: Unary function
struct IUnaryLayer {
  virtual ~IUnaryLayer() {}
  virtual Expr apply(Expr) = 0;
  virtual Expr apply(const std::vector<Expr>& es) {
    ABORT_IF(es.size() > 1, "Not implemented"); // simple stub
    return apply(es.front());
  }
};

struct IHasShortList {
  virtual void setShortlist(Ptr<data::Shortlist> shortlist) = 0;
  virtual void clear() = 0;
};

// Embedding from corpus sub-batch to (emb, mask)
struct IEmbeddingLayer {
  virtual std::tuple<Expr/*embeddings*/, Expr/*mask*/> apply(Ptr<data::SubBatch> subBatch) const = 0;

  virtual Expr apply(const Words& embIdx, const Shape& shape) const = 0;

  // alternative from indices directly
  virtual Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const = 0;
  virtual ~IEmbeddingLayer() {}
};

// base class for Encoder and Decoder classes, which have embeddings and a batch index (=stream index)
class EncoderDecoderLayerBase : public LayerBase {
protected:
  const std::string prefix_;
  const bool embeddingFix_;
  const float dropoutEmbeddings_; // this drops out full embedding vectors 
  const bool inference_;
  const size_t batchIndex_;
  mutable std::vector<Ptr<IEmbeddingLayer>> embeddingLayers_; // (lazily created)

  EncoderDecoderLayerBase(Ptr<ExpressionGraph> graph, 
                          Ptr<Options> options, 
                          const std::string& prefix, 
                          size_t batchIndex,
                          float dropoutEmbeddings,
                          bool embeddingFix) :
      LayerBase(graph, options),
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
  // get embedding layer; lazily create on first call
  Ptr<IEmbeddingLayer> getEmbeddingLayer(bool ulr = false) const;
};

class FactoredVocab;

// To support factors, any output projection (that is followed by a softmax) must
// retain multiple outputs, one for each factor. Such layer returns not a single Expr,
// but a Logits object that contains multiple.
// This allows to compute softmax values in a factored manner, where we never create
// a fully expanded list of all factor combinations.
class RationalLoss;
class Logits {
public:
    Logits() {}
    explicit Logits(Ptr<RationalLoss> logits) { // single-output constructor
      logits_.push_back(logits);
    }
    explicit Logits(Expr logits); // single-output constructor from Expr only (RationalLoss has no count)
    Logits(std::vector<Ptr<RationalLoss>>&& logits, Ptr<FactoredVocab> embeddingFactorMapping) // factored-output constructor
      : logits_(std::move(logits)), factoredVocab_(embeddingFactorMapping) {}
    Expr getLogits() const; // assume it holds logits: get them, possibly aggregating over factors
    Expr getFactoredLogits(size_t groupIndex, Ptr<data::Shortlist> shortlist = nullptr, const std::vector<IndexType>& hypIndices = {}, size_t beamSize = 0) const; // get logits for only one factor group, with optional reshuffle
    //Ptr<RationalLoss> getRationalLoss() const; // assume it holds a loss: get that
    Expr applyLossFunction(const Words& labels, const std::function<Expr(Expr/*logits*/,Expr/*indices*/)>& lossFn) const;
    Logits applyUnaryFunction(const std::function<Expr(Expr)>& f) const; // clone this but apply f to all loss values
    Logits applyUnaryFunctions(const std::function<Expr(Expr)>& f1, const std::function<Expr(Expr)>& fother) const; // clone this but apply f1 to first and fother to to all other values

    struct MaskedFactorIndices {
      std::vector<WordIndex> indices; // factor index, or 0 if masked
      std::vector<float> masks;
      void reserve(size_t n) { indices.reserve(n); masks.reserve(n); }
      void push_back(size_t factorIndex); // push back into both arrays, setting mask and index to 0 for invalid entries
      MaskedFactorIndices() {}
      MaskedFactorIndices(const Words& words) { indices = toWordIndexVector(words); } // we can leave masks uninitialized for this special use case
    };
    std::vector<MaskedFactorIndices> factorizeWords(const Words& words) const; // breaks encoded Word into individual factor indices
    Tensor getFactoredLogitsTensor(size_t factorGroup) const; // used for breakDown() only
    size_t getNumFactorGroups() const { return logits_.size(); }
    bool empty() const { return logits_.empty(); }
    Logits withCounts(const Expr& count) const; // create new Logits with 'count' implanted into all logits_
private:
    // helper functions
    Ptr<ExpressionGraph> graph() const;
    Expr constant(const Shape& shape, const std::vector<float>&    data) const { return graph()->constant(shape, inits::fromVector(data)); }
    Expr constant(const Shape& shape, const std::vector<uint32_t>& data) const { return graph()->constant(shape, inits::fromVector(data));  }
    template<typename T> Expr constant(const std::vector<T>& data) const { return constant(Shape{(int)data.size()}, data); } // same as constant() but assuming vector
    Expr indices(const std::vector<uint32_t>& data) const { return graph()->indices(data); } // actually the same as constant(data) for this data type
    std::vector<float> getFactorMasks(size_t factorGroup, const std::vector<WordIndex>& indices) const;
private:
    // members
    // @TODO: we don't use the RationalLoss component anymore, can be removed again, and replaced just by the Expr
    std::vector<Ptr<RationalLoss>> logits_; // [group id][B..., num factors in group]
    Ptr<FactoredVocab> factoredVocab_;
};

// Unary function that returns a Logits object
// Also implements IUnaryLayer, since Logits can be cast to Expr.
// This interface is implemented by all layers that are of the form of a unary function
// that returns multiple logits, to support factors.
struct IUnaryLogitLayer : public IUnaryLayer {
  virtual Logits applyAsLogits(Expr) = 0;
  virtual Logits applyAsLogits(const std::vector<Expr>& es) {
    ABORT_IF(es.size() > 1, "Not implemented"); // simple stub
    return applyAsLogits(es.front());
  }
  virtual Expr apply(Expr e) override { return applyAsLogits(e).getLogits(); }
  virtual Expr apply(const std::vector<Expr>& es) override { return applyAsLogits(es).getLogits(); }
};

namespace mlp {

class Dense : public LayerBase, public IUnaryLayer {
public:
  Dense(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : LayerBase(graph, options) {}

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

      Expr W = g->param(
          name + "_W" + num, {in->shape()[-1], dim}, inits::glorotUniform());
      Expr b = g->param(name + "_b" + num, {1, dim}, inits::zeros());

      if(useLayerNorm) {
        if(useNematusNorm) {
          auto ln_s = g->param(
              name + "_ln_s" + num, {1, dim}, inits::fromValue(1.f));
          auto ln_b = g->param(name + "_ln_b" + num, {1, dim}, inits::zeros());

          outputs.push_back(
              layerNorm(affine(in, W, b), ln_s, ln_b, NEMATUS_LN_EPS));
        } else {
          auto gamma = g->param(
              name + "_gamma" + num, {1, dim}, inits::fromValue(1.0));

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

  Expr apply(Expr input) override { return apply(std::vector<Expr>({input})); }
};

} // namespace mlp

class LSH;

namespace mlp {

class Output : public LayerBase, public IUnaryLogitLayer, public IHasShortList {
private:
  // parameters held by this layer
  Expr Wt_; // weight matrix is stored transposed for efficiency
  Expr b_;
  Expr lemmaEt_; // re-embedding matrix for lemmas [lemmaDimEmb x lemmaVocabSize]
  bool isLegacyUntransposedW{false}; // legacy-model emulation: W is stored in non-transposed form
  bool hasBias_{true};

  Expr cachedShortWt_;  // short-listed version, cached (cleared by clear())
  Expr cachedShortb_;   // these match the current value of shortlist_
  Expr cachedShortLemmaEt_;
  Ptr<FactoredVocab> factoredVocab_;
  
  // optional parameters set/updated after construction
  Expr tiedParam_;
  Ptr<data::Shortlist> shortlist_;
  Ptr<LSH> lsh_;

  void lazyConstruct(int inputDim);
public:
  Output(Ptr<ExpressionGraph> graph, Ptr<Options> options)
    : LayerBase(graph, options), 
      hasBias_{!options->get<bool>("output-omit-bias", false)} {
    clear();
  }

  void tieTransposed(Expr tied) {
    if (Wt_)
      ABORT_IF(tiedParam_.get() != tied.get(), "Tied output projection cannot be changed once weights have been created");
    else
      tiedParam_ = tied;
  }

  void setShortlist(Ptr<data::Shortlist> shortlist) override final {
    if (shortlist_)
      ABORT_IF(shortlist.get() != shortlist_.get(), "Output shortlist cannot be changed except after clear()");
    else {
      ABORT_IF(cachedShortWt_ || cachedShortb_ || cachedShortLemmaEt_, "No shortlist but cached parameters??");
      shortlist_ = shortlist;
    }
    // cachedShortWt_ and cachedShortb_ will be created lazily inside apply()
  }

  // this is expected to be called in sync with graph->clear(), which invalidates
  // cachedShortWt_ etc. in the graph's short-term cache
  void clear() override final {
    shortlist_ = nullptr;
    cachedShortWt_ = nullptr;
    cachedShortb_  = nullptr;
    cachedShortLemmaEt_ = nullptr;
  }

  Logits applyAsLogits(Expr input) override final;
};

}  // namespace mlp


// --- a few layers with built-in parameters created on the fly, without proper object
// @TODO: change to a proper layer object

// like affine() but with built-in parameters, activation, and dropout
static inline
Expr denseInline(Expr x, 
                std::string prefix, 
                std::string suffix, 
                int outDim,
                Ptr<inits::NodeInitializer> initFn = inits::glorotUniform(), 
                const std::function<Expr(Expr)>& actFn = nullptr, 
                float dropProb = 0.0f)
{
  auto graph = x->graph();

  auto W = graph->param(prefix + "_W" + suffix, { x->shape()[-1], outDim }, inits::glorotUniform());
  auto b = graph->param(prefix + "_b" + suffix, { 1,              outDim }, inits::zeros());

  x = affine(x, W, b);
  if (actFn)
    x = actFn(x);
  x = dropout(x, dropProb); // @TODO: check for infernce?
  return x;
}

static inline
Expr layerNorm(Expr x, std::string prefix, std::string suffix = std::string()) {
  int dimModel = x->shape()[-1];
  auto scale = x->graph()->param(prefix + "_ln_scale" + suffix, { 1, dimModel }, inits::ones());
  auto bias  = x->graph()->param(prefix + "_ln_bias"  + suffix, { 1, dimModel }, inits::zeros());
  return marian::layerNorm(x, scale, bias, 1e-6f);
}

}  // namespace marian
