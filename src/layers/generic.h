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
};

// base class for Encoder and Decoder classes, which have embeddings and a batch index (=stream index)
class EncoderDecoderLayerBase : public LayerBase {
protected:
  const std::string prefix_;
  const bool embeddingFix_;
  const float dropout_;
  const bool inference_;
  const size_t batchIndex_;
  mutable std::vector<Ptr<IEmbeddingLayer>> embeddingLayers_; // (lazily created)

  EncoderDecoderLayerBase(Ptr<ExpressionGraph> graph, Ptr<Options> options, const std::string& prefix, size_t batchIndex,
        float dropout,
        bool embeddingFix) :
      LayerBase(graph, options),
      prefix_(options->get<std::string>("prefix", prefix)),
      embeddingFix_(embeddingFix),
      dropout_(dropout),
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
    Expr constant(const Shape& shape, const std::vector<float>&    data) const { return graph()->constant(shape, inits::fromVector(data), Type::float32); }
    Expr constant(const Shape& shape, const std::vector<uint32_t>& data) const { return graph()->constant(shape, inits::fromVector(data), Type::uint32);  }
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

class Output : public LayerBase, public IUnaryLogitLayer, public IHasShortList {
private:
  // parameters held by this layer
  Expr Wt_; // weight matrix is stored transposed for efficiency
  Expr b_;
  Expr lemmaEt_; // re-embedding matrix for lemmas [lemmaDimEmb x lemmaVocabSize]
  bool isLegacyUntransposedW{false}; // legacy-model emulation: W is stored in non-transposed form
  Expr cachedShortWt_;  // short-listed version, cached (cleared by clear())
  Expr cachedShortb_;   // these match the current value of shortlist_
  Expr cachedShortLemmaEt_;
  Ptr<FactoredVocab> factoredVocab_;

  // optional parameters set/updated after construction
  Expr tiedParam_;
  Ptr<data::Shortlist> shortlist_;

  void lazyConstruct(int inputDim);
public:
  Output(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : LayerBase(graph, options) {
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

// A regular embedding layer.
// Note that this also applies dropout if the option is passed (pass 0 when in inference mode).
// It is best to not use Embedding directly, but rather via getEmbeddingLayer() in
// EncoderDecoderLayerBase, which knows to pass on all required parameters from options.
class Embedding : public LayerBase, public IEmbeddingLayer {
  Expr E_;
  Ptr<FactoredVocab> factoredVocab_;
  Expr multiRows(const Words& data, float dropProb) const;
public:
  Embedding(Ptr<ExpressionGraph> graph, Ptr<Options> options);

  std::tuple<Expr/*embeddings*/, Expr/*mask*/> apply(Ptr<data::SubBatch> subBatch) const override final;

  Expr apply(const Words& words, const Shape& shape) const override final;

  Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const override final;
};

class ULREmbedding : public LayerBase, public IEmbeddingLayer {
  std::vector<Expr> ulrEmbeddings_; // @TODO: These could now better be written as 6 named class members
public:
  ULREmbedding(Ptr<ExpressionGraph> graph, Ptr<Options> options) : LayerBase(graph, options) {
    std::string name = "url_embed"; //opt<std::string>("prefix");
    int dimKeys = opt<int>("dimTgtVoc");
    int dimQueries = opt<int>("dimSrcVoc");
    int dimEmb = opt<int>("dimEmb");
    int dimUlrEmb =  opt<int>("dimUlrEmb"); // ULR mono embed size
    bool fixed = opt<bool>("fixed", false);

    // Embedding layer initialization should depend only on embedding size, hence fanIn=false
    auto initFunc = inits::glorotUniform(/*fanIn=*/false, /*fanOut=*/true);

    std::string queryFile = opt<std::string>("ulrQueryFile");
    std::string keyFile = opt<std::string>("ulrKeysFile");
    bool trainTrans = opt<bool>("ulrTrainTransform", false);
    if (!queryFile.empty() && !keyFile.empty()) {
      initFunc = inits::fromWord2vec(queryFile, dimQueries, dimUlrEmb, false);
      name = "ulr_query";
      fixed = true;
      auto query_embed = graph_->param(name, { dimQueries, dimUlrEmb }, initFunc, fixed);
      ulrEmbeddings_.push_back(query_embed);
      // keys embeds
      initFunc = inits::fromWord2vec(keyFile, dimKeys, dimUlrEmb, false);
      name = "ulr_keys";
      fixed = true;
      auto key_embed = graph_->param(name, { dimKeys, dimUlrEmb }, initFunc, fixed);
      ulrEmbeddings_.push_back(key_embed);
      // actual  trainable embedding
      initFunc = inits::glorotUniform();
      name = "ulr_embed";
      fixed = false;
      auto ulr_embed = graph_->param(name, {dimKeys , dimEmb }, initFunc, fixed);  // note the reverse dim
      ulrEmbeddings_.push_back(ulr_embed);
      // init  trainable src embedding
      name = "ulr_src_embed";
      auto ulr_src_embed = graph_->param(name, { dimQueries, dimEmb }, initFunc, fixed);
      ulrEmbeddings_.push_back(ulr_src_embed);
      // ulr transformation matrix
      //initFunc = inits::eye(1.f); // identity matrix  - is it ok to init wiht identity or shall we make this to the fixed case only
      if (trainTrans) {
        initFunc = inits::glorotUniform();
        fixed = false;
      }
      else
      {
        initFunc = inits::eye(); // identity matrix
        fixed = true;
      }
      name = "ulr_transform";
      auto ulrTransform = graph_->param(name, { dimUlrEmb, dimUlrEmb }, initFunc, fixed);
      ulrEmbeddings_.push_back(ulrTransform);

      initFunc = inits::fromValue(1.f);  // TBD: we should read sharable flags here - 1 means all sharable - 0 means no universal embeddings - should be zero for top freq only
      fixed = true;
      name = "ulr_shared";
      auto share_embed = graph_->param(name, { dimQueries, 1 }, initFunc, fixed);
      ulrEmbeddings_.push_back(share_embed);
    }
  }

  std::tuple<Expr/*embeddings*/, Expr/*mask*/> apply(Ptr<data::SubBatch> subBatch) const override final {
    auto queryEmbed   = ulrEmbeddings_[0]; // Q : dimQueries*dimUlrEmb
    auto keyEmbed     = ulrEmbeddings_[1]; // K : dimKeys*dimUlrEmb
    auto uniEmbed     = ulrEmbeddings_[2]; // E : dimQueries*dimEmb
    auto srcEmbed     = ulrEmbeddings_[3]; // I : dimQueries*dimEmb
    auto ulrTransform = ulrEmbeddings_[4]; // A : dimUlrEmb *dimUlrEmb
    auto ulrSharable  = ulrEmbeddings_[5]; // alpha : dimQueries*1
    int dimBatch = (int)subBatch->batchSize();
    int dimEmb = uniEmbed->shape()[-1];
    int dimWords = (int)subBatch->batchWidth();
    // D = K.A.QT
    // dimm(K) = univ_tok_vocab*uni_embed_size
    // dim A = uni_embed_size*uni_embed_size
    // dim Q: uni_embed_size * total_merged_vocab_size
    // dim D = univ_tok_vocab * total_merged_vocab_size
    // note all above can be precombuted and serialized if A is not trainiable and during decoding (TBD)
    // here we need to handle the mini-batch
    // extract raws corresponding to Xs in this minibatch from Q
    auto embIdx = toWordIndexVector(subBatch->data());
    auto queryEmbeddings = rows(queryEmbed, embIdx);
    auto srcEmbeddings = rows(srcEmbed, embIdx);   // extract trainable src embeddings
    auto alpha = rows(ulrSharable, embIdx);  // extract sharable flags
    auto qt = dot(queryEmbeddings, ulrTransform, false, false);  //A: transform embeddings based on similarity A :  dimUlrEmb*dimUlrEmb
    auto sqrtDim=std::sqrt((float)queryEmbeddings->shape()[-1]);
    qt = qt/sqrtDim;  // normalize accordin to embed size to avoid dot prodcut growing large in magnitude with larger embeds sizes
    auto z = dot(qt, keyEmbed, false, true);      // query-key similarity
    float dropProb = this->options_->get<float>("ulr-dropout", 0.0f);  // default no dropout
    z = dropout(z, dropProb);
    float tau = this->options_->get<float>("ulr-softmax-temperature", 1.0f);  // default no temperature
    // temperature in softmax is to control randomness of predictions
    // high temperature Softmax outputs are more close to each other
    // low temperatures the softmax become more similar to  "hardmax"
    auto weights = softmax(z / tau);  // assume default  is dim=-1, what about temprature? - scaler ??
    auto chosenEmbeddings = dot(weights, uniEmbed);  // AVERAGE
    auto chosenEmbeddings_mix = srcEmbeddings + alpha * chosenEmbeddings;  // this should be elementwise  broadcast
    auto batchEmbeddings = reshape(chosenEmbeddings_mix, { dimWords, dimBatch, dimEmb });
    auto graph = ulrEmbeddings_.front()->graph();
    auto batchMask = graph->constant({ dimWords, dimBatch, 1 },
                                     inits::fromVector(subBatch->mask()));
    batchEmbeddings = dropout(batchEmbeddings, options_->get<float>("dropout", 0.0f), {batchEmbeddings->shape()[-3], 1, 1});
    return std::make_tuple(batchEmbeddings, batchMask);
  }

  Expr apply(const Words& words, const Shape& shape) const override final {
    return applyIndices(toWordIndexVector(words), shape);
  }

  Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const override final {
    embIdx; shape;
    ABORT("not implemented"); // @TODO: implement me
  }
};
}  // namespace marian
