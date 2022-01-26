#pragma once
#include "generic.h"
#include "marian.h"

namespace marian {

class FactoredVocab;

/**
 * A regular embedding layer.
 * Note that this also applies dropout if the option is passed (pass 0 when in inference mode).
 * It is best to not use Embedding directly, but rather via getEmbeddingLayer() in
 * EncoderDecoderLayerBase, which knows to pass on all required parameters from options.
 */
class Embedding : public LayerBase, public IEmbeddingLayer {
  Expr E_;
  Expr FactorEmbMatrix_; // Factors embedding matrix if combining lemma and factors embeddings with concatenation
  Ptr<FactoredVocab> factoredVocab_;
  Expr multiRows(const Words& data, float dropProb) const;
  Expr embedWithConcat(const Words& data) const;
  bool inference_{false};

public:
  /**
   * Construct a regular embedding layer in the graph.
   * @param graph The expression graph.
   * @param options The options used for this embedding layer.
   */
  Embedding(Ptr<ExpressionGraph> graph, Ptr<Options> options);

  /**
   * Apply/Link this embedding layer (with the given batch of sentences) to the expression graph.
   * @param subBatch The batch of sentences
   * @return The expression tuple holding the embedding layer and the masking layer
   */
  std::tuple<Expr /*embeddings*/, Expr /*mask*/> apply(
      Ptr<data::SubBatch> subBatch) const override final;

  /**
   * Apply/Link this embedding layer (with the given words and shape) to the expression graph.
   * @param words Sequence of vocabulary items
   * @param shape Shape of the words
   * @return The expression holding the embedding layer
   */
  Expr apply(const Words& words, const Shape& shape) const override final;

  /**
   * Apply/Link this embedding layer (with the given WordIndex vector and shape) to the expression graph.
   * @param embIdx The vector of WordIndex objects
   * @param shape Shape of the WordIndex vector
   * @return The expression holding the embedding layer
   */
  Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const override final;
};

/**
 * Universal Language Representation (ULR) word embedding layer.
 * It is under development.
 * @todo applyIndices() is not implemented
 */
class ULREmbedding : public LayerBase, public IEmbeddingLayer {
  std::vector<Expr> ulrEmbeddings_;  // @TODO: These could now better be written as 6 named class members
  bool inference_{false};

public:
  ULREmbedding(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : LayerBase(graph, options), inference_(opt<bool>("inference")) {
    std::string name = "url_embed";  // opt<std::string>("prefix");
    int dimKeys      = opt<int>("dimTgtVoc");
    int dimQueries   = opt<int>("dimSrcVoc");
    int dimEmb       = opt<int>("dimEmb");
    int dimUlrEmb    = opt<int>("dimUlrEmb");  // ULR mono embed size
    bool fixed       = opt<bool>("fixed", false);

    // Embedding layer initialization should depend only on embedding size, hence fanIn=false
    auto initFunc = inits::glorotUniform(/*fanIn=*/false, /*fanOut=*/true);

    std::string queryFile = opt<std::string>("ulrQueryFile");
    std::string keyFile   = opt<std::string>("ulrKeysFile");
    bool trainTrans       = opt<bool>("ulrTrainTransform", false);
    if(!queryFile.empty() && !keyFile.empty()) {
      initFunc         = inits::fromWord2vec(queryFile, dimQueries, dimUlrEmb, false);
      name             = "ulr_query";
      fixed            = true;
      auto query_embed = graph_->param(name, {dimQueries, dimUlrEmb}, initFunc, fixed);
      ulrEmbeddings_.push_back(query_embed);
      // keys embeds
      initFunc       = inits::fromWord2vec(keyFile, dimKeys, dimUlrEmb, false);
      name           = "ulr_keys";
      fixed          = true;
      auto key_embed = graph_->param(name, {dimKeys, dimUlrEmb}, initFunc, fixed);
      ulrEmbeddings_.push_back(key_embed);
      // actual  trainable embedding
      initFunc = inits::glorotUniform();
      name     = "ulr_embed";
      fixed    = false;
      auto ulr_embed = graph_->param(name, {dimKeys, dimEmb}, initFunc, fixed);  // note the reverse dim
      ulrEmbeddings_.push_back(ulr_embed);
      // init  trainable src embedding
      name               = "ulr_src_embed";
      auto ulr_src_embed = graph_->param(name, {dimQueries, dimEmb}, initFunc, fixed);
      ulrEmbeddings_.push_back(ulr_src_embed);
      // ulr transformation matrix
      // initFunc = inits::eye(1.f); // identity matrix  - is it ok to init wiht identity or shall
      // we make this to the fixed case only
      if(trainTrans) {
        initFunc = inits::glorotUniform();
        fixed    = false;
      } else {
        initFunc = inits::eye();  // identity matrix
        fixed    = true;
      }
      name              = "ulr_transform";
      auto ulrTransform = graph_->param(name, {dimUlrEmb, dimUlrEmb}, initFunc, fixed);
      ulrEmbeddings_.push_back(ulrTransform);

      initFunc = inits::fromValue(
          1.f);  // TBD: we should read sharable flags here - 1 means all sharable - 0 means no
                 // universal embeddings - should be zero for top freq only
      fixed            = true;
      name             = "ulr_shared";
      auto share_embed = graph_->param(name, {dimQueries, 1}, initFunc, fixed);
      ulrEmbeddings_.push_back(share_embed);
    }
  }

  std::tuple<Expr /*embeddings*/, Expr /*mask*/> apply(
      Ptr<data::SubBatch> subBatch) const override final {
    auto queryEmbed   = ulrEmbeddings_[0];  // Q : dimQueries*dimUlrEmb
    auto keyEmbed     = ulrEmbeddings_[1];  // K : dimKeys*dimUlrEmb
    auto uniEmbed     = ulrEmbeddings_[2];  // E : dimQueries*dimEmb
    auto srcEmbed     = ulrEmbeddings_[3];  // I : dimQueries*dimEmb
    auto ulrTransform = ulrEmbeddings_[4];  // A : dimUlrEmb *dimUlrEmb
    auto ulrSharable  = ulrEmbeddings_[5];  // alpha : dimQueries*1
    int dimBatch      = (int)subBatch->batchSize();
    int dimEmb        = uniEmbed->shape()[-1];
    int dimWords      = (int)subBatch->batchWidth();
    // D = K.A.QT
    // dimm(K) = univ_tok_vocab*uni_embed_size
    // dim A = uni_embed_size*uni_embed_size
    // dim Q: uni_embed_size * total_merged_vocab_size
    // dim D = univ_tok_vocab * total_merged_vocab_size
    // note all above can be precombuted and serialized if A is not trainiable and during decoding
    // (TBD) here we need to handle the mini-batch extract raws corresponding to Xs in this
    // minibatch from Q
    auto embIdx          = toWordIndexVector(subBatch->data());
    auto queryEmbeddings = rows(queryEmbed, embIdx);
    auto srcEmbeddings   = rows(srcEmbed, embIdx);     // extract trainable src embeddings
    auto alpha           = rows(ulrSharable, embIdx);  // extract sharable flags
    auto qt              = dot(queryEmbeddings, ulrTransform, false, false);  // A: transform embeddings based on similarity A :  dimUlrEmb*dimUlrEmb
    auto sqrtDim         = std::sqrt((float)queryEmbeddings->shape()[-1]);
    qt = qt / sqrtDim;  // normalize accordin to embed size to avoid dot prodcut growing large in
                        // magnitude with larger embeds sizes
    auto z         = dot(qt, keyEmbed, false, true);                   // query-key similarity
    float dropProb = this->options_->get<float>("ulr-dropout", 0.0f);  // default no dropout
    if(!inference_)
      z = dropout(z, dropProb);

    float tau
        = this->options_->get<float>("ulr-softmax-temperature", 1.0f);  // default no temperature
    // temperature in softmax is to control randomness of predictions
    // high temperature Softmax outputs are more close to each other
    // low temperatures the softmax become more similar to  "hardmax"
    auto weights = softmax(z / tau);  // assume default  is dim=-1, what about temprature? - scaler ??
    auto chosenEmbeddings = dot(weights, uniEmbed);  // AVERAGE
    auto chosenEmbeddings_mix = srcEmbeddings + alpha * chosenEmbeddings;  // this should be elementwise  broadcast
    auto batchEmbeddings = reshape(chosenEmbeddings_mix, {dimWords, dimBatch, dimEmb});
    auto graph           = ulrEmbeddings_.front()->graph();
    auto batchMask = graph->constant({dimWords, dimBatch, 1}, inits::fromVector(subBatch->mask()));
    if(!inference_)
      batchEmbeddings = dropout(batchEmbeddings,
                                options_->get<float>("dropout-embeddings", 0.0f),
                                {batchEmbeddings->shape()[-3], 1, 1});
    return std::make_tuple(batchEmbeddings, batchMask);
  }

  Expr apply(const Words& words, const Shape& shape) const override final {
    return applyIndices(toWordIndexVector(words), shape);
  }

  Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const override final {
    embIdx;
    shape;
    ABORT("not implemented");  // @TODO: implement me
  }
};

}  // namespace marian
