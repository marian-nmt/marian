#pragma once

#include "marian.h"
#include "models/states.h"

namespace marian {

class EncoderBase {
protected:
  Ptr<Options> options_;
  std::string prefix_{"encoder"};
  bool inference_{false};
  size_t batchIndex_{0};

  // @TODO: This used to be virtual, but is never overridden.
  // virtual
  std::tuple<Expr, Expr> lookup(Ptr<ExpressionGraph> graph,
    Expr srcEmbeddings,
    Ptr<data::CorpusBatch> batch) const {
    auto subBatch = (*batch)[batchIndex_];
    int dimBatch = (int)subBatch->batchSize();
    int dimEmb = srcEmbeddings->shape()[-1];
    int dimWords = (int)subBatch->batchWidth();
    auto chosenEmbeddings = rows(srcEmbeddings, subBatch->data());
    auto batchEmbeddings = reshape(chosenEmbeddings, { dimWords, dimBatch, dimEmb });
    auto batchMask = graph->constant({ dimWords, dimBatch, 1 },
                                     inits::from_vector(subBatch->mask()));

    return std::make_tuple(batchEmbeddings, batchMask);
  }

  std::tuple<Expr, Expr> ulrLookup(Ptr<ExpressionGraph> graph,
      std::vector<Expr> urlEmbeddings,
    Ptr<data::CorpusBatch> batch) const {
    auto subBatch = (*batch)[batchIndex_];
    // is their a better way to do this?
    assert(urlEmbeddings.size() == 6);
    auto queryEmbed = urlEmbeddings[0]; //Q : dimQueries*dimUlrEmb
    auto keyEmbed = urlEmbeddings[1]; // K  : dimKeys*dimUlrEmb
    auto uniEmbed = urlEmbeddings[2]; // E  : dimQueries*dimEmb
    auto srcEmbed = urlEmbeddings[3]; // I  : dimQueries*dimEmb
    auto ulrTransform = urlEmbeddings[4]; //A   : dimUlrEmb *dimUlrEmb
    auto ulrSharable = urlEmbeddings[5]; //alpha  : dimQueries*1
    int dimBatch = (int)subBatch->batchSize();
    int dimEmb = uniEmbed->shape()[-1];
    int dimWords = (int)subBatch->batchWidth();
    // D = K.A.QT
    // dimm(K) = univ_tok_vocab*uni_embed_size
    // dim A = uni_embed_size*uni_embed_size
    // dim Q: uni_embed_size * total_merged_vocab_size
    // dim D = univ_tok_vocab * total_merged_vocab_size
    // note all above can be precombuted and serialized if A is not trainiabale and during decoding (TBD)
    // here we need to handle the mini-batch
    // extract raws corresponding to Xs in this mini batch from Q
    auto queryEmbeddings = rows(queryEmbed, subBatch->data());
    auto srcEmbeddings = rows(srcEmbed, subBatch->data());   // extract trainable src embeddings
    auto alpha = rows(ulrSharable, subBatch->data());  // extract sharable flags
    auto qt = dot(queryEmbeddings, ulrTransform, false, false);  //A: transform embeddings based on similarity A :  dimUlrEmb *dimUlrEmb
    auto z = dot(qt, keyEmbed, false, true);      // query-key similarity 
    float dropProb = this->options_->get<float>("ulr-dropout", 0.0f);  // default no dropout
    z = dropout(z, dropProb);
    float tau = this->options_->get<float>("ulr-softmax-temperature", 1.0f);  // default no temperature
    // temperature in softmax is to control randomness of predictions
    // high temperature Softmax outputs are more close to each other
    // low temperatures the softmax become more similar to  "hardmax" 
    auto weights = softmax(z / tau);  // assume default  is dim=-1, what about temprature? - scaler ??
    auto chosenEmbeddings = dot(weights, uniEmbed);  // THIS IS WRONG  - IT SHOULD BE AVERAGE 
    auto chosenEmbeddings_mix = srcEmbeddings + alpha * chosenEmbeddings;  // this should be elementwise  broadcast
    auto batchEmbeddings = reshape(chosenEmbeddings_mix, { dimWords, dimBatch, dimEmb });
    auto batchMask = graph->constant({ dimWords, dimBatch, 1 },
                                     inits::from_vector(subBatch->mask()));
    return std::make_tuple(batchEmbeddings, batchMask);
  }
public: 
  EncoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "encoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 0)) {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>)
      = 0;

  template <typename T>
  T opt(const std::string& key) const {
    return options_->get<T>(key);
  }

  virtual void clear() = 0;
};

}  // namespace marian
