#pragma once

#include "marian.h"
#include "states.h"

#include "data/shortlist.h"
#include "layers/constructors.h"
#include "layers/generic.h"

namespace marian {

class DecoderBase {
protected:
  Ptr<Options> options_;
  std::string prefix_{"decoder"};
  bool inference_{false};
  size_t batchIndex_{1};
  std::vector<Ptr<IEmbeddingLayer>> embeddingLayers_; // (lazily created)

  Ptr<data::Shortlist> shortlist_;

public:
  DecoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "decoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 1)) {}

  virtual ~DecoderBase() {}

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph>,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  void lazyCreateEmbeddingLayer(Ptr<ExpressionGraph> graph) {
    // @TODO: code dup with EncoderTransformer
    if (embeddingLayers_.size() <= batchIndex_ || !embeddingLayers_[batchIndex_]) { // lazy
      if (embeddingLayers_.size() <= batchIndex_)
        embeddingLayers_.resize(batchIndex_ + 1);
      int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
      int dimEmb = opt<int>("dim-emb");
      auto embFactory = embedding()("dimVocab", dimVoc)("dimEmb", dimEmb);
      if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
        embFactory("prefix", "Wemb");
      else
        embFactory("prefix", prefix_ + "_Wemb");
      if(options_->has("embedding-fix-trg"))
        embFactory("fixed", opt<bool>("embedding-fix-trg"));
      if(options_->hasAndNotEmpty("embedding-vectors")) {
        auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
        embFactory("embFile", embFiles[batchIndex_])  //
            ("normalization", opt<bool>("embedding-normalization"));
      }
      embFactory("vocab", opt<std::vector<std::string>>("vocabs")[batchIndex_]); // for factored embeddings
      embeddingLayers_[batchIndex_] = embFactory.construct(graph);
    }
  }

  virtual void embeddingsFromBatch(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state,
                                   Ptr<data::CorpusBatch> batch) {
    auto subBatch = (*batch)[batchIndex_];

    lazyCreateEmbeddingLayer(graph);
    Expr y, yMask; std::tie
    (y, yMask) = embeddingLayers_[batchIndex_]->apply(subBatch);

    const Words& data =
      /*if*/ (shortlist_) ?
        shortlist_->mappedIndices()
      /*else*/ :
        subBatch->data();
    Expr yData = graph->indices(toWordIndexVector(data));

    auto yShifted = shift(y, {1, 0, 0});

    state->setTargetHistoryEmbeddings(yShifted);
    state->setTargetMask(yMask);
    state->setTargetWords(data);
  }

  virtual void embeddingsFromPrediction(Ptr<ExpressionGraph> graph,
                                        Ptr<DecoderState> state,
                                        const Words& words,
                                        int dimBatch,
                                        int dimBeam) {
    lazyCreateEmbeddingLayer(graph);
    Expr selectedEmbs;
    int dimEmb = opt<int>("dim-emb");
    if(words.empty()) {
      selectedEmbs = graph->constant({1, 1, dimBatch, dimEmb}, inits::zeros);
    } else {
      selectedEmbs = embeddingLayers_[batchIndex_]->apply(words, {dimBeam, 1, dimBatch, dimEmb});
    }
    state->setTargetHistoryEmbeddings(selectedEmbs);
  }

  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) { return {}; }; // [tgt index][beam depth, max src length, batch size, 1]

  virtual Ptr<data::Shortlist> getShortlist() { return shortlist_; }
  virtual void setShortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
  }

  template <typename T>
  T opt(const std::string& key) const {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string& key, const T& def) {
    return options_->get<T>(key, def);
  }

  virtual void clear() = 0;
};

}  // namespace marian
