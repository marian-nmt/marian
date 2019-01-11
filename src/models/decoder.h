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

  Ptr<data::Shortlist> shortlist_;

public:
  DecoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "decoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 1)) {}

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph>,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  std::vector<Ptr<IEmbeddingLayer>> embedding_; // @TODO: move away, also rename
  virtual void embeddingsFromBatch(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state,
                                   Ptr<data::CorpusBatch> batch) {

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    int dimEmb = opt<int>("dim-emb");

    // @TODO: code dup with EncoderTransformer
    if (embedding_.empty() || !embedding_[batchIndex_]) { // lazy
      embedding_.resize(batch->sets());
      auto embFactory = embedding()("dimVocab", dimVoc)("dimEmb", dimEmb);
      if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
        embFactory("prefix", "Wemb");
      else
        embFactory("prefix", prefix_ + "_Wemb");
      if(options_->has("embedding-fix-trg"))
        embFactory("fixed", opt<bool>("embedding-fix-trg"));
      if(options_->has("embedding-vectors")) {
        auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
        embFactory("embFile", embFiles[batchIndex_])  //
            ("normalization", opt<bool>("embedding-normalization"));
      }
      if (options_->has("embedding-factors")) {
        embFactory("embedding-factors", opt<std::vector<std::string>>("embedding-factors"));
        embFactory("vocab", opt<std::vector<std::string>>("vocabs")[batchIndex_]);
      }
      embedding_[batchIndex_] = embFactory.construct(graph);
    }

    auto subBatch = (*batch)[batchIndex_];

    Expr y, yMask; std::tie
    (y, yMask) = embedding_[batchIndex_]->apply(subBatch);

    Expr yData;
    if(shortlist_) {
      yData = graph->indices(shortlist_->mappedIndices());
    } else {
      yData = graph->indices(subBatch->data());
    }

    auto yShifted = shift(y, {1, 0, 0});

    state->setTargetEmbeddings(yShifted);
    state->setTargetMask(yMask);
    state->setTargetIndices(yData);
  }

  virtual void embeddingsFromPrediction(Ptr<ExpressionGraph> graph,
                                        Ptr<DecoderState> state,
                                        const std::vector<IndexType>& embIdx,
                                        int dimBatch,
                                        int dimBeam) {
    int dimTrgEmb = opt<int>("dim-emb");
    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    Expr selectedEmbs;
    if(embIdx.empty()) {
      selectedEmbs = graph->constant({1, 1, dimBatch, dimTrgEmb}, inits::zeros);
    } else {
      // embeddings are loaded from model during translation, no fixing required
      auto yEmbFactory = embedding()  //
          ("dimVocab", dimTrgVoc)     //
          ("dimEmb", dimTrgEmb);
  
      if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
        yEmbFactory("prefix", "Wemb");
      else
        yEmbFactory("prefix", prefix_ + "_Wemb");
  
      auto yEmb = yEmbFactory.construct(graph);

      selectedEmbs = yEmb->apply(embIdx, dimBatch, dimBeam);
    }
    state->setTargetEmbeddings(selectedEmbs);
  }

  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) { return {}; };

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
