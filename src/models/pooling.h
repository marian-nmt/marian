#pragma once

#include "models/encdec.h"
#include "models/s2s.h"

namespace marian {

class EncoderStatePooling : public EncoderState {
private:
  Expr context_;
  Expr attended_;
  Expr mask_;
  Ptr<data::CorpusBatch> batch_;

public:
  EncoderStatePooling(Expr context, Expr attended, Expr mask, Ptr<data::CorpusBatch> batch)
      : context_(context), attended_(attended), mask_(mask), batch_(batch) {}

  Expr getContext() { return context_; }
  Expr getAttended() { return attended_; }
  Expr getMask() { return mask_; }

  virtual const std::vector<size_t>& getSourceWords() {
    return batch_->front()->indeces();
  }
};


class EncoderPooling : public EncoderBase {
public:
  template <class... Args>
  EncoderPooling(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t batchIdx) {
    using namespace keywords;

    int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[batchIdx];
    int dimSrcEmb = options_->get<int>("dim-emb");

    auto xEmb = Embedding(prefix_ + "_Wemb", dimSrcVoc, dimSrcEmb)(graph);

    Expr w, xMask;
    std::tie(w, xMask) = prepareSource(xEmb, batch, batchIdx);

    int dimBatch = w->shape()[0];
    int dimSrcWords = w->shape()[2];

    int dimMaxLength = options_->get<size_t>("max-length") + 1;
    auto pEmb = Embedding(prefix_ + "_Pemb", dimMaxLength, dimSrcEmb)(graph);

    std::vector<size_t> pIndices;
    for(int i = 0; i < dimSrcWords; ++i)
      for(int j = 0; j < dimBatch; j++)
        pIndices.push_back(i);

    auto p = reshape(rows(pEmb, pIndices), {dimBatch, dimSrcEmb, dimSrcWords});
    auto x = w + p;

    int k = 5;

    auto padding = graph->zeros(shape={dimBatch, dimSrcEmb, k / 2});
    auto xpad = concatenate({padding, x, padding}, axis=2);

    std::vector<Expr> means;
    for(int i = 0; i < dimSrcWords; ++i) {
      std::vector<Expr> preAvg;
      for(int j = 0; j < k; ++j)
        preAvg.push_back(step(xpad, i + j));

      means.push_back(mean(concatenate(preAvg, axis=2), axis=2));
    }
    auto xMeans = concatenate(means, axis=2);

    return New<EncoderStatePooling>(xMeans, x, xMask, batch);
  }
};

typedef EncoderDecoder<EncoderPooling, DecoderS2S> Pooling;


}
