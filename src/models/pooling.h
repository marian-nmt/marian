#pragma once

#include "models/encdec.h"
#include "models/s2s.h"

namespace marian {

class EncoderStatePooling : public EncoderState {
private:
  Expr context_;
  Expr contextStart_;
  Expr mask_;
  Ptr<data::CorpusBatch> batch_;

public:
  EncoderStatePooling(Expr context, Expr contextStart, Expr mask, Ptr<data::CorpusBatch> batch)
      : context_(context), contextStart_(contextStart), mask_(mask), batch_(batch) {}

  Expr getContext() { return context_; }
  Expr getContextStart() { return contextStart_; }
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

    int k = 10;

    auto padding = graph->zeros(shape={dimBatch, dimSrcEmb, k / 2});
    auto xpad = concatenate({padding, x, padding}, axis=2);

    int width = xpad->shape()[2];
    std::vector<float> pv(width);
    std::iota(std::begin(pv), std::end(pv), -k / 2);

    auto r = graph->constant({1, 1, width}, init=inits::from_vector(pv));

    std::vector<Expr> means;
    for(int i = 0; i < dimSrcWords; ++i) {
      auto gauss = exp(square(r - i) / -(k / 2.f));

      //std::vector<Expr> preAvg;
      //for(int j = 0; j < k; ++j)
      //  preAvg.push_back(step(xpad, i + j));

      means.push_back(mean(xpad * gauss, axis=2));
    }

    auto xMeans = concatenate(means, axis=2);

    return New<EncoderStatePooling>(x, xMeans, xMask, batch);
  }
};

class DecoderPooling : public DecoderS2S {
public:
  template <class... Args>
  DecoderPooling(Ptr<Config> options, Args... args)
      : DecoderS2S(options, args...) {}

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
  using namespace keywords;

  auto encStatePooling = std::dynamic_pointer_cast<EncoderStatePooling>(encState);
  auto meanContext = weighted_average(
      encStatePooling->getContextStart(), encStatePooling->getMask(), axis = 2);

  bool layerNorm = options_->get<bool>("layer-normalization");
  auto start = Dense(prefix_ + "_ff_state",
                     options_->get<int>("dim-rnn"),
                     activation = act::tanh,
                     normalize = layerNorm)(meanContext);

  // @TODO: review this
  RNNStates startStates;
  for(int i = 0; i < options_->get<size_t>("layers-dec"); ++i)
    startStates.push_back(RNNState{start, start});

  return New<DecoderStateS2S>(startStates, nullptr, encState);
}

};

typedef EncoderDecoder<EncoderPooling, DecoderPooling> Pooling;


}
