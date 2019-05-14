#pragma once

#include "marian.h"
#include "states.h"

#include "data/shortlist.h"
#include "layers/constructors.h"
#include "layers/generic.h"

namespace marian {

class DecoderBase : public EncoderDecoderLayerBase {
protected:
  Ptr<data::Shortlist> shortlist_;

public:
  DecoderBase(Ptr<Options> options) :
    EncoderDecoderLayerBase("decoder", /*batchIndex=*/1, options, /*embeddingFixParamName=*/"embedding-fix-trg") {}

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph>,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  virtual void embeddingsFromBatch(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state,
                                   Ptr<data::CorpusBatch> batch) {
    auto subBatch = (*batch)[batchIndex_];

    Expr y, yMask; std::tie
    (y, yMask) = getEmbeddingLayer(graph)->apply(subBatch);
    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if (dropoutTrg) {
      int trgWords = y->shape()[-3];
      y = dropout(y, dropoutTrg, {trgWords, 1, 1});
    }

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
    auto embeddingLayer = getEmbeddingLayer(graph);
    Expr selectedEmbs;
    int dimEmb = opt<int>("dim-emb");
    if(words.empty()) {
      selectedEmbs = graph->constant({1, 1, dimBatch, dimEmb}, inits::zeros);
    } else {
      selectedEmbs = embeddingLayer->apply(words, {dimBeam, 1, dimBatch, dimEmb});
      // dropout target words   --does not make sense here since this is always inference. Keep it regular though.
      float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
      if (dropoutTrg) {
        int trgWords = selectedEmbs->shape()[-3];
        selectedEmbs = dropout(selectedEmbs, dropoutTrg, { trgWords, 1, 1 });
      }
    }
    state->setTargetHistoryEmbeddings(selectedEmbs);
  }

  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) { return {}; };

  virtual Ptr<data::Shortlist> getShortlist() { return shortlist_; }
  virtual void setShortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
  }

  virtual void clear() = 0;
};

}  // namespace marian
