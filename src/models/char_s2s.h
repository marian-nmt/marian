#pragma once

#include "marian.h"

#include "layers/convolution.h"
#include "models/s2s.h"

namespace marian {

class CharS2SEncoder : public EncoderS2S {
  using EncoderS2S::EncoderS2S;
  
public:
  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) override {
    graph_ = graph;
    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask; std::tie
    (batchEmbeddings, batchMask) = getEmbeddingLayer()->apply(batch->front());

    int dimEmb = opt<int>("dim-emb");
    auto convSizes = options_->get<std::vector<int>>("char-conv-filters-num");
    auto convWidths
        = options_->get<std::vector<int>>("char-conv-filters-widths");
    int stride = opt<int>("char-stride");
    int highwayNum = opt<int>("char-highway");

    auto conved = CharConvPooling(
        prefix_ + "conv_pooling", dimEmb, convWidths, convSizes, stride)(
        batchEmbeddings, batchMask);

    auto inHighway = conved;
    for(int i = 0; i < highwayNum; ++i) {
      inHighway = highway(prefix_ + "_" + std::to_string(i), inHighway);
    }

    Expr stridedMask = getStridedMask(graph, batch, stride);
    Expr context = applyEncoderRNN(
        graph, inHighway, stridedMask, opt<std::string>("enc-type"));

    return New<EncoderState>(context, stridedMask, batch);
  }

protected:
  Expr getStridedMask(Ptr<ExpressionGraph> graph,
                      Ptr<data::CorpusBatch> batch,
                      int stride) {
    auto subBatch = (*batch)[batchIndex_];

    size_t dimBatch = subBatch->batchSize();

    std::vector<float> strided;
    for(size_t wordIdx = 0; wordIdx < subBatch->mask().size();
        wordIdx += stride * dimBatch) {
      for(size_t j = wordIdx; j < wordIdx + dimBatch; ++j) {
        strided.push_back(subBatch->mask()[j]);
      }
    }
    size_t dimWords = strided.size() / dimBatch;
    auto stridedMask
        = graph->constant({(int)dimWords, (int)dimBatch, 1}, inits::fromVector(strided));
    return stridedMask;
  }
};
}  // namespace marian
