#pragma once

#include "models/s2s.h"
#include "layers/convolution.h"

namespace marian {

class CharS2SEncoder : public EncoderS2S {
public:
  CharS2SEncoder(Ptr<Options> options) : EncoderS2S(options) {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) {
    auto embeddings = buildSourceEmbeddings(graph);

    using namespace keywords;
    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
        = EncoderBase::lookup(embeddings, batch);

    // apply dropout over source words
    float dropProb = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProb) {
      int srcWords = batchEmbeddings->shape()[-3];
      auto dropMask = graph->dropout(dropProb, {srcWords, 1, 1});
      batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
    }

    int dimEmb = opt<int>("dim-emb");
    auto convSizes = options_->get<std::vector<int>>("char-conv-filters-num");
    auto convWidths = options_->get<std::vector<int>>("char-conv-filters-widths");
    int stride = opt<int>("char-stride");
    int highwayNum = opt<int>("char-highway");

    auto conved = CharConvPooling(
        prefix_ + "conv_pooling",
        dimEmb,
        convWidths,
        convSizes,
        stride)
      (batchEmbeddings, batchMask);

    auto inHighway = conved;
    for (int i = 0; i < highwayNum; ++i) {
      inHighway = highway(prefix_ +"_" + std::to_string(i), inHighway);
    }

    Expr stridedMask = getStridedMask(graph, batch, stride);
    Expr context = applyEncoderRNN(
        graph, inHighway, stridedMask, opt<std::string>("enc-type"));

    return New<EncoderState>(context, stridedMask, batch);
  }
protected:
  Expr getStridedMask(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch,
                      int stride) {
    auto subBatch = (*batch)[batchIndex_];

    int dimBatch = subBatch->batchSize();

    std::vector<float> strided;
    for (size_t wordIdx = 0; wordIdx < subBatch->mask().size(); wordIdx += stride * dimBatch) {
      for (size_t j = wordIdx; j < wordIdx + dimBatch; ++j) {
        strided.push_back(subBatch->mask()[j]);
      }
    }
    int dimWords = strided.size() / dimBatch;
    auto stridedMask = graph->constant({dimWords, dimBatch, 1},
                                       keywords::init = inits::from_vector(strided));
    return stridedMask;
  }
};

}
