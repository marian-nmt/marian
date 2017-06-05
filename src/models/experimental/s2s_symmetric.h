#pragma once

#include "models/s2s.h"

namespace marian {

template <class EncDec>
class EncoderDecoderSymmetric : public EncDec {
private:
  Ptr<EncDec> inverse_;

public:
  template <class... Args>
  EncoderDecoderSymmetric(Ptr<Config> options, Args... args)
      : EncDec(options, std::vector<size_t>{0, 1}, args...),
        inverse_(New<EncDec>(options,
                             std::vector<size_t>{1, 0},
                             keywords::prefix = "inverse_",
                             args...)) {}

  virtual void clear(Ptr<ExpressionGraph> graph) {
    EncDec::clear(graph);
    inverse_->clear(graph);
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true) {
    using namespace keywords;

    if(EncDec::inference_) {
      return EncDec::build(graph, batch, clearGraph);
    } else {
      auto cost1 = EncDec::build(graph, batch, clearGraph);
      auto cost2 = inverse_->build(graph, batch, false);

      auto dec1 = std::dynamic_pointer_cast<DecoderS2S>(EncDec::decoder_);
      auto dec2 = std::dynamic_pointer_cast<DecoderS2S>(inverse_->getDecoder());

      auto srcA = concatenate(dec1->getAlignments(), axis = 3);
      auto trgA = concatenate(dec2->getAlignments(), axis = 3);

      int dimBatch = srcA->shape()[0];
      int dimSrc = srcA->shape()[2];
      int dimTrg = srcA->shape()[3];

      std::vector<size_t> reorder;
      for(int j = 0; j < dimTrg; ++j)
        for(int i = 0; i < dimSrc; ++i)
          reorder.push_back(i * dimTrg + j);

      if(dimBatch == 1) {
        srcA = reshape(srcA, {dimTrg, dimSrc});
        trgA = reshape(trgA, {dimSrc, dimTrg});
      } else {
        srcA = reshape(srcA, {dimSrc, dimBatch, dimTrg});
        trgA = reshape(trgA, {dimTrg, dimBatch, dimSrc});
      }
      debug(srcA, "srcA");
      debug(trgA, "trgA");

      int dimSrcTrg = dimSrc * dimTrg;
      trgA = rows(reshape(trgA, {dimSrcTrg, dimBatch}), reorder);

      auto symmetricAlignment
          = mean(scalar_product(reshape(srcA, {dimBatch, 1, dimSrcTrg}),
                                reshape(trgA, {dimBatch, 1, dimSrcTrg}),
                                axis = 2),
                 axis = 0);
      return cost1 + cost2 - symmetricAlignment;
    }
  }
};

typedef EncoderDecoderSymmetric<S2S> SymmetricS2S;
}
