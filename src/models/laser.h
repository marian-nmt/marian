#pragma once

#include "marian.h"

#include "layers/constructors.h"
#include "rnn/constructors.h"

namespace marian {

// Re-implements the LASER BiLSTM encoder from:
// Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond
// Mikel Artetxe, Holger Schwenk
// https://arxiv.org/abs/1812.10464

class EncoderLaser : public EncoderBase {
  using EncoderBase::EncoderBase;

public:
  Expr applyEncoderRNN(Ptr<ExpressionGraph> graph,
                       Expr embeddings,
                       Expr mask) {
    int depth = opt<int>("enc-depth");    
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");

    Expr output = embeddings;

    auto applyRnn = [&](int layer, rnn::dir direction, Expr input, Expr mask) {

        std::string paramPrefix = prefix_ + "_" + opt<std::string>("enc-cell");
        paramPrefix += "_l" + std::to_string(layer);
        if(direction == rnn::dir::backward)
            paramPrefix += "_reverse";

        auto rnnFactory = rnn::rnn()
            ("type", opt<std::string>("enc-cell"))
            ("direction", (int)direction)
            ("dimInput", input->shape()[-1])
            ("dimState", opt<int>("dim-rnn"))
            ("dropout", dropoutRnn)
            ("layer-normalization", opt<bool>("layer-normalization"))
            ("skip", opt<bool>("skip"))
            .push_back(rnn::cell()("prefix", paramPrefix));

        return rnnFactory.construct(graph)->transduce(input, mask);
    };

    for(int i = 0; i < depth; ++i) {
        output = concatenate({applyRnn(i, rnn::dir::forward, output, mask),
                              applyRnn(i, rnn::dir::backward, output, mask)},
                              /*axis =*/ -1);
    }

    return output;
  }

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) override {
    graph_ = graph;
    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask; std::tie
    (batchEmbeddings, batchMask) = getEmbeddingLayer()->apply((*batch)[batchIndex_]);

    Expr context = applyEncoderRNN(graph_, batchEmbeddings, batchMask);

    return New<EncoderState>(context, batchMask, batch);
  }

  void clear() override {}
};

}