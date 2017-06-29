#pragma once

#include "marian.h"

namespace marian {

class EncoderSimple : public EncoderBase {
public:

  template <class... Args>
  EncoderSimple(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t encoderIndex) {
    using namespace keywords;

    // create source embeddings matrix
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[encoderIndex];
    auto embeddings = embedding(graph)
                      ("prefix", prefix_ + "_Wemb")
                      ("dimVocab", dimVoc)
                      ("dimEmb", opt<int>("dim-emb"))
                      .construct();

    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
      = EncoderBase::lookup(embeddings, batch, encoderIndex);

    // create an encoder context based on the embeddings only
    return New<EncoderState>(batchEmbeddings, batchMask, batch);
  }
};

class DecoderSimple : public DecoderBase {

public:
  template <class... Args>
  DecoderSimple(Ptr<Config> options, Args... args)
      : DecoderBase(options, args...) {}

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
    using namespace keywords;

    rnn::States empty;
    return New<DecoderState>(empty, nullptr, encState);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    // average the source context weighted by the batch mask
    // this will remove padded zeros from the average
    auto context = weighted_average(state->getEncoderState()->getContext(),
                                    state->getEncoderState()->getMask(),
                                    axis = 2);

    auto embeddings = state->getTargetEmbeddings();

    rnn::States decoderStates({{embeddings, nullptr}});

    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l1")
                  ("dim", opt<int>("dim-emb"))
                  ("activation", mlp::act::tanh);

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();
    auto layer2 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l2")
                  ("dim", dimTrgVoc);

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto logits = mlp::mlp(graph)
                  .push_back(layer1)
                  .push_back(layer2)
                  ->apply(embeddings, context);

    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits,
                             state->getEncoderState());
  }
};

typedef EncoderDecoder<EncoderSimple, DecoderSimple> Simple;

}
