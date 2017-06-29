#pragma once

#include "marian.h"

namespace marian {

class EncoderSutskever : public EncoderBase {
public:

  template <class... Args>
  EncoderSutskever(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t encoderIndex) {
    using namespace keywords;

    // create source embeddings
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

    // backward RNN for encoding
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnnBw = rnn::rnn(graph)
                 ("type", "lstm")
                 ("prefix", prefix_)
                 ("direction", rnn::dir::backward)
                 ("dimInput", opt<int>("dim-emb"))
                 ("dimState", opt<int>("dim-rnn"))
                 ("dropout", dropoutRnn)
                 ("layer-normalization", opt<bool>("layer-normalization"))
                 .push_back(rnn::cell(graph))
                 .construct();

    auto context = rnnBw->transduce(batchEmbeddings, batchMask);

    return New<EncoderState>(context, batchMask, batch);
  }
};

class DecoderSutskever : public DecoderBase {
public:
  template <class... Args>
  DecoderSutskever(Ptr<Config> options, Args... args)
      : DecoderBase(options, args...) {}

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
    using namespace keywords;

    // Use first encoded word as start state
    auto start = marian::step(encState->getContext(), 0);

    rnn::States startStates({{start, start}});
    return New<DecoderState>(startStates, nullptr, encState);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();

    // forward RNN for decoder
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnn = rnn::rnn(graph)
               ("type", "lstm")
               ("prefix", prefix_)
               ("dimInput", opt<int>("dim-emb"))
               ("dimState", opt<int>("dim-rnn"))
               ("dropout", dropoutRnn)
               ("layer-normalization", opt<bool>("layer-normalization"))
               .push_back(rnn::cell(graph))
               .construct();

    // apply RNN to embeddings, initialized with encoder context mapped into
    // decoder space
    auto decoderContext = rnn->transduce(embeddings, state->getStates());

    // retrieve the last state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States decoderStates = rnn->lastCellStates();

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
                  ->apply(embeddings, decoderContext);

    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderState());
  }

};

typedef EncoderDecoder<EncoderSutskever, DecoderSutskever> Sutskever;

}
