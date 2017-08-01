#pragma once

#include "marian.h"

namespace marian {

class DummyEncoder : public EncoderBase {
public:

  template <class... Args>
  DummyEncoder(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t encoderIndex) {
    using namespace keywords;

    int dimBatch = batch->size();
    int dimRnn = opt<int>("dim-rnn");

    Expr context = graph->constant({dimBatch, dimRnn}, init=inits::zeros);
    size_t decoderLayers = opt<size_t>("dec-depth");
    size_t decoderHighDepth = opt<size_t>("dec-cell-high-depth");

    return New<EncoderState>(context, nullptr, batch);
  }
};

class DecoderLM : public DecoderBase {
private:
  Ptr<rnn::RNN> rnn_;
  Expr tiedOutputWeights_;

Ptr<rnn::RNN> constructDecoderRNN(Ptr<ExpressionGraph> graph,
                                  Ptr<DecoderState> state) {
  float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
  auto rnn = rnn::rnn(graph)
             ("type", opt<std::string>("dec-cell"))
             ("dimInput", opt<int>("dim-emb"))
             ("dimState", opt<int>("dim-rnn"))
             ("dropout", dropoutRnn)
             ("layer-normalization", opt<bool>("layer-normalization"))
             ("skip", opt<bool>("skip"));

  size_t decoderLayers = opt<size_t>("dec-depth");
  size_t decoderBaseDepth = opt<size_t>("dec-cell-base-depth");
  size_t decoderHighDepth = opt<size_t>("dec-cell-high-depth");

  // setting up conditional (transitional) cell
  auto baseCell = rnn::stacked_cell(graph);
  for(int i = 1; i <= decoderBaseDepth; ++i) {
    auto paramPrefix = prefix_ + "_cell" + std::to_string(i);
    baseCell.push_back(rnn::cell(graph)
                       ("prefix", paramPrefix)
                       ("final", i > 1));
  }
  // Add cell to RNN (first layer)
  rnn.push_back(baseCell);

  // Add more cells to RNN (stacked RNN)
  for(int i = 2; i <= decoderLayers; ++i) {
    // deep transition
    auto highCell = rnn::stacked_cell(graph);

    for(int j = 1; j <= decoderHighDepth; j++) {
      auto paramPrefix = prefix_ + "_l" + std::to_string(i) + "_cell" + std::to_string(j);
      highCell.push_back(rnn::cell(graph)
                         ("prefix", paramPrefix));
    }

    // Add cell to RNN (more layers)
    rnn.push_back(highCell);
  }

  return rnn.construct();
}
public:
  template <class... Args>
  DecoderLM(Ptr<Config> options, Args... args)
      : DecoderBase(options, args...) {}

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
    using namespace keywords;

    // average the source context weighted by the batch mask
    // this will remove padded zeros from the average
    auto start = encState->getContext();
    auto graph = start->graph();

    rnn::States startStates(opt<size_t>("dec-depth"), {start, start});
    return New<DecoderState>(startStates, nullptr, encState);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[2];
      auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
      embeddings = dropout(embeddings, mask = trgWordDrop);
    }

    if(!rnn_)
      rnn_ = constructDecoderRNN(graph, state);

    // apply RNN to embeddings, initialized with encoder context mapped into
    // decoder space
    auto decoderContext = rnn_->transduce(embeddings, state->getStates());

    // retrieve the last state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States decoderStates = rnn_->lastCellStates();

    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l1")
                  ("dim", opt<int>("dim-emb"))
                  ("activation", mlp::act::tanh)
                  ("layer-normalization", opt<bool>("layer-normalization"));

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();

    auto layer2 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l2")
                  ("dim", dimTrgVoc);
    if(opt<bool>("tied-embeddings"))
      layer2.tie_transposed("W", prefix_ + "_Wemb");

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto logits = mlp::mlp(graph)
                  .push_back(layer1)
                  .push_back(layer2)
                  ->apply(embeddings, decoderContext);

    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderState());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments() {
    return {};
  }
};

class LM : public EncoderDecoder<DummyEncoder, DecoderLM> {
public:
  template <class... Args>
  LM(Ptr<Config> options, Args... args)
      : EncoderDecoder(options, {0}, args...) {}
};

}
