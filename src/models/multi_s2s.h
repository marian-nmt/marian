#pragma once

#include "marian.h"
#include "models/s2s.h"

namespace marian {

struct EncoderStateMultiS2S : public EncoderState {
  EncoderStateMultiS2S(Ptr<EncoderState> e1, Ptr<EncoderState> e2)
      : enc1(e1), enc2(e2) {}

  virtual Expr getContext() { return enc1->getContext(); }
  virtual Expr getMask() { return enc2->getMask(); }

  Ptr<EncoderState> enc1;
  Ptr<EncoderState> enc2;

  virtual const std::vector<size_t>& getSourceWords() {
    return enc1->getSourceWords();
  }
};

typedef DecoderState DecoderStateMultiS2S;

template <class Encoder1, class Encoder2>
class MultiEncoder : public EncoderBase {
private:
  Ptr<Encoder1> encoder1_;
  Ptr<Encoder2> encoder2_;

public:
  template <class... Args>
  MultiEncoder(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...),
        encoder1_(
            New<Encoder1>(options, keywords::prefix = "encoder1", args...)),
        encoder2_(
            New<Encoder2>(options, keywords::prefix = "encoder2", args...)) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t batchIdx = 0) {
    auto encState1 = encoder1_->build(graph, batch, 0);
    auto encState2 = encoder2_->build(graph, batch, 1);

    return New<EncoderStateMultiS2S>(encState1, encState2);
  }
};

class MultiDecoderS2S : public DecoderBase {
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
    if(i == 1) {
      auto mEncState = std::static_pointer_cast<EncoderStateMultiS2S>(
        state->getEncoderState());

      baseCell.push_back(rnn::attention(graph)
                         ("prefix", prefix_ + "_att1")
                         .set_state(mEncState->enc1));
      baseCell.push_back(rnn::attention(graph)
                         ("prefix", prefix_ + "_att2")
                         .set_state(mEncState->enc2));
    }
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
  MultiDecoderS2S(Ptr<Config> options, Args... args)
      : DecoderBase(options, args...) {}

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
    using namespace keywords;

    auto mEncState = std::static_pointer_cast<EncoderStateMultiS2S>(encState);

    auto meanContext1 = weighted_average(
        mEncState->enc1->getContext(), mEncState->enc1->getMask(), axis = 2);

    auto meanContext2 = weighted_average(
        mEncState->enc2->getContext(), mEncState->enc2->getMask(), axis = 2);

    auto graph = meanContext1->graph();

    // apply single layer network to mean to map into decoder space
    auto mlp = mlp::mlp(graph)
               .push_back(mlp::dense(graph)
                          ("prefix", prefix_ + "_ff_state")
                          ("dim", opt<int>("dim-rnn"))
                          ("activation", mlp::act::tanh)
                          ("layer-normalization", opt<bool>("layer-normalization")));
    auto start = mlp->apply(meanContext1, meanContext2);

    rnn::States startStates(opt<size_t>("dec-depth"), {start, start});
    return New<DecoderState>(startStates, nullptr, mEncState);
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

    // retrieve all the aligned contexts computed by the attention mechanism
    auto att1 = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    auto alignedContext1 = att1->getContext();

    auto att2 = rnn_->at(0)->as<rnn::StackedCell>()->at(2)->as<rnn::Attention>();
    auto alignedContext2 = att2->getContext();

    auto alignedContext = concatenate({alignedContext1,
                                       alignedContext2},
                                       axis=1);

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
                  ->apply(embeddings, decoderContext, alignedContext);

    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderState());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments() {
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    return att->getAlignments();
  }
};

typedef MultiEncoder<EncoderS2S, EncoderS2S> MultiEncoderS2S;

class MultiS2S : public EncoderDecoder<MultiEncoderS2S, MultiDecoderS2S> {
public:
  template <class... Args>
  MultiS2S(Ptr<Config> options, Args... args)
      : EncoderDecoder(options, {0, 1, 2}, args...) {}
};

}
