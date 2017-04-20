#pragma once

#include "models/s2s.h"
#include "models/encdec.h"

namespace marian {

struct EncoderStateMultiS2S : public EncoderState {
  EncoderStateMultiS2S(Ptr<EncoderState> e1, Ptr<EncoderState> e2)
  : enc1(e1), enc2(e2) {}
  
  virtual Expr getContext() { return nullptr; }
  virtual Expr getMask() { return nullptr; }

  Ptr<EncoderState> enc1;
  Ptr<EncoderState> enc2;
};

typedef DecoderStateS2S DecoderStateMultiS2S;

template <class Encoder1, class Encoder2>
class MultiEncoder : public EncoderBase {
  private:
    Ptr<Encoder1> encoder1_;
    Ptr<Encoder2> encoder2_;

  public:
    template <class ...Args>
    MultiEncoder(Ptr<Config> options, Args ...args)
     : EncoderBase(options, args...),
       encoder1_(New<Encoder1>(options, keywords::prefix="encoder1", args...)),
       encoder2_(New<Encoder2>(options, keywords::prefix="encoder2", args...)) {}

    Ptr<EncoderState>
    build(Ptr<ExpressionGraph> graph,
          Ptr<data::CorpusBatch> batch,
          size_t batchIdx = 0) {

      auto encState1 = encoder1_->build(graph, batch, 0);
      auto encState2 = encoder2_->build(graph, batch, 1);

      return New<EncoderStateMultiS2S>(encState1, encState2);
    }
};

template <class Cell1, class Attention1, class Attention2, class Cell2>
class MultiAttentionCell {
  private:
    Ptr<Cell1> cell1_;
    Ptr<Cell2> cell2_;
    Ptr<Attention1> att1_;
    Ptr<Attention2> att2_;

  public:

    template <class ...Args>
    MultiAttentionCell(Ptr<ExpressionGraph> graph,
                  const std::string prefix,
                  int dimInput,
                  int dimState,
                  Ptr<Attention1> att1,
                  Ptr<Attention2> att2,
                  Args ...args)
    {
      cell1_ = New<Cell1>(graph,
                          prefix + "_cell1",
                          dimInput,
                          dimState,
                          keywords::final=false,
                          args...);

      att1_ = New<Attention1>(att1);
      att2_ = New<Attention2>(att2);

      cell2_ = New<Cell2>(graph,
                          prefix + "_cell2",
                          att1_->outputDim() + att2_->outputDim(),
                          dimState,
                          keywords::final=true,
                          args...);
    }

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      return apply2(apply1(input), state, mask);
    }

    Expr apply1(Expr input) {
      return cell1_->apply1(input);
    }

    Expr apply2(Expr xW, Expr state, Expr mask = nullptr) {
      auto hidden = cell1_->apply2(xW, state, mask);

      auto alignedSourceContext1 = att1_->apply(hidden);
      auto alignedSourceContext2 = att2_->apply(hidden);

      auto alignedSourceContext = concatenate({alignedSourceContext1, alignedSourceContext2},
                                              keywords::axis=1);

      return cell2_->apply(alignedSourceContext, hidden, mask);
    }

    Ptr<Attention1> getAttention1() {
      return att1_;
    }

    Ptr<Attention2> getAttention2() {
      return att2_;
    }

    Expr getContexts() {
      auto context1 = concatenate(att1_->getContexts(), keywords::axis=2);
      auto context2 = concatenate(att2_->getContexts(), keywords::axis=2);

      return concatenate({context1, context2}, keywords::axis=1);
    }

    Expr getLastContext() {
      return concatenate({att1_->getContexts().back(),
                          att2_->getContexts().back()},
                         keywords::axis=1);
    }
};

typedef MultiAttentionCell<GRU, GlobalAttention, GlobalAttention, GRU> MultiCGRU;

class MultiDecoderS2S : public DecoderBase {
  private:
    Ptr<GlobalAttention> attention1_;
    Ptr<GlobalAttention> attention2_;

  public:

    template <class ...Args>
    MultiDecoderS2S(Ptr<Config> options, Args ...args)
     : DecoderBase(options, args...) {}

    virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
      using namespace keywords;

      auto mEncState = std::static_pointer_cast<EncoderStateMultiS2S>(encState);

      auto meanContext1 = weighted_average(mEncState->enc1->getContext(),
                                           mEncState->enc1->getMask(),
                                           axis=2);

      auto meanContext2 = weighted_average(mEncState->enc2->getContext(),
                                           mEncState->enc2->getMask(),
                                           axis=2);

      bool layerNorm = options_->get<bool>("layer-normalization");

      auto start = Dense("ff_state",
                         options_->get<int>("dim-rnn"),
                         activation=act::tanh,
                         normalize=layerNorm)(meanContext1, meanContext2);
      
      std::vector<Expr> startStates(options_->get<size_t>("layers-dec"), start);
      return New<DecoderStateMultiS2S>(startStates, nullptr, mEncState);
    }

    virtual Ptr<DecoderState> step(Expr embeddings,
                                   Ptr<DecoderState> state,
                                   bool single) {
      using namespace keywords;

      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      int dimTrgEmb = options_->get<int>("dim-emb");
      int dimDecState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("layer-normalization");
      bool skipDepth = options_->get<bool>("skip");
      size_t decoderLayers = options_->get<size_t>("layers-dec");

      float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
      float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

      auto graph = embeddings->graph();

      if(dropoutTrg) {
        int trgWords = embeddings->shape()[2];
        auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
        embeddings = dropout(embeddings, mask=trgWordDrop);
      }

      auto mEncState
        = std::static_pointer_cast<EncoderStateMultiS2S>(state->getEncoderState());

      if(!attention1_)
        attention1_ = New<GlobalAttention>("decoder_att1",
                                           mEncState->enc1,
                                           dimDecState,
                                           normalize=layerNorm);
      if(!attention2_)
        attention2_ = New<GlobalAttention>("decoder_att2",
                                           mEncState->enc2,
                                           dimDecState,
                                           normalize=layerNorm);

      RNN<MultiCGRU> rnnL1(graph, "decoder",
                           dimTrgEmb, dimDecState,
                           attention1_, attention2_,
                           normalize=layerNorm,
                           dropout_prob=dropoutRnn);

      auto decState = std::dynamic_pointer_cast<DecoderStateMultiS2S>(state);
      auto stateL1 = rnnL1(embeddings, decState->getStates()[0]);

      auto alignedContext = single ?
        rnnL1.getCell()->getLastContext() :
        rnnL1.getCell()->getContexts();

      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr outputLn;
      if(decoderLayers > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < decState->getStates().size(); ++i)
          statesIn.push_back(decState->getStates()[i]);

        std::vector<Expr> statesLn;
        std::tie(outputLn, statesLn) = MLRNN<GRU>(graph, "decoder",
                                                  decoderLayers - 1,
                                                  dimDecState, dimDecState,
                                                  normalize=layerNorm,
                                                  dropout_prob=dropoutRnn,
                                                  skip=skipDepth,
                                                  skip_first=skipDepth)
                                                 (stateL1, statesIn);

        statesOut.insert(statesOut.end(),
                         statesLn.begin(), statesLn.end());
      }
      else {
        outputLn = stateL1;
      }

      //// 2-layer feedforward network for outputs and cost
      auto logitsL1 = Dense("ff_logit_l1", dimTrgEmb,
                            activation=act::tanh,
                            normalize=layerNorm)
                        (embeddings, outputLn, alignedContext);

      auto logitsOut = Dense("ff_logit_l2", dimTrgVoc)(logitsL1);

      if(lexProbs_)
        logitsOut = LexicalBias(lexProbs_->getLf(),
                                rnnL1.getCell()->getAttention1(),
                                1e-3, single)(logitsOut);
          
      return New<DecoderStateMultiS2S>(statesOut, logitsOut,
                                       state->getEncoderState());
    }

};

typedef MultiEncoder<EncoderS2S, EncoderS2S> MultiEncoderS2S;
typedef EncoderDecoder<MultiEncoderS2S, MultiDecoderS2S> MultiS2S;

}
