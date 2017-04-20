#pragma once

#include "models/encdec.h"
#include "layers/attention.h"

namespace marian {

typedef AttentionCell<GRU, GlobalAttention, GRU> CGRU;

class EncoderStateS2S : public EncoderState {
  private:
    Expr context_;
    Expr mask_;
    
  public:
    EncoderStateS2S(Expr context, Expr mask)
    : context_(context), mask_(mask) {}
    
    Expr getContext() { return context_; }
    Expr getMask() { return mask_; }
};

class DecoderStateS2S : public DecoderState {
  private:
    std::vector<Expr> states_;
    Expr probs_;
    Ptr<EncoderState> encState_;
    
  public:
    DecoderStateS2S(const std::vector<Expr> states,
                     Expr probs,
                     Ptr<EncoderState> encState)
    : states_(states), probs_(probs), encState_(encState) {}
    
    
    Ptr<EncoderState> getEncoderState() { return encState_; }
    Expr getProbs() { return probs_; }
    void setProbs(Expr probs) { probs_ = probs; }
    
    Ptr<DecoderState> select(const std::vector<size_t>& selIdx) {
      int numSelected = selIdx.size();
      int dimState = states_[0]->shape()[1];
      
      std::vector<Expr> selectedStates;
      for(auto state : states_) {
        selectedStates.push_back(
          reshape(rows(state, selIdx),
                  {1, dimState, 1, numSelected})
        );
      }
      
      return New<DecoderStateS2S>(selectedStates, probs_, encState_);
    }

    
    const std::vector<Expr>& getStates() { return states_; }
};

class EncoderS2S : public EncoderBase {
  public:
    template <class ...Args>
    EncoderS2S(Ptr<Config> options, Args... args)
     : EncoderBase(options, args...) {}

    Ptr<EncoderState>
    build(Ptr<ExpressionGraph> graph,
          Ptr<data::CorpusBatch> batch,
          size_t batchIdx = 0) {

      using namespace keywords;

      int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[batchIdx];
      int dimSrcEmb = options_->get<int>("dim-emb");
      int dimEncState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("layer-normalization");
      bool skipDepth = options_->get<bool>("skip");
      size_t encoderLayers = options_->get<size_t>("layers-enc");

      float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
      float dropoutSrc = inference_ ? 0 : options_->get<float>("dropout-src");

      auto xEmb = Embedding(prefix_ + "_Wemb", dimSrcVoc, dimSrcEmb)(graph);

      Expr x, xMask;
      std::tie(x, xMask) = prepareSource(xEmb, batch, batchIdx);

      if(dropoutSrc) {
        int dimBatch = x->shape()[0];
        int srcWords = x->shape()[2];
        auto srcWordDrop = graph->dropout(dropoutSrc, {dimBatch, 1, srcWords});
        x = dropout(x, mask=srcWordDrop);
      }

      auto xFw = RNN<GRU>(graph, prefix_ + "_bi",
                          dimSrcEmb, dimEncState,
                          normalize=layerNorm,
                          dropout_prob=dropoutRnn)
                         (x);

      auto xBw = RNN<GRU>(graph, prefix_ + "_bi_r",
                          dimSrcEmb, dimEncState,
                          normalize=layerNorm,
                          direction=dir::backward,
                          dropout_prob=dropoutRnn)
                         (x, mask=xMask);

      if(encoderLayers > 1) {
        auto xBi = concatenate({xFw, xBw}, axis=1);

        Expr xContext;
        std::vector<Expr> states;
        std::tie(xContext, states)
          = MLRNN<GRU>(graph, prefix_, encoderLayers - 1,
                       2 * dimEncState, dimEncState,
                       normalize=layerNorm,
                       skip=skipDepth,
                       dropout_prob=dropoutRnn)
                      (xBi);
        return New<EncoderStateS2S>(xContext, xMask);
      }
      else {
        auto xContext = concatenate({xFw, xBw}, axis=1);
        return New<EncoderStateS2S>(xContext, xMask);
      }
    }
};

class DecoderS2S : public DecoderBase {
  private:
    Ptr<GlobalAttention> attention_;

  public:

    template <class ...Args>
    DecoderS2S(Ptr<Config> options, Args ...args)
     : DecoderBase(options, args...) {}

    virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
      using namespace keywords;

      auto meanContext = weighted_average(encState->getContext(),
                                          encState->getMask(),
                                          axis=2);

      bool layerNorm = options_->get<bool>("layer-normalization");
      auto start = Dense("ff_state",
                         options_->get<int>("dim-rnn"),
                         activation=act::tanh,
                         normalize=layerNorm)(meanContext);
      
      std::vector<Expr> startStates(options_->get<size_t>("layers-dec"), start);
      return New<DecoderStateS2S>(startStates, nullptr, encState);
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
        int dimBatch = embeddings->shape()[0];
        int trgWords = embeddings->shape()[2];
        auto trgWordDrop = graph->dropout(dropoutTrg, {dimBatch, 1, trgWords});
        embeddings = dropout(embeddings, mask=trgWordDrop);
      }

      if(!attention_)
        attention_ = New<GlobalAttention>("decoder",
                                          state->getEncoderState(),
                                          dimDecState,
                                          dropout_prob=dropoutRnn,
                                          normalize=layerNorm);
      RNN<CGRU> rnnL1(graph, "decoder",
                      dimTrgEmb, dimDecState,
                      attention_,
                      dropout_prob=dropoutRnn,
                      normalize=layerNorm);

      auto stateS2S = std::dynamic_pointer_cast<DecoderStateS2S>(state);
      auto stateL1 = rnnL1(embeddings, stateS2S->getStates()[0]);
      auto alignedContext = single ?
        rnnL1.getCell()->getLastContext() :
        rnnL1.getCell()->getContexts();

      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr outputLn;
      if(decoderLayers > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < stateS2S->getStates().size(); ++i)
          statesIn.push_back(stateS2S->getStates()[i]);

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

      if(lf_) {        
        auto alignmentsVec = rnnL1.getCell()->getAttention()->getAlignments();
        Expr aln;
        if(single) {
          aln = alignmentsVec.back();
        }
        else {
          aln = concatenate(alignmentsVec, axis=3);
        }
        
        logitsOut = lexical_bias(logitsOut, aln, 1e-3, lf_);
      }
          
      
      return New<DecoderStateS2S>(statesOut, logitsOut,
                                  state->getEncoderState());
    }

};

typedef EncoderDecoder<EncoderS2S, DecoderS2S> S2S;

}
