#pragma once

#include "models/encdec.h"
#include "layers/attention.h"

namespace marian {

typedef AttentionCell<GRU, GlobalAttention, GRU> CGRU;

class EncoderGNMT : public EncoderBase {
  public:
    template <class ...Args>
    EncoderGNMT(Ptr<Config> options, Args... args)
     : EncoderBase(options, args...) {}

    Ptr<EncoderState>
    build(Ptr<ExpressionGraph> graph,
          Ptr<data::CorpusBatch> batch,
          size_t batchIdx = 0) {

      using namespace keywords;

      int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[batchIdx];
      int dimSrcEmb = options_->get<int>("dim-emb");
      int dimEncState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("normalize");
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
        return New<EncoderState>(EncoderState{xContext, xMask});
      }
      else {
        auto xContext = concatenate({xFw, xBw}, axis=1);
        return New<EncoderState>(EncoderState{xContext, xMask});
      }
    }
};

class DecoderGNMT : public DecoderBase {
  private:
    Ptr<GlobalAttention> attention_;

  public:

    template <class ...Args>
    DecoderGNMT(Ptr<Config> options, Args ...args)
     : DecoderBase(options, args...) {}

    virtual std::tuple<Expr, std::vector<Expr>>
    step(Expr embeddings,
         std::vector<Expr> states,
         Ptr<EncoderState> encState,
         bool single) {
      using namespace keywords;

      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      int dimTrgEmb = options_->get<int>("dim-emb");
      int dimDecState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("normalize");
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
                                          encState,
                                          dimDecState,
                                          dropout_prob=dropoutRnn,
                                          normalize=layerNorm);
      RNN<CGRU> rnnL1(graph, "decoder",
                      dimTrgEmb, dimDecState,
                      attention_,
                      dropout_prob=dropoutRnn,
                      normalize=layerNorm);

      auto stateL1 = rnnL1(embeddings, states[0]);
      auto alignedContext = single ?
        rnnL1.getCell()->getLastContext() :
        rnnL1.getCell()->getContexts();

      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr outputLn;
      if(decoderLayers > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < states.size(); ++i)
          statesIn.push_back(states[i]);

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

      auto logitsL2 = Dense("ff_logit_l2", dimTrgVoc)
                        (logitsL1);

      return std::make_tuple(logitsL2, statesOut);
    }

};

typedef Seq2Seq<EncoderGNMT, DecoderGNMT> GNMT;

}
