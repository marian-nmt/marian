#include "models/encdec.h"

namespace marian {

  class EncoderGNMT : public EncoderBase {
  public:
    EncoderGNMT(Ptr<Config> options)
     : EncoderBase(options) {}

    std::tuple<Expr, Expr>
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

      auto xEmb = Embedding("Wemb", dimSrcVoc, dimSrcEmb)(graph);

      Expr x, xMask;
      std::tie(x, xMask) = prepareSource(xEmb, batch, batchIdx);

      auto xFw = RNN<GRU>(graph, "encoder_bi",
                          dimSrcEmb, dimEncState,
                          normalize=layerNorm)
                         (x);

      auto xBw = RNN<GRU>(graph, "encoder_bi_r",
                          dimSrcEmb, dimEncState,
                          normalize=layerNorm,
                          direction=dir::backward)
                         (x, mask=xMask);

      if(encoderLayers > 1) {
        auto xBi = concatenate({xFw, xBw}, axis=1);

        Expr xContext;
        std::vector<Expr> states;
        std::tie(xContext, states)
          = MLRNN<GRU>(graph, "encoder", encoderLayers - 1,
                       2 * dimEncState, dimEncState,
                       normalize=layerNorm,
                       skip=skipDepth)
                      (xBi);
        return std::make_tuple(xContext, xMask);
      }
      else {
        auto xContext = concatenate({xFw, xBw}, axis=1);
        return std::make_tuple(xContext, xMask);
      }
    }
};

class DecoderGNMT : public DecoderBase {

  public:
    DecoderGNMT(Ptr<Config> options)
     : DecoderBase(options) {}

    virtual std::tuple<Expr, std::vector<Expr>>
    step(Expr embeddings,
         std::vector<Expr> states,
         Expr context,
         Expr contextMask) {
      using namespace keywords;

      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      int dimTrgEmb = options_->get<int>("dim-emb");
      int dimDecState = options_->get<int>("dim-rnn");
      bool layerNorm = options_->get<bool>("normalize");
      bool skipDepth = options_->get<bool>("skip");
      size_t decoderLayers = options_->get<size_t>("layers-dec");


      auto graph = embeddings->graph();

      auto attention = New<GlobalAttention>("decoder",
                                            context, dimDecState,
                                            mask=contextMask,
                                            normalize=layerNorm);
      RNN<CGRU> rnnL1(graph, "decoder",
                      dimTrgEmb, dimDecState,
                      attention,
                      normalize=layerNorm);
      auto stateL1 = rnnL1(embeddings, states[0]);
      auto alignedContext = rnnL1.getCell()->getContexts();

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
