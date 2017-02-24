#pragma once

#include "data/corpus.h"
#include "training/config.h"
#include "graph/expression_graph.h"
#include "layers/rnn.h"
#include "layers/param_initializers.h"
#include "layers/generic.h"
#include "common/logging.h"

namespace marian {

class EncDec {
  private:
    Ptr<Config> options_;

    int dimSrcVoc_{40000};
    int dimSrcEmb_{512};
    int dimEncState_{1024};

    int dimTrgVoc_{40000};
    int dimTrgEmb_{512};
    int dimDecState_{1024};

    int dimBatch_{64};

    bool normalize_;
    bool skip_;
    int encoderLayers_{8};
    int decoderLayers_{8};

    void setDims(Ptr<ExpressionGraph> graph,
                 Ptr<data::CorpusBatch> batch) {
      dimSrcVoc_ = graph->get("Wemb") ? graph->get("Wemb")->shape()[0] : dimSrcVoc_;
      dimSrcEmb_ = graph->get("Wemb") ? graph->get("Wemb")->shape()[1] : dimSrcEmb_;
      dimEncState_ = graph->get("encoder_U") ? graph->get("encoder_U")->shape()[0] : dimEncState_;

      dimTrgVoc_ = graph->get("Wemb_dec") ? graph->get("Wemb_dec")->shape()[0] : dimTrgVoc_;
      dimTrgEmb_ = graph->get("Wemb_dec") ? graph->get("Wemb_dec")->shape()[1] : dimTrgEmb_;
      dimDecState_ = graph->get("decoder_U") ? graph->get("decoder_U")->shape()[0] : dimDecState_;

      dimBatch_ = batch->size();
    }

  public:

    EncDec(Ptr<Config> options)
    : options_(options) {

      auto dimVocabs = options->get<std::vector<int>>("dim-vocabs");

      dimSrcVoc_   = dimVocabs[0];
      dimSrcEmb_   = options->get<int>("dim-emb");
      dimEncState_ = options->get<int>("dim-rnn");
      dimTrgVoc_   = dimVocabs[1];
      dimTrgEmb_   = options->get<int>("dim-emb");
      dimDecState_ = options->get<int>("dim-rnn");
      dimBatch_    = options->get<int>("mini-batch");

      encoderLayers_ = options->get<int>("layers-enc");
      decoderLayers_ = options->get<int>("layers-dec");
      skip_ = options->get<bool>("skip");
      normalize_ = options->get<bool>("normalize");
    }


    void load(Ptr<ExpressionGraph> graph,
              const std::string& name) {
      graph->load(name);
    }

    void save(Ptr<ExpressionGraph> graph,
              const std::string& name) {
      graph->save(name);
    }

    std::tuple<Expr, Expr>
    prepareSource(Expr emb, Ptr<data::CorpusBatch> batch, size_t index) {
      using namespace keywords;
      std::vector<size_t> indeces;
      std::vector<float> mask;

      for(auto& word : (*batch)[index]) {
        for(auto i: word.first)
          indeces.push_back(i);
        for(auto m: word.second)
          mask.push_back(m);
      }

      int dimBatch = batch->size();
      int dimEmb = emb->shape()[1];
      int dimWords = (int)(*batch)[index].size();

      auto graph = emb->graph();
      auto x = reshape(rows(emb, indeces), {dimBatch, dimEmb, dimWords});
      auto xMask = graph->constant(shape={dimBatch, 1, dimWords},
                                   init=inits::from_vector(mask));
      return std::make_tuple(x, xMask);
    }

    std::tuple<Expr, Expr, Expr>
    prepareTarget(Expr emb, Ptr<data::CorpusBatch> batch, size_t index) {
      using namespace keywords;

      std::vector<size_t> indeces;
      std::vector<float> mask;
      std::vector<float> findeces;

      for(int j = 0; j < (*batch)[index].size(); ++j) {
        auto& trgWordBatch = (*batch)[index][j];

        for(auto i : trgWordBatch.first) {
          findeces.push_back((float)i);
          if(j < (*batch)[index].size() - 1)
            indeces.push_back(i);
        }

        for(auto m : trgWordBatch.second)
            mask.push_back(m);
      }

      int dimBatch = batch->size();
      int dimEmb = emb->shape()[1];
      int dimWords = (int)(*batch)[index].size();

      auto graph = emb->graph();

      auto y = reshape(rows(emb, indeces),
                       {dimBatch, dimEmb, dimWords - 1});

      auto yMask = graph->constant(shape={dimBatch, 1, dimWords},
                                  init=inits::from_vector(mask));
      auto yIdx = graph->constant(shape={(int)findeces.size(), 1},
                                  init=inits::from_vector(findeces));

      return std::make_tuple(y, yMask, yIdx);
    }

    std::tuple<Expr, Expr> encoder(Ptr<ExpressionGraph> graph,
                                   Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      auto xEmb = Embedding("Wemb", dimSrcVoc_, dimSrcEmb_)(graph);

      Expr x, xMask;
      std::tie(x, xMask) = prepareSource(xEmb, batch, 0);

      auto xFw = RNN<GRU>(graph, "encoder_bi",
                          dimSrcEmb_, dimEncState_,
                          normalize=normalize_)
                         (x);

      auto xBw = RNN<GRU>(graph, "encoder_bi_r",
                          dimSrcEmb_, dimEncState_,
                          normalize=normalize_,
                          direction=dir::backward)
                         (x, mask=xMask);

      if(encoderLayers_ > 1) {
        auto xBi = concatenate({xFw, xBw}, axis=1);
        auto xContexts = MLRNN<GRU>(graph, "encoder", encoderLayers_ - 1,
                              2 * dimEncState_, dimEncState_,
                              normalize=normalize_,
                              residual=skip_)
                             (xBi);
        return std::make_tuple(xContexts.back(), xMask);
      }
      else {
        auto xContext = concatenate({xFw, xBw}, axis=1);
        return std::make_tuple(xContext, xMask);
      }
    }

    std::tuple<Expr, std::vector<Expr>>
    step(Expr embeddings,
         std::vector<Expr> states,
         Expr context,
         Expr contextMask) {
      using namespace keywords;

      auto graph = embeddings->graph();

      auto attention = New<GlobalAttention>("decoder",
                                            context, dimDecState_,
                                            mask=contextMask,
                                            normalize=normalize_);
      RNN<CGRU> rnnL1(graph, "decoder",
                      dimTrgEmb_, dimDecState_,
                      attention,
                      normalize=normalize_);
      auto stateL1 = rnnL1(embeddings, states[0]);
      auto alignedContext = rnnL1.getCell()->getContexts();

      std::vector<Expr> statesOut;
      statesOut.push_back(stateL1);

      Expr stateLast;
      if(decoderLayers_ > 1) {
        std::vector<Expr> statesIn;
        for(int i = 1; i < states.size(); ++i)
          statesIn.push_back(states[i]);

        auto statesLn = MLRNN<GRU>(graph, "decoder",
                                   decoderLayers_ - 1,
                                   dimDecState_, dimDecState_,
                                   normalize=normalize_,
                                   residual=skip_)
                                  (stateL1, statesIn);

        statesOut.insert(statesOut.end(),
                         statesLn.begin(), statesLn.end());

        stateLast = statesOut.back();
      }
      else {
        stateLast = stateL1;
      }

      //// 2-layer feedforward network for outputs and cost
      auto logitsL1 = Dense("ff_logit_l1", dimTrgEmb_,
                            activation=act::tanh,
                            normalize=normalize_)
                        (embeddings, stateLast, alignedContext);

      auto logitsL2 = Dense("ff_logit_l2", dimTrgVoc_)
                        (logitsL1);

      return std::make_tuple(logitsL2, statesOut);
    }

    Expr buildStartState(Expr context, Expr mask) {
      using namespace keywords;

      auto meanContext = weighted_average(context, mask, axis=2);
      auto start = Dense("ff_state",
                         dimDecState_,
                         activation=act::tanh,
                         normalize=normalize_)(meanContext);
      return start;
    }

    std::tuple<Expr, Expr, Expr> buildEmbeddings(Ptr<ExpressionGraph> graph,
                                                 Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      auto yEmb = Embedding("Wemb_dec", dimTrgVoc_, dimTrgEmb_)(graph);
      Expr y, yMask, yIdx;
      std::tie(y, yMask, yIdx) = prepareTarget(yEmb, batch, 1);
      auto yEmpty = graph->zeros(shape={dimBatch_, dimTrgEmb_});
      auto yShifted = concatenate({yEmpty, y}, axis=2);

      return std::make_tuple(yShifted, yMask, yIdx);
    }

    void clear(Ptr<ExpressionGraph> graph,
               Ptr<data::CorpusBatch> batch) {
      graph->clear();
      setDims(graph, batch);
    }

    std::tuple<std::vector<Expr>, Expr, Expr>
    buildEncoder(Ptr<ExpressionGraph> graph,
                 Ptr<data::CorpusBatch> batch) {
      using namespace keywords;
      clear(graph, batch);

      Expr srcContext, srcMask;
      std::tie(srcContext, srcMask) = encoder(graph, batch);
      auto startState = buildStartState(srcContext, srcMask);

      std::vector<Expr> startStates(decoderLayers_, startState);

      return std::make_tuple(startStates, srcContext, srcMask);
    }


    Expr build(Ptr<ExpressionGraph> graph,
               Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      std::vector<Expr> startStates;
      Expr srcContext, srcMask;
      std::tie(startStates, srcContext, srcMask) = buildEncoder(graph, batch);

      Expr trgEmbeddings, trgMask, trgIdx;
      std::tie(trgEmbeddings, trgMask, trgIdx) = buildEmbeddings(graph, batch);

      Expr trgLogits;
      std::vector<Expr> trgStates;
      std::tie(trgLogits, trgStates) = step(trgEmbeddings,
                                            startStates,
                                            srcContext,
                                            srcMask);

      auto cost = CrossEntropyCost("cost")(trgLogits, trgIdx,
                                           mask=trgMask);

      return cost;
    }
};

}
