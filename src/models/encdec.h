#pragma once

#include "data/corpus.h"
#include "training/config.h"
#include "graph/expression_graph.h"
#include "layers/rnn.h"
#include "layers/param_initializers.h"
#include "layers/generic.h"
#include "3rd_party/cnpy/cnpy.h"
#include "common/logging.h"

namespace marian {

class EncDec {
  private:
    Ptr<Config> options_;

    Ptr<RNN<CGRU>> rnn_;

    int dimSrcVoc_{40000};
    int dimSrcEmb_{512};
    int dimEncState_{1024};

    int dimTrgVoc_{40000};
    int dimTrgEmb_{512};
    int dimDecState_{1024};

    int dimBatch_{64};

    bool normalize_;

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

      normalize_ = options->get<bool>("normalize");

      dimSrcVoc_   = dimVocabs[0];
      dimSrcEmb_   = options->get<int>("dim-emb");
      dimEncState_ = options->get<int>("dim-rnn");
      dimTrgVoc_   = dimVocabs[0];
      dimTrgEmb_   = options->get<int>("dim-emb");
      dimDecState_ = options->get<int>("dim-rnn");
      dimBatch_    = options->get<int>("mini-batch");
    }


    void load(Ptr<ExpressionGraph> graph,
              const std::string& name) {
      using namespace keywords;

      LOG(info) << "Loading model from " << name;

      auto numpy = cnpy::npz_load(name);

      for(auto it : numpy) {
        auto name = it.first;

        Shape shape;
        if(it.second.shape.size() == 2) {
          shape.set(0, it.second.shape[0]);
          shape.set(1, it.second.shape[1]);
        }
        else if(it.second.shape.size() == 1) {
          shape.set(0, 1);
          shape.set(1, it.second.shape[0]);
        }

        graph->param(name, shape,
                     init=inits::from_numpy(it.second));
      }
    }

    void save(Ptr<ExpressionGraph> graph,
              const std::string& name) {

      LOG(info) << "Saving model to " << name;

      unsigned shape[2];
      std::string mode = "w";

      cudaSetDevice(graph->getDevice());
      for(auto p : graph->params().getMap()) {
        std::vector<float> v;
        p.second->val() >> v;

        unsigned dim;
        if(p.second->shape()[0] == 1) {
          shape[0] = p.second->shape()[1];
          dim = 1;
        }
        else {
          shape[0] = p.second->shape()[0];
          shape[1] = p.second->shape()[1];
          dim = 2;
        }
        std::string pName = p.first;
        cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
        mode = "a";
      }
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

      auto xFw = MLRNN<GRU>(graph, "encoder", 2,
                            dimSrcEmb_, dimEncState_,
                            normalize=normalize_,
                            direction=dir::forward)
                           (x);

      auto xBw = MLRNN<GRU>(graph, "encoder_r", 2,
                            dimSrcEmb_, dimEncState_,
                            normalize=normalize_,
                            direction=dir::backward)
                           (x, mask=xMask);

      auto xContext = concatenate({xFw.back(), xBw.back()}, axis=1);
      return std::make_tuple(xContext, xMask);
    }

    std::tuple<Expr, Expr> step(Expr hyps,
                                const std::vector<size_t> hypIdx = {},
                                const std::vector<size_t> embIdx = {}) {
      using namespace keywords;
      auto graph = hyps->graph();

      Expr selectedHyps, selectedEmbs;
      if(embIdx.empty()) {
        selectedHyps = hyps;
        selectedEmbs = graph->constant(shape={1, dimTrgEmb_},
                                       init=inits::zeros);
      }
      else {
        selectedHyps = rows(hyps, hypIdx);

        auto yEmb = Embedding("Wemb_dec", dimTrgVoc_, dimTrgEmb_)(graph);
        selectedEmbs = rows(yEmb, embIdx);
      }

      Expr newHyps, logits;
      std::tie(newHyps, logits) = step(selectedHyps, selectedEmbs, true);
      return std::make_tuple(newHyps, logsoftmax(logits));
    }

    std::tuple<Expr, Expr> step(Expr yInStates, Expr yEmbeddings,
                                bool single = false) {
      using namespace keywords;

      auto yOutStates = (*rnn_)(yEmbeddings, yInStates);
      auto yCtx = single ?
        rnn_->getCell()->getLastContext() :
        rnn_->getCell()->getContexts();


      //// 2-layer feedforward network for outputs and cost
      auto yLogitsL1 = Dense("ff_logit_l1", dimTrgEmb_,
                             activation=act::tanh,
                             normalize=normalize_)
                         (yEmbeddings, yOutStates, yCtx);

      auto yLogitsL2 = Dense("ff_logit_l2", dimTrgVoc_)
                         (yLogitsL1);

      return std::make_tuple(yOutStates, yLogitsL2);
    }

    Expr startState(Expr context, Expr mask) {
      using namespace keywords;

      auto meanContext = weighted_average(context, mask, axis=2);
      auto start = Dense("ff_state",
                         dimDecState_,
                         activation=act::tanh,
                         normalize=normalize_)(meanContext);
      return start;
    }

    Expr buildEncoder(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
      using namespace keywords;
      graph->clear();
      rnn_.reset();
      setDims(graph, batch);

      Expr xContext, xMask;
      std::tie(xContext, xMask) = encoder(graph, batch);

      auto attention = New<GlobalAttention>("decoder",
                                            xContext, dimDecState_,
                                            mask=xMask, normalize=normalize_);
      rnn_ = New<RNN<CGRU>>(graph, "decoder",
                            dimTrgEmb_, dimDecState_,
                            attention,
                            normalize=normalize_);

      return startState(xContext, xMask);
    }

    std::tuple<Expr, Expr, Expr> embeddings(Ptr<ExpressionGraph> graph,
                                            Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      auto yEmb = Embedding("Wemb_dec", dimTrgVoc_, dimTrgEmb_)(graph);
      Expr y, yMask, yIdx;
      std::tie(y, yMask, yIdx) = prepareTarget(yEmb, batch, 1);
      auto yEmpty = graph->zeros(shape={dimBatch_, dimTrgEmb_});
      auto yShifted = concatenate({yEmpty, y}, axis=2);

      return std::make_tuple(yShifted, yMask, yIdx);
    }

    Expr build(Ptr<ExpressionGraph> graph,
               Ptr<data::CorpusBatch> batch) {
      using namespace keywords;
      graph->clear();
      rnn_.reset();
      setDims(graph, batch);

      Expr xContext, xMask;
      std::tie(xContext, xMask) = encoder(graph, batch);
      auto yStartStates = startState(xContext, xMask);

      Expr yEmbeddings, yMask, yIdx;
      std::tie(yEmbeddings, yMask, yIdx) = embeddings(graph, batch);

      auto attention = New<GlobalAttention>("decoder",
                                            xContext, dimDecState_,
                                            mask=xMask, normalize=normalize_);
      rnn_ = New<RNN<CGRU>>(graph, "decoder",
                            dimTrgEmb_, dimDecState_,
                            attention,
                            normalize=normalize_);

      Expr yOutStates, yLogits;
      std::tie(yOutStates, yLogits) = step(yStartStates, yEmbeddings);

      auto cost = CrossEntropyCost("cost")(yLogits, yIdx, mask=yMask);

      return cost;
    }
};

}
