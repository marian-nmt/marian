#pragma once

#include "data/corpus.h"
#include "training/config.h"
#include "graph/expression_graph.h"
#include "layers/rnn.h"
#include "layers/param_initializers.h"
#include "layers/generic.h"
#include "common/logging.h"

namespace marian {

class EncoderBase {
  protected:
    Ptr<Config> options_;

    virtual std::tuple<Expr, Expr>
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

  public:
    EncoderBase(Ptr<Config> options)
     : options_(options) {}

    virtual std::tuple<Expr, Expr>
    build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>, size_t = 0) = 0;
};

class DecoderBase {
  protected:
    Ptr<Config> options_;

    virtual std::tuple<Expr, Expr, Expr>
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

  public:
    DecoderBase(Ptr<Config> options)
     : options_(options) {}

    virtual std::tuple<Expr, Expr, Expr>
    groundTruth(Ptr<ExpressionGraph> graph,
                Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      int dimBatch  = batch->size();
      int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();
      int dimTrgEmb = options_->get<int>("dim-emb");

      auto yEmb = Embedding("Wemb_dec", dimTrgVoc, dimTrgEmb)(graph);
      Expr y, yMask, yIdx;
      std::tie(y, yMask, yIdx) = prepareTarget(yEmb, batch, 1);
      auto yEmpty = graph->zeros(shape={dimBatch, dimTrgEmb});
      auto yShifted = concatenate({yEmpty, y}, axis=2);

      return std::make_tuple(yShifted, yMask, yIdx);
    }

    virtual Expr
    buildStartState(Expr context, Expr mask) {
      using namespace keywords;

      auto meanContext = weighted_average(context, mask, axis=2);

      bool layerNorm = options_->get<bool>("normalize");
      auto start = Dense("ff_state",
                         options_->get<int>("dim-rnn"),
                         activation=act::tanh,
                         normalize=layerNorm)(meanContext);
      return start;
    }

    virtual std::tuple<Expr, std::vector<Expr>>
    step(Expr embeddings, std::vector<Expr> states,
         Expr context, Expr contextMask) = 0;
};

template <class Encoder, class Decoder>
class Seq2Seq {
  protected:
    Ptr<Config> options_;
    Ptr<EncoderBase> encoder_;
    Ptr<DecoderBase> decoder_;

  public:

    Seq2Seq(Ptr<Config> options)
     : options_(options),
       encoder_(New<Encoder>(options)),
       decoder_(New<Decoder>(options))
    {}

     virtual void load(Ptr<ExpressionGraph> graph,
                       const std::string& name) {
      graph->load(name);
    }

    virtual void save(Ptr<ExpressionGraph> graph,
                      const std::string& name) {
      graph->save(name);
    }

    virtual std::tuple<std::vector<Expr>, Expr, Expr>
    buildEncoder(Ptr<ExpressionGraph> graph,
                 Ptr<data::CorpusBatch> batch) {
      using namespace keywords;
      graph->clear();

      Expr srcContext, srcMask;
      std::tie(srcContext, srcMask) = encoder_->build(graph, batch);
      auto startState = decoder_->buildStartState(srcContext, srcMask);

      size_t decoderLayers = options_->get<size_t>("layers-dec");
      std::vector<Expr> startStates(decoderLayers, startState);

      return std::make_tuple(startStates, srcContext, srcMask);
    }

    virtual std::tuple<Expr, std::vector<Expr>>
    step(Expr embeddings,
         std::vector<Expr> states,
         Expr context,
         Expr contextMask) {
      return decoder_->step(embeddings, states, context, contextMask);
    }

    virtual Expr build(Ptr<ExpressionGraph> graph,
                       Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      std::vector<Expr> startStates;
      Expr srcContext, srcMask;
      std::tie(startStates, srcContext, srcMask) = buildEncoder(graph, batch);

      Expr trgEmbeddings, trgMask, trgIdx;
      std::tie(trgEmbeddings, trgMask, trgIdx) = decoder_->groundTruth(graph, batch);

      Expr trgLogits;
      std::vector<Expr> trgStates;
      std::tie(trgLogits, trgStates) = decoder_->step(trgEmbeddings,
                                                      startStates,
                                                      srcContext,
                                                      srcMask);

      auto cost = CrossEntropyCost("cost")(trgLogits, trgIdx,
                                           mask=trgMask);

      return cost;
    }

};

}
