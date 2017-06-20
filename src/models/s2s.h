#pragma once

#include "common/options.h"
#include "rnn/constructors.h"
#include "models/encdec.h"
#include "layers/constructors.h"

namespace marian {

class EncoderS2S : public EncoderBase {
public:
  template <class... Args>
  EncoderS2S(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t encoderIndex) {
    using namespace keywords;

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[encoderIndex];
    auto embeddings = Embedding(prefix_ + "_Wemb",
                                dimVoc,
                                opt<int>("dim-emb"))(graph);

    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
      = prepareSource(embeddings, batch, encoderIndex);

    float dropProb = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProb) {
      int srcWords = batchEmbeddings->shape()[2];
      auto dropMask = graph->dropout(dropProb, {1, 1, srcWords});
      batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
    }

    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnnFw = rnn::rnn(graph)
                 ("type", opt<std::string>("cell-enc"))
                 ("prefix", prefix_ + "_bi")
                 ("dimInput", opt<int>("dim-emb"))
                 ("dimState", opt<int>("dim-rnn"))
                 ("dropout", dropoutRnn)
                 ("normalize", opt<bool>("layer-normalization"))
                 .push_back(rnn::cell(graph));

    auto rnnBw = rnnFw.clone()
                 ("prefix", prefix_ + "_bi_r")
                 ("direction", rnn::dir::backward);

    auto context = concatenate({rnnFw->transduce(batchEmbeddings),
                                rnnBw->transduce(batchEmbeddings, batchMask)},
                                axis=1);

    if(opt<size_t>("layers-enc") > 1) {
      auto rnnML = rnn::rnn(graph)
                   ("type", opt<std::string>("cell-enc"))
                   ("dimInput", 2 * opt<int>("dim-rnn"))
                   ("dimState", opt<int>("dim-rnn"))
                   ("dropout", dropoutRnn)
                   ("normalize", opt<bool>("layer-normalization"))
                   ("skip", opt<bool>("skip"))
                   ("skipFirst", opt<bool>("skip"));

      for(int i = 0; i < opt<size_t>("layers-enc") - 1; ++i)
        rnnML.push_back(rnn::cell(graph)
                        ("prefix", prefix_ + "_l" + std::to_string(i)));

      context = rnnML->transduce(context);
    }

    return New<EncoderState>(context, batchMask, batch);
  }
};

class DecoderS2S : public DecoderBase {
private:
  Ptr<rnn::RNN> rnn_;
  Expr tiedOutputWeights_;

public:
  template <class... Args>
  DecoderS2S(Ptr<Config> options, Args... args)
      : DecoderBase(options, args...) {}

  virtual Ptr<DecoderState> startState(Ptr<EncoderState> encState) {
    using namespace keywords;

    auto meanContext = weighted_average(encState->getContext(),
                                        encState->getMask(),
                                        axis = 2);

    auto graph = meanContext->graph();
    auto mlp = mlp::mlp(graph)
               .push_back(mlp::dense(graph)
                          ("prefix", prefix_ + "_ff_state")
                          ("dim", opt<int>("dim-rnn"))
                          ("activation", mlp::act::tanh)
                          ("normalization", opt<bool>("layer-normalization")));
    auto start = mlp->apply(meanContext);

    rnn::States startStates(opt<size_t>("layers-dec"), {start, start});
    return New<DecoderState>(startStates, nullptr, encState);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();

    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[2];
      auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
      embeddings = dropout(embeddings, mask = trgWordDrop);
    }

    if(!rnn_) {
      float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
      auto rnn = rnn::rnn(graph)
                 ("type", opt<std::string>("cell-dec"))
                 ("dimInput", opt<int>("dim-emb"))
                 ("dimState", opt<int>("dim-rnn"))
                 ("dropout", dropoutRnn)
                 ("normalize", opt<bool>("layer-normalization"))
                 ("skip", opt<bool>("skip"));

      auto attCell = rnn::stacked_cell(graph)
                     .push_back(rnn::cell(graph)
                                ("prefix", prefix_ + "_cell1"))
                     .push_back(rnn::attention(graph)
                                ("prefix", prefix_)
                                .set_state(state->getEncoderState()))
                     .push_back(rnn::cell(graph)
                                ("prefix", prefix_ + "_cell2")
                                ("final", true));

      rnn.push_back(attCell);
      size_t decoderLayers = opt<size_t>("layers-dec");
      for(int i = 0; i < decoderLayers - 1; ++i)
        rnn.push_back(rnn::cell(graph)
                      ("prefix", prefix_ + "_l" + std::to_string(i)));

      rnn_ = rnn.construct();
    }

    auto decContext = rnn_->transduce(embeddings, state->getStates());
    rnn::States decStates = rnn_->lastCellStates();

    bool single = state->doSingleStep();
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    auto alignedContext = single ? att->getContexts().back() :
                                   concatenate(att->getContexts(),
                                               keywords::axis = 2);

    auto layer1 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l1")
                  ("dim", opt<int>("dim-emb"))
                  ("activation", mlp::act::tanh)
                  ("normalization", opt<bool>("layer-normalization"));

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();
    auto layer2 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l2")
                  ("dim", dimTrgVoc);

    if(opt<bool>("tied-embeddings")) {
      UTIL_THROW2("Tied embeddings currently not implemented");
    }

    auto logits = mlp::mlp(graph)
                  .push_back(layer1)
                  .push_back(layer2)
                  ->apply(embeddings, decContext, alignedContext);

    return New<DecoderState>(decStates, logits, state->getEncoderState());
  }

  virtual const std::vector<Expr> getAlignments() {
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    return att->getAlignments();
  }
};

typedef EncoderDecoder<EncoderS2S, DecoderS2S> S2S;

}
