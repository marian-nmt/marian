#pragma once

#include "common/options.h"
#include "rnn/constructors.h"

#include "models/encdec.h"

namespace marian {

class EncoderStateS2S : public EncoderState {
private:
  Expr context_;
  Expr mask_;
  Ptr<data::CorpusBatch> batch_;

public:
  EncoderStateS2S(Expr context, Expr mask, Ptr<data::CorpusBatch> batch)
      : context_(context), mask_(mask), batch_(batch) {}

  Expr getContext() { return context_; }
  Expr getAttended() { return context_; }
  Expr getMask() { return mask_; }

  virtual const std::vector<size_t>& getSourceWords() {
    return batch_->front()->indeces();
  }
};

class DecoderStateS2S : public DecoderState {
private:
  rnn::States states_;
  Expr probs_;
  Ptr<EncoderState> encState_;

public:
  DecoderStateS2S(const rnn::States& states,
                  Expr probs,
                  Ptr<EncoderState> encState)
      : states_(states), probs_(probs), encState_(encState) {}

  Ptr<EncoderState> getEncoderState() { return encState_; }
  Expr getProbs() { return probs_; }
  void setProbs(Expr probs) { probs_ = probs; }

  Ptr<DecoderState> select(const std::vector<size_t>& selIdx) {
    return New<DecoderStateS2S>(states_.select(selIdx), probs_, encState_);
  }

  const rnn::States& getStates() { return states_; }
};

class EncoderS2S : public EncoderBase {
public:
  template <class... Args>
  EncoderS2S(Ptr<Config> options, Args... args)
      : EncoderBase(options, args...) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch,
                          size_t batchIdx) {
    using namespace keywords;

    int dimSrcVoc = options_->get<std::vector<int>>("dim-vocabs")[batchIdx];
    int dimSrcEmb = options_->get<int>("dim-emb");

    int dimEncState = options_->get<int>("dim-rnn");
    bool layerNorm = options_->get<bool>("layer-normalization");
    bool skipDepth = options_->get<bool>("skip");
    size_t encoderLayers = options_->get<size_t>("layers-enc");

    bool amun = options_->get<std::string>("type") == "amun";
    UTIL_THROW_IF2(amun && options_->get<int>("layers-enc") > 1,
                   "--type amun does not currently support multiple encoder "
                   "layers, use --type s2s");
    UTIL_THROW_IF2(amun && options_->get<bool>("skip"),
                   "--type amun does not currently support skip connections, "
                   "use --type s2s");

    float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
    float dropoutSrc = inference_ ? 0 : options_->get<float>("dropout-src");

    auto xEmb = Embedding(prefix_ + "_Wemb", dimSrcVoc, dimSrcEmb)(graph);

    Expr x, xMask;
    std::tie(x, xMask) = prepareSource(xEmb, batch, batchIdx);

    if(dropoutSrc) {
      int dimBatch = x->shape()[0];
      int srcWords = x->shape()[2];
      auto srcWordDrop = graph->dropout(dropoutSrc, {1, 1, srcWords});
      x = dropout(x, mask = srcWordDrop);
    }

    auto cellType = options_->get<std::string>("cell-enc");
    UTIL_THROW_IF2(amun && cellType != "gru",
                   "--type amun does not currently support other rnn cells than gru, "
                   "use --type s2s");

    auto rnnFw = rnn::rnn(graph)
                 ("type", cellType)
                 ("prefix", prefix_ + "_bi")
                 ("dimInput", dimSrcEmb)
                 ("dimState", dimEncState)
                 ("dropout", dropoutRnn)
                 ("normalize", layerNorm)
                 .push_back(rnn::cell(graph));

    auto rnnBw = rnnFw.clone()
                 ("prefix", prefix_ + "_bi_r")
                 ("direction", (int)rnn::backward);

    auto context = concatenate({rnnFw->transduce(x),
                                rnnBw->transduce(x, xMask)},
                                axis=1);

    if(encoderLayers > 1) {
      auto rnnML = rnn::rnn(graph)
                   ("type", cellType)
                   ("dimInput", 2 * dimEncState)
                   ("dimState", dimEncState)
                   ("dropout", dropoutRnn)
                   ("normalize", layerNorm)
                   ("skip", skipDepth)
                   ("skipFirst", skipDepth);

      for(int i = 0; i < encoderLayers - 1; ++i)
        rnnML.push_back(rnn::cell(graph)
                        ("prefix", prefix_ + "_l" + std::to_string(i)));

      context = rnnML->transduce(context);
    }

    return New<EncoderStateS2S>(context, xMask, batch);
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

    auto meanContext = weighted_average(
        encState->getContext(), encState->getMask(), axis = 2);

    bool layerNorm = options_->get<bool>("layer-normalization");
    auto start = Dense(prefix_ + "_ff_state",
                       options_->get<int>("dim-rnn"),
                       activation = act::tanh,
                       normalize = layerNorm)(meanContext);

    // @TODO: review this
    rnn::States startStates;
    for(int i = 0; i < options_->get<size_t>("layers-dec"); ++i)
      startStates.push_back(rnn::State{start, start});

    return New<DecoderStateS2S>(startStates, nullptr, encState);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();

    int dimTrgEmb
        = options_->get<int>("dim-emb") + options_->get<int>("dim-pos");

    int dimDecState = options_->get<int>("dim-rnn");
    bool layerNorm = options_->get<bool>("layer-normalization");
    bool skipDepth = options_->get<bool>("skip");
    size_t decoderLayers = options_->get<size_t>("layers-dec");

    float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
    float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

    auto cellType = options_->get<std::string>("cell-dec");


    bool amun = options_->get<std::string>("type") == "amun";
    UTIL_THROW_IF2(amun && options_->get<bool>("skip"),
                   "--type amun does not currently support skip connections, "
                   "use --type s2s");
    UTIL_THROW_IF2(amun && options_->get<int>("layers-dec") > 1,
                   "--type amun does not currently support multiple decoder "
                   "layers, use --type s2s");
    UTIL_THROW_IF2(amun && options_->get<bool>("tied-embeddings"),
                   "--type amun does not currently support tied embeddings, "
                   "use --type s2s");
    UTIL_THROW_IF2(amun && cellType != "gru",
                   "--type amun does not currently support other rnn cells than gru, "
                   "use --type s2s");


    auto stateS2S = std::dynamic_pointer_cast<DecoderStateS2S>(state);

    auto embeddings = stateS2S->getTargetEmbeddings();

    if(dropoutTrg) {
      int dimBatch = embeddings->shape()[0];
      int trgWords = embeddings->shape()[2];
      auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
      embeddings = dropout(embeddings, mask = trgWordDrop);
    }

    if(!rnn_) {

      auto rnn = rnn::rnn(graph)
                 ("type", cellType)
                 ("dimInput", dimTrgEmb)
                 ("dimState", dimDecState)
                 ("dropout", dropoutRnn)
                 ("normalize", layerNorm)
                 ("skip", skipDepth);

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
      for(int i = 0; i < decoderLayers - 1; ++i)
        rnn.push_back(rnn::cell(graph)
                      ("prefix", prefix_ + "_l" + std::to_string(i)));

      rnn_ = rnn.construct();
      
    }

    auto decContext = rnn_->transduce(embeddings, stateS2S->getStates()[0]);
    rnn::States decStates = rnn_->lastCellStates();

    bool single = stateS2S->doSingleStep();
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    auto alignedContext = single ? att->getContexts().back() :
                                   concatenate(att->getContexts(),
                                               keywords::axis = 2);

    //// 2-layer feedforward network for outputs and cost
    auto logitsL1
        = Dense(prefix_ + "_ff_logit_l1",
                dimTrgEmb,
                activation = act::tanh,
                normalize = layerNorm)(embeddings, decContext, alignedContext);

    Expr logitsOut;
    if(options_->get<bool>("tied-embeddings")) {
      if(!tiedOutputWeights_)
        tiedOutputWeights_ = transpose(graph->get(prefix_ + "_Wemb"));

      logitsOut = DenseTied(
          prefix_ + "_ff_logit_l2", tiedOutputWeights_, dimTrgVoc)(logitsL1);
    } else {
      logitsOut = Dense(prefix_ + "_ff_logit_l2", dimTrgVoc)(logitsL1);
    }

    return New<DecoderStateS2S>(decStates, logitsOut, state->getEncoderState());
  }

  const std::vector<Expr> getAlignments() {
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    return att->getAlignments();
  }
};

typedef EncoderDecoder<EncoderS2S, DecoderS2S> S2S;
}
