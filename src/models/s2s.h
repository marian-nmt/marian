#pragma once

#include "layers/attention.h"
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
  Expr getMask() { return mask_; }

  virtual const std::vector<size_t>& getSourceWords() {
    return batch_->front()->indeces();
  }
};

class DecoderStateS2S : public DecoderState {
private:
  RNNStates states_;
  Expr probs_;
  Ptr<EncoderState> encState_;

public:
  DecoderStateS2S(const RNNStates& states,
                  Expr probs,
                  Ptr<EncoderState> encState)
      : states_(states), probs_(probs), encState_(encState) {}

  Ptr<EncoderState> getEncoderState() { return encState_; }
  Expr getProbs() { return probs_; }
  void setProbs(Expr probs) { probs_ = probs; }

  Ptr<DecoderState> select(const std::vector<size_t>& selIdx) {
    return New<DecoderStateS2S>(states_.select(selIdx), probs_, encState_);
  }

  const RNNStates& getStates() { return states_; }
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

    RNNStates statesFw = RNN<GRU>(graph,
                                   prefix_ + "_bi",
                                   dimSrcEmb,
                                   dimEncState,
                                   normalize = layerNorm,
                                   dropout_prob = dropoutRnn)(x);
    auto xFw = statesFw.outputs();

    RNNStates statesBw = RNN<GRU>(graph,
                                   prefix_ + "_bi_r",
                                   dimSrcEmb,
                                   dimEncState,
                                   normalize = layerNorm,
                                   direction = dir::backward,
                                   dropout_prob = dropoutRnn)(x, mask = xMask);
    auto xBw = statesBw.outputs();

    //if(encoderLayers > 1) {
    //  auto xBi = concatenate({xFw, xBw}, axis = 1);
    //
    //  Expr xContext;
    //  std::vector<Expr> states;
    //  std::tie(xContext, states) = MLRNN<GRU>(graph,
    //                                          prefix_,
    //                                          encoderLayers - 1,
    //                                          2 * dimEncState,
    //                                          dimEncState,
    //                                          normalize = layerNorm,
    //                                          skip = skipDepth,
    //                                          dropout_prob = dropoutRnn)(xBi);
    //  return New<EncoderStateS2S>(xContext, xMask, batch);
    //} else {
      auto xContext = concatenate({xFw, xBw}, axis = 1);
      return New<EncoderStateS2S>(xContext, xMask, batch);
    //}
  }
};

class DecoderS2S : public DecoderBase {
private:
  Ptr<GlobalAttention> attention_;
  Ptr<RNN<CGRU>> rnnL1;
  Ptr<MLRNN<GRU>> rnnLn;
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

    int dimBatch = start->shape()[0];
    int dimState = options_->get<int>("dim-rnn");

    auto graph = start->graph();
    auto cell = graph->zeros(keywords::shape = {dimBatch, dimState});

    // @TODO: review this
    RNNStates startStates;
    for(int i = 0; i < options_->get<size_t>("layers-dec"); ++i)
      startStates.push_back(RNNState{start, cell});

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

    auto stateS2S = std::dynamic_pointer_cast<DecoderStateS2S>(state);

    auto embeddings = stateS2S->getTargetEmbeddings();

    if(dropoutTrg) {
      int dimBatch = embeddings->shape()[0];
      int trgWords = embeddings->shape()[2];
      auto trgWordDrop = graph->dropout(dropoutTrg, {1, 1, trgWords});
      embeddings = dropout(embeddings, mask = trgWordDrop);
    }

    if(!attention_)
      attention_ = New<GlobalAttention>(prefix_,
                                        state->getEncoderState(),
                                        dimDecState,
                                        dropout_prob = dropoutRnn,
                                        normalize = layerNorm);

    if(!rnnL1)
      rnnL1 = New<RNN<CGRU>>(graph,
                              prefix_,
                              dimTrgEmb,
                              dimDecState,
                              attention_,
                              dropout_prob = dropoutRnn,
                              normalize = layerNorm);

    RNNStates statesL1 = (*rnnL1)(embeddings, stateS2S->getStates()[0]);

    bool single = stateS2S->doSingleStep();
    auto alignedContext = single ? rnnL1->getCell()->getLastContext() :
                                   rnnL1->getCell()->getContexts();

    RNNStates statesOut = statesL1;
    auto outputLn = statesOut.outputs();

    //Expr outputLn;
    //if(decoderLayers > 1) {
    //  std::vector<Expr> statesIn;
    //  for(int i = rnnL1->numStates(); i < stateS2S->getStates().size(); ++i)
    //    statesIn.push_back(stateS2S->getStates()[i]);
    //
    //  if(!rnnLn)
    //    rnnLn = New<MLRNN<GRU>>(graph,
    //                            prefix_,
    //                            decoderLayers - 1,
    //                            dimDecState,
    //                            dimDecState,
    //                            normalize = layerNorm,
    //                            dropout_prob = dropoutRnn,
    //                            skip = skipDepth,
    //                            skip_first = skipDepth);
    //
    //  std::vector<Expr> statesLn;
    //  std::tie(outputLn, statesLn) = (*rnnLn)(stateL1, statesIn);
    //
    //  statesOut.insert(statesOut.end(), statesLn.begin(), statesLn.end());
    //} else {
    //  outputLn = stateL1;
    //}

    //// 2-layer feedforward network for outputs and cost
    auto logitsL1
        = Dense(prefix_ + "_ff_logit_l1",
                dimTrgEmb,
                activation = act::tanh,
                normalize = layerNorm)(embeddings, outputLn, alignedContext);

    Expr logitsOut;
    if(options_->get<bool>("tied-embeddings")) {
      if(!tiedOutputWeights_)
        tiedOutputWeights_ = transpose(graph->get(prefix_ + "_Wemb"));

      logitsOut = DenseTied(
          prefix_ + "_ff_logit_l2", tiedOutputWeights_, dimTrgVoc)(logitsL1);
    } else
      logitsOut = Dense(prefix_ + "_ff_logit_l2", dimTrgVoc)(logitsL1);

    return New<DecoderStateS2S>(statesOut, logitsOut, state->getEncoderState());
  }

  const std::vector<Expr> getAlignments() {
    return attention_->getAlignments();
  }
};

typedef EncoderDecoder<EncoderS2S, DecoderS2S> S2S;
}
