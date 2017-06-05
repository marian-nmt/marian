#pragma once

#include "models/s2s.h"

namespace marian {

class EncoderDecoderRec : public EncoderDecoder<EncoderS2S, DecoderS2S> {
private:
  Ptr<GlobalAttention> attention_;
  Ptr<RNN<CGRU>> rnnL1;

public:
  template <class... Args>
  EncoderDecoderRec(Ptr<Config> options, Args... args)
      : EncoderDecoder(options, args...) {}

  virtual std::tuple<Expr, Expr> reconstructorGroundTruth(
      Ptr<DecoderState> state,
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    int dimVoc = options_->get<std::vector<int>>("dim-vocabs").front();
    int dimEmb = options_->get<int>("dim-emb");
    int dimPos = options_->get<int>("dim-pos");

    auto rEmb = Embedding("encoder_Wemb", dimVoc, dimEmb)(graph);

    auto subBatch = batch->front();
    int dimBatch = subBatch->batchSize();
    int dimWords = subBatch->batchWidth();

    auto chosenEmbeddings = rows(rEmb, subBatch->indeces());

    auto r = reshape(chosenEmbeddings, {dimBatch, dimEmb, dimWords});

    auto rMask = graph->constant({dimBatch, 1, dimWords},
                                 init = inits::from_vector(subBatch->mask()));

    auto rIdx = graph->constant({(int)subBatch->indeces().size(), 1},
                                init = inits::from_vector(subBatch->indeces()));

    auto rShifted = shift(r, {0, 0, 1, 0});

    state->setTargetEmbeddings(rShifted);

    return std::make_tuple(rMask, rIdx);
  }

  virtual Ptr<DecoderState> reconstructorStep(Ptr<DecoderState> state) {
    using namespace keywords;

    int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").front();

    int dimTrgEmb = options_->get<int>("dim-emb");

    int dimDecState = options_->get<int>("dim-rnn");
    bool layerNorm = options_->get<bool>("layer-normalization");

    float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
    float dropoutSrc = inference_ ? 0 : options_->get<float>("dropout-trg");

    auto stateS2S = std::dynamic_pointer_cast<DecoderStateS2S>(state);

    auto embeddings = stateS2S->getTargetEmbeddings();
    auto graph = embeddings->graph();

    if(dropoutSrc) {
      int dimBatch = embeddings->shape()[0];
      int srcWords = embeddings->shape()[2];
      auto srcWordDrop = graph->dropout(dropoutSrc, {dimBatch, 1, srcWords});
      embeddings = dropout(embeddings, mask = srcWordDrop);
    }

    auto context = state->getEncoderState()->getContext();

    // std::cerr << "T:" << attention_ << std::endl;
    // @TODO: proper clearing
    // if(!attention_)
    attention_ = New<GlobalAttention>("decoder_rec",
                                      state->getEncoderState(),
                                      dimDecState,
                                      dropout_prob = dropoutRnn,
                                      normalize = layerNorm);

    // std::cerr << "T:" << attention_ << std::endl;
    // if(!rnnL1)
    rnnL1 = New<RNN<CGRU>>(graph,
                           "decoder_rec",
                           dimTrgEmb,
                           dimDecState,
                           attention_,
                           dropout_prob = dropoutRnn,
                           normalize = layerNorm);
    auto stateL1 = (*rnnL1)(embeddings, stateS2S->getStates()[0]);

    bool single = stateS2S->doSingleStep();
    auto alignedContext = single ? rnnL1->getCell()->getLastContext() :
                                   rnnL1->getCell()->getContexts();

    std::vector<Expr> statesOut;
    statesOut.push_back(stateL1);

    auto outputLn = stateL1;

    //// 2-layer feedforward network for outputs and cost
    auto logitsL1
        = Dense("ff_logit_l1_rec",
                dimTrgEmb,
                activation = act::tanh,
                normalize = layerNorm)(embeddings, outputLn, alignedContext);

    auto logitsOut = Dense("ff_logit_l2_rec", dimTrgVoc)(logitsL1);

    return New<DecoderStateS2S>(statesOut, logitsOut, state->getEncoderState());
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true) {
    using namespace keywords;

    if(clearGraph)
      clear(graph);

    auto decState = startState(graph, batch);

    Expr trgMask, trgIdx;
    std::tie(trgMask, trgIdx)
        = decoder_->groundTruth(decState, graph, batch, batchIndices_.back());

    auto nextDecState = step(decState);

    auto cost = CrossEntropyCost("cost")(
        nextDecState->getProbs(), trgIdx, mask = trgMask);

    auto stateS2S = std::dynamic_pointer_cast<DecoderStateS2S>(nextDecState);
    auto recContext = stateS2S->getStates().back();
    auto meanContext = weighted_average(recContext, trgMask, axis = 2);

    bool layerNorm = options_->get<bool>("layer-normalization");
    auto start = Dense("ff_state_rec",
                       options_->get<int>("dim-rnn"),
                       activation = act::tanh,
                       normalize = layerNorm)(meanContext);
    std::vector<Expr> startStates(1, start);

    auto recState = New<DecoderStateS2S>(
        startStates, nullptr, New<EncoderStateS2S>(recContext, trgMask, batch));
    Expr srcMask, srcIdx;
    std::tie(srcMask, srcIdx)
        = reconstructorGroundTruth(recState, graph, batch);

    auto nextRecState = reconstructorStep(recState);

    auto recCost = CrossEntropyCost("recScore")(
        nextRecState->getProbs(), srcIdx, mask = srcMask);

    auto srcA = concatenate(
        std::dynamic_pointer_cast<DecoderS2S>(decoder_)->getAlignments(),
        axis = 3);
    auto trgA = concatenate(attention_->getAlignments(), axis = 3);

    int dimBatch = srcA->shape()[0];
    int dimSrc = srcA->shape()[2];
    int dimTrg = srcA->shape()[3];
    int dimSrcTrg = dimSrc * dimTrg;

    std::vector<size_t> reorder;
    for(int j = 0; j < dimTrg; ++j)
      for(int i = 0; i < dimSrc; ++i)
        reorder.push_back(i * dimTrg + j);

    // if(dimBatch == 1) {
    //  srcA = reshape(srcA, {dimTrg, dimSrc});
    //  trgA = reshape(trgA, {dimSrc, dimTrg});
    //}
    // else {
    //  srcA = reshape(srcA, {dimSrc, dimBatch, dimTrg});
    //  trgA = reshape(trgA, {dimTrg, dimBatch, dimSrc});
    //}
    // debug(srcA, "srcA");
    // debug(trgA, "trgA");

    trgA = rows(reshape(trgA, {dimSrcTrg, dimBatch}), reorder);

    // if(dimBatch == 1)
    //  trgA = reshape(trgA, {dimTrg, dimSrc});
    // else
    //  trgA = reshape(trgA, {dimSrc, dimBatch, dimTrg});
    //
    // debug(trgA, "reordered");

    auto symmetricAlignment
        = mean(scalar_product(reshape(srcA, {dimBatch, 1, dimSrcTrg}),
                              reshape(trgA, {dimBatch, 1, dimSrcTrg}),
                              axis = 2),
               axis = 0);
    //      return cost + recCost - debug(symmetricAlignment, "trace");
    float gamma = 0.5;
    return cost + recCost - gamma * symmetricAlignment;
  }
};
}
