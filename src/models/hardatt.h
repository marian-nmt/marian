#pragma once

#include "marian.h"

#include "layers/generic.h"
#include "rnn/attention_constructors.h"
#include "rnn/types.h"

#include <numeric>

namespace marian {

class DecoderStateHardAtt : public DecoderState {
protected:
  std::vector<size_t> attentionIndices_;

public:
  DecoderStateHardAtt(const rnn::States& states,
                      Expr probs,
                      std::vector<Ptr<EncoderState>>& encStates,
                      Ptr<data::CorpusBatch> batch,
                      const std::vector<size_t>& attentionIndices)
      : DecoderState(states, probs, encStates, batch),
        attentionIndices_(attentionIndices) {}

  virtual Ptr<DecoderState> select(const std::vector<size_t>& selIdx,
                                   int beamSize) {
    std::vector<size_t> selectedAttentionIndices;
    for(auto i : selIdx)
      selectedAttentionIndices.push_back(attentionIndices_[i]);

    return New<DecoderStateHardAtt>(states_.select(selIdx, beamSize),
                                    probs_,
                                    encStates_,
                                    batch_,
                                    selectedAttentionIndices);
  }

  virtual void setAttentionIndices(
      const std::vector<size_t>& attentionIndices) {
    attentionIndices_ = attentionIndices;
  }

  virtual std::vector<size_t>& getAttentionIndices() {
    ABORT_IF(attentionIndices_.empty(), "Empty attention indices");
    return attentionIndices_;
  }

  virtual void blacklist(Expr totalCosts, Ptr<data::CorpusBatch> batch) {
    auto attentionIdx = getAttentionIndices();
    int dimVoc = totalCosts->shape()[-1];
    for(int i = 0; i < attentionIdx.size(); i++) {
      if(batch->front()->data()[attentionIdx[i]] != 0) {
        totalCosts->val()->set(i * dimVoc + EOS_ID,
                               std::numeric_limits<float>::lowest());
      } else {
        totalCosts->val()->set(i * dimVoc + STP_ID,
                               std::numeric_limits<float>::lowest());
      }
    }
  }
};

class DecoderHardAtt : public DecoderBase {
protected:
  Ptr<rnn::RNN> rnn_;
  std::unordered_set<Word> specialSymbols_;

public:
  DecoderHardAtt(Ptr<Options> options) : DecoderBase(options) {
    if(options->has("special-vocab")) {
      auto spec = options->get<std::vector<size_t>>("special-vocab");
      specialSymbols_.insert(spec.begin(), spec.end());
    }
  }

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) {
    using namespace keywords;

    std::vector<Expr> meanContexts;
    for(auto& encState : encStates) {
      // average the source context weighted by the batch mask
      // this will remove padded zeros from the average
      meanContexts.push_back(weighted_average(
          encState->getContext(), encState->getMask(), axis = -3));
    }

    Expr start;
    if(!meanContexts.empty()) {
      // apply single layer network to mean to map into decoder space
      auto mlp = mlp::mlp(graph)                                     //
                     .push_back(mlp::dense(graph)                    //
                                ("prefix", prefix_ + "_ff_state")    //
                                ("dim", opt<int>("dim-rnn"))         //
                                ("activation", (int)mlp::act::tanh)  //
                                ("layer-normalization",
                                 opt<bool>("layer-normalization")));
      start = mlp->apply(meanContexts);
    }

    rnn::States startStates(opt<size_t>("dec-depth"), {start, start});
    return New<DecoderStateHardAtt>(
        startStates, nullptr, encStates, batch, std::vector<size_t>({0}));
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto type = options_->get<std::string>("type");

    int dimTrgVoc = options_->get<std::vector<int>>("dim-vocabs").back();

    int dimTrgEmb = options_->get<int>("dim-emb");

    int dimDecState = options_->get<int>("dim-rnn");
    bool layerNorm = options_->get<bool>("layer-normalization");
    bool skipDepth = options_->get<bool>("skip");

    size_t decoderLayers = options_->get<size_t>("dec-depth");
    auto cellType = options_->get<std::string>("dec-cell");

    float dropoutRnn = inference_ ? 0 : options_->get<float>("dropout-rnn");
    float dropoutTrg = inference_ ? 0 : options_->get<float>("dropout-trg");

    auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAtt>(state);

    auto trgEmbeddings = stateHardAtt->getTargetEmbeddings();

    auto context = stateHardAtt->getEncoderStates()[0]->getContext();
    int dimContext = context->shape()[-1];
    int dimSrcWords = context->shape()[-3];

    int dimBatch = context->shape()[-2];
    int dimTrgWords = trgEmbeddings->shape()[-3];
    int dimBeam = trgEmbeddings->shape()[-4];

    if(dropoutTrg) {
      trgEmbeddings
          = dropout(trgEmbeddings, dropoutTrg, {dimTrgWords, dimBatch, 1});
    }

    auto flatContext = reshape(context, {dimBatch * dimSrcWords, dimContext});
    auto attendedContext
        = rows(flatContext, stateHardAtt->getAttentionIndices());
    attendedContext = reshape(attendedContext,
                              {dimBeam, dimTrgWords, dimBatch, dimContext});

    auto rnnInputs = concatenate({trgEmbeddings, attendedContext}, axis = -1);
    int dimInput = rnnInputs->shape()[-1];

    if(!rnn_) {
      auto rnn = rnn::rnn(graph)              //
          ("type", cellType)                  //
          ("dimInput", dimInput)              //
          ("dimState", dimDecState)           //
          ("dropout", dropoutRnn)             //
          ("layer-normalization", layerNorm)  //
          ("skip", skipDepth);

      if(type == "hard-soft-att") {
        auto attCell = rnn::stacked_cell(graph)         //
                           .push_back(rnn::cell(graph)  //
                                      ("prefix", prefix_ + "_cell1"));
        for(int i = 0; i < state->getEncoderStates().size(); ++i) {
          std::string prefix = prefix_;
          if(state->getEncoderStates().size() > 1)
            prefix += "_att" + std::to_string(i + 1);

          attCell.push_back(rnn::attention(graph)  //
                            ("prefix", prefix)     //
                                .set_state(state->getEncoderStates()[i]));
        }

        attCell.push_back(rnn::cell(graph)                //
                          ("prefix", prefix_ + "_cell2")  //
                          ("final", true));
        rnn.push_back(attCell);
      } else {
        rnn.push_back(rnn::cell(graph)("prefix", prefix_));
      }

      for(int i = 0; i < decoderLayers - 1; ++i)
        rnn.push_back(rnn::cell(graph)  //
                      ("prefix", prefix_ + "_l" + std::to_string(i)));

      rnn_ = rnn.construct();
    }

    auto decContext = rnn_->transduce(rnnInputs, stateHardAtt->getStates());
    rnn::States decStates = rnn_->lastCellStates();

    //// 2-layer feedforward network for outputs and cost
    auto out = mlp::mlp(graph)
                   .push_back(mlp::dense(graph)                     //
                              ("prefix", prefix_ + "_ff_logit_l1")  //
                              ("dim", dimTrgEmb)                    //
                              ("activation", (int)mlp::act::tanh)   //
                              ("layer-normalization", layerNorm))   //
                   .push_back(mlp::dense(graph)                     //
                              ("prefix", prefix_ + "_ff_logit_l2")  //
                              ("dim", dimTrgVoc));

    Expr logits;
    if(type == "hard-soft-att") {
      std::vector<Expr> alignedContexts;
      for(int k = 0; k < state->getEncoderStates().size(); ++k) {
        // retrieve all the aligned contexts computed by the attention mechanism
        auto att = rnn_->at(0)
                       ->as<rnn::StackedCell>()
                       ->at(k + 1)
                       ->as<rnn::Attention>();
        alignedContexts.push_back(att->getContext());
      }

      Expr alignedContext;
      if(alignedContexts.size() > 1)
        alignedContext = concatenate(alignedContexts, axis = -1);
      else if(alignedContexts.size() == 1)
        alignedContext = alignedContexts[0];

      logits = out->apply(rnnInputs, decContext, alignedContext);
    } else {
      logits = out->apply(rnnInputs, decContext);
    }

    return New<DecoderStateHardAtt>(decStates,
                                    logits,
                                    stateHardAtt->getEncoderStates(),
                                    stateHardAtt->getBatch(),
                                    stateHardAtt->getAttentionIndices());
  }

  const std::vector<Expr> getAlignments() {
    auto att = rnn_->at(0)->as<rnn::StackedCell>()->at(1)->as<rnn::Attention>();
    return att->getAlignments();
  }

  void embeddingsFromBatch(Ptr<ExpressionGraph> graph,
                           Ptr<DecoderState> state,
                           Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    DecoderBase::embeddingsFromBatch(graph, state, batch);

    auto subBatch = (*batch)[batchIndex_];
    int dimBatch = subBatch->batchSize();
    int dimWords = subBatch->batchWidth();

    std::vector<size_t> attentionIndices(dimBatch, 0);
    std::vector<size_t> currentPos(dimBatch, 0);
    std::iota(currentPos.begin(), currentPos.end(), 0);

    for(int i = 0; i < dimWords - 1; ++i) {
      for(int j = 0; j < dimBatch; ++j) {
        size_t word = subBatch->data()[i * dimBatch + j];
        if(specialSymbols_.count(word))
          currentPos[j] += dimBatch;
        attentionIndices.push_back(currentPos[j]);
      }
    }

    std::dynamic_pointer_cast<DecoderStateHardAtt>(state)->setAttentionIndices(
        attentionIndices);
  }

  virtual void embeddingsFromPrediction(Ptr<ExpressionGraph> graph,
                                        Ptr<DecoderState> state,
                                        const std::vector<size_t>& embIdx,
                                        int dimBatch,
                                        int beamSize) {
    DecoderBase::embeddingsFromPrediction(graph, state, embIdx, dimBatch, beamSize);

    auto stateHardAtt = std::dynamic_pointer_cast<DecoderStateHardAtt>(state);

    int dimSrcWords = state->getEncoderStates()[0]->getContext()->shape()[-3];

    if(embIdx.empty()) {
      stateHardAtt->setAttentionIndices({0});
    } else {
      for(size_t i = 0; i < embIdx.size(); ++i)
        if(specialSymbols_.count(embIdx[i])) {
          stateHardAtt->getAttentionIndices()[i]++;
          if(stateHardAtt->getAttentionIndices()[i] >= dimSrcWords)
            stateHardAtt->getAttentionIndices()[i] = dimSrcWords - 1;
        }
    }
  }

  void clear() { rnn_ = nullptr; }
};
}
