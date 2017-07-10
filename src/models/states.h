#pragma once

#include "common/definitions.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "rnn/types.h"

namespace marian {

class EncoderState {
private:
  Expr context_;
  Expr mask_;
  Ptr<data::CorpusBatch> batch_;

public:
  EncoderState(Expr context, Expr mask, Ptr<data::CorpusBatch> batch)
      : context_(context), mask_(mask), batch_(batch) {}

  EncoderState() {}

  virtual Expr getContext() { return context_; }
  virtual Expr getAttended() { return context_; }
  virtual Expr getMask() { return mask_; }

  virtual const std::vector<size_t>& getSourceWords() {
    return batch_->front()->indices();
  }
};

class DecoderState {
protected:
  Ptr<EncoderState> encState_;

  Expr targetEmbeddings_;
  Expr probs_;
  bool singleStep_{false};
  rnn::States states_;

public:
  DecoderState(const rnn::States& states,
               Expr probs,
               Ptr<EncoderState> encState)
      : states_(states), probs_(probs), encState_(encState) {}

  virtual Ptr<EncoderState> getEncoderState() { return encState_; }
  virtual Expr getProbs() { return probs_; }
  virtual void setProbs(Expr probs) { probs_ = probs; }

  virtual Ptr<DecoderState> select(const std::vector<size_t>& selIdx) {
    return New<DecoderState>(states_.select(selIdx), probs_, encState_);
  }

  virtual const rnn::States& getStates() { return states_; }

  virtual Expr getTargetEmbeddings() { return targetEmbeddings_; };

  virtual void setTargetEmbeddings(Expr targetEmbeddings) {
    targetEmbeddings_ = targetEmbeddings;
  }

  virtual bool doSingleStep() { return singleStep_; };

  virtual void setSingleStep(bool singleStep = true) {
    singleStep_ = singleStep;
  }

  virtual const std::vector<size_t>& getSourceWords() {
    return getEncoderState()->getSourceWords();
  }

  virtual void blacklist(Expr totalCosts, Ptr<data::CorpusBatch> batch) {}
};

}
