#pragma once

#include "marian.h"
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
    return batch_->front()->data();
  }
};

class DecoderState {
protected:
  std::vector<Ptr<EncoderState>> encStates_;

  Expr targetEmbeddings_;
  Expr targetMask_;
  Expr targetIndices_;

  Expr probs_;
  rnn::States states_;
  Ptr<data::CorpusBatch> batch_;

  size_t position_{0};

public:
  DecoderState(const rnn::States& states,
               Expr probs,
               std::vector<Ptr<EncoderState>>& encStates,
               Ptr<data::CorpusBatch> batch)
      : states_(states), probs_(probs), encStates_(encStates), batch_(batch) {}

  virtual std::vector<Ptr<EncoderState>>& getEncoderStates() {
    return encStates_;
  }

  virtual Expr getProbs() { return probs_; }
  virtual void setProbs(Expr probs) { probs_ = probs; }

  virtual Ptr<DecoderState> select(const std::vector<size_t>& selIdx,
                                   int beamSize) {
    auto selectedState = New<DecoderState>(
        states_.select(selIdx, beamSize), probs_, encStates_, batch_);
    selectedState->setPosition(getPosition());
    return selectedState;
  }

  virtual const rnn::States& getStates() { return states_; }

  virtual Expr getTargetEmbeddings() { return targetEmbeddings_; };

  virtual void setTargetEmbeddings(Expr targetEmbeddings) {
    targetEmbeddings_ = targetEmbeddings;
  }

  virtual Expr getTargetIndices() { return targetIndices_; };

  virtual void setTargetIndices(Expr targetIndices) {
    targetIndices_ = targetIndices;
  }

  virtual Expr getTargetMask() { return targetMask_; };

  virtual void setTargetMask(Expr targetMask) { targetMask_ = targetMask; }

  virtual const std::vector<size_t>& getSourceWords() {
    return getEncoderStates()[0]->getSourceWords();
  }

  Ptr<data::CorpusBatch> getBatch() {
    return batch_;
  }

  size_t getPosition() { return position_; }
  void setPosition(size_t position) { position_ = position; }

  virtual void blacklist(Expr totalCosts, Ptr<data::CorpusBatch> batch) {}
};
}
